import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import WeightedRandomSampler
import os
import time
import numpy as np
from contextlib import nullcontext
from typing import Dict, Tuple, Optional
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

from models import StudentModel, TeacherModel, DistillationLoss, DistillationMetrics


class Trainer:
    """
    Knowledge distillation trainer with two-phase training strategy.
    
    Phase 1: Train classifier head only (backbone frozen)
    Phase 2: Light fine-tuning with last backbone blocks unfrozen
    """
    
    def __init__(
        self,
        student_model: StudentModel,
        teacher_model: TeacherModel,
        distillation_loss: DistillationLoss,
        device: torch.device,
        log_dir: str = "logs",
        save_dir: str = "checkpoints",
        use_class_weights: bool = False,
        use_balanced_sampler: bool = False,
        use_amp: bool = True,
        amp_dtype: str = "float16",
        channels_last: bool = True,
        grad_clip_norm: Optional[float] = None,
        use_fused_adamw: bool = True
    ):
        self.student_model = student_model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.distillation_loss = distillation_loss.to(device)
        self.device = device

        # Performance options
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.channels_last = bool(channels_last and device.type == "cuda")
        self._amp_dtype = torch.float16 if str(amp_dtype).lower() == "float16" else torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.grad_clip_norm = grad_clip_norm
        self.use_fused_adamw = bool(use_fused_adamw and device.type == "cuda")

        if self.channels_last:
            self.student_model = self.student_model.to(memory_format=torch.channels_last)
            self.teacher_model = self.teacher_model.to(memory_format=torch.channels_last)
        
        # Class imbalance handling
        self.use_class_weights = use_class_weights
        self.use_balanced_sampler = use_balanced_sampler
        self.class_weights = None
        
        # Logging
        self.log_dir = log_dir
        self.save_dir = save_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.training_phase = "phase1"
        self.label_mapping = {}
        self.class_names = []

    def _autocast_ctx(self):
        """Build AMP context manager for CUDA, else a no-op context."""
        if self.use_amp:
            return torch.cuda.amp.autocast(dtype=self._amp_dtype)
        return nullcontext()
    
    def calculate_class_weights(self, dataset) -> torch.Tensor:
        """
        Calculate class weights for handling class imbalance.
        
        Args:
            dataset: Training dataset with labels
            
        Returns:
            Class weights tensor
        """
        # Extract all labels from dataset
        if hasattr(dataset, 'labels'):
            labels = dataset.labels
        else:
            # Extract labels by iterating through dataset
            labels = []
            for _, label in dataset:
                labels.append(label)
        
        # Calculate balanced class weights
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=labels
        )
        
        # Convert to tensor
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        print(f"Class weights calculated: {class_weights.cpu().numpy()}")
        print(f"Class distribution: {np.bincount(labels)}")
        
        return class_weights
    
    def create_balanced_sampler(self, dataset) -> WeightedRandomSampler:
        """
        Create balanced sampler for handling class imbalance.
        
        Args:
            dataset: Training dataset with labels
            
        Returns:
            WeightedRandomSampler
        """
        # Extract all labels from dataset
        if hasattr(dataset, 'labels'):
            labels = dataset.labels
        else:
            # Extract labels by iterating through dataset
            labels = []
            for _, label in dataset:
                labels.append(label)
        
        # Calculate sample weights
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        sample_weights = [class_weights[label] for label in labels]
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        print(f"Balanced sampler created with {len(sample_weights)} samples")
        print(f"Class weights: {class_weights}")
        
        return sampler
    
    def setup_class_imbalance_handling(self, train_dataset):
        """
        Setup class imbalance handling based on configuration.
        
        Args:
            train_dataset: Training dataset
        """
        if self.use_class_weights:
            self.class_weights = self.calculate_class_weights(train_dataset)
            # Update distillation loss with class weights
            self.distillation_loss.set_class_weights(self.class_weights)
            print("Class weights enabled for loss calculation")
        
        if self.use_balanced_sampler:
            self.balanced_sampler = self.create_balanced_sampler(train_dataset)
            print("Balanced sampler enabled for data loading")
    
    def setup_phase(
        self,
        phase: str,
        learning_rate: float,
        weight_decay: float,
        freeze_backbone: bool,
        unfreeze_blocks: int = 1
    ) -> optim.Optimizer:
        """
        Setup training phase with appropriate optimizer and model configuration.
        
        Args:
            phase: Current training phase ("phase1" or "phase2")
            learning_rate: Learning rate for this phase
            weight_decay: Weight decay for regularization
            freeze_backbone: Whether to freeze backbone
            unfreeze_blocks: Number of backbone blocks to unfreeze (phase2 only)
            
        Returns:
            Configured optimizer
        """
        self.training_phase = phase
        
        # Configure model based on phase
        if phase == "phase1":
            # Phase 1: Freeze backbone, train classifier only
            self.student_model.freeze_backbone_params()
            print(f"Phase 1: Backbone frozen, training classifier head only")
        elif phase == "phase2":
            # Phase 2: Unfreeze last blocks for light fine-tuning
            if not freeze_backbone:
                self.student_model.unfreeze_last_blocks(unfreeze_blocks)
                print(f"Phase 2: Unfrozen last {unfreeze_blocks} backbone blocks")
            else:
                self.student_model.freeze_backbone_params()
                print(f"Phase 2: Backbone remains frozen")
        
        # Get trainable parameters
        trainable_params = [p for p in self.student_model.parameters() if p.requires_grad]
        
        # Create optimizer
        adamw_kwargs = {
            'lr': float(learning_rate),
            'weight_decay': float(weight_decay)
        }
        if self.use_fused_adamw:
            adamw_kwargs['fused'] = True

        try:
            optimizer = optim.AdamW(trainable_params, **adamw_kwargs)
        except TypeError:
            # Older PyTorch builds may not support fused AdamW.
            adamw_kwargs.pop('fused', None)
            optimizer = optim.AdamW(trainable_params, **adamw_kwargs)
        
        print(f"Phase {phase[-1]}: {len(trainable_params)} trainable parameters")
        print(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
        
        return optimizer
    
    def train_epoch(
        self,
        train_loader,
        optimizer: optim.Optimizer,
        epoch: int,
        print_every: int = 50,
        log_every: int = 50
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer for this phase
            epoch: Current epoch number
            print_every: Print frequency
            
        Returns:
            Dictionary of training metrics
        """
        self.student_model.train()
        self.teacher_model.eval()  # Teacher is always in eval mode
        
        metrics = DistillationMetrics(num_classes=self.student_model.num_classes)
        epoch_start_time = time.time()
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [{self.training_phase}]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move to device
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if self.channels_last and images.dim() == 4:
                images = images.contiguous(memory_format=torch.channels_last)
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            with torch.inference_mode():
                with self._autocast_ctx():
                    teacher_logits = self.teacher_model(images)

            with self._autocast_ctx():
                student_logits = self.student_model(images)

                # Calculate loss
                total_loss, loss_dict = self.distillation_loss(
                    student_logits, teacher_logits, labels
                )

            # Backward pass
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        max_norm=float(self.grad_clip_norm)
                    )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        max_norm=float(self.grad_clip_norm)
                    )
                optimizer.step()
            
            # Update metrics
            batch_size = images.size(0)
            metrics.update(student_logits, labels, loss_dict, batch_size)
            
            # Update progress bar
            if batch_idx % print_every == 0:
                current_metrics = metrics.get_metrics()
                pbar.set_postfix({
                    'loss': f"{current_metrics.get('total_loss', 0):.4f}",
                    'acc': f"{current_metrics.get('accuracy', 0):.4f}"
                })
            
            # Log to tensorboard
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % max(log_every, 1) == 0:
                self.writer.add_scalar('Train/BatchLoss', loss_dict['total_loss'], global_step)
                self.writer.add_scalar(
                    'Train/BatchAccuracy',
                    metrics.get_metrics().get('accuracy', 0),
                    global_step
                )
        
        # Calculate epoch metrics
        epoch_metrics = metrics.get_metrics()
        epoch_time = time.time() - epoch_start_time
        
        # Handle empty metrics case
        if not epoch_metrics:
            print(f"Warning: No training data processed in epoch {epoch}")
            return {
                'total_loss': 0.0,
                'accuracy': 0.0,
                'hard_loss': 0.0,
                'soft_loss': 0.0,
                'total_samples': 0,
                'correct_predictions': 0
            }
        
        # Log epoch metrics
        self.writer.add_scalar('Train/EpochLoss', epoch_metrics['total_loss'], epoch)
        self.writer.add_scalar('Train/EpochAccuracy', epoch_metrics['accuracy'], epoch)
        self.writer.add_scalar('Train/EpochTime', epoch_time, epoch)
        
        print(f"Epoch {epoch} [{self.training_phase}] completed in {epoch_time:.2f}s")
        print(f"Train Loss: {epoch_metrics['total_loss']:.4f}, "
              f"Train Acc: {epoch_metrics['accuracy']:.4f}")
        
        return epoch_metrics
    
    def save_checkpoint(
        self,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        phase: str = "phase1"
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'student_model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'phase': phase,
            'model_info': self.student_model.get_model_info(),
            'distillation_config': self.distillation_loss.get_loss_info(),
            'label_mapping': self.label_mapping,
            'class_names': self.class_names,
            'idx_to_class': {idx: name for name, idx in self.label_mapping.items()}
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.save_dir, f'checkpoint_epoch_{epoch}_phase_{phase}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_student_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with accuracy: {metrics.get('val_accuracy', 0):.4f}")
        
        # Save latest model
        latest_path = os.path.join(self.save_dir, 'latest_student_model.pth')
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.student_model.load_state_dict(checkpoint['student_model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.training_phase = checkpoint.get('phase', 'phase1')
        self.label_mapping = checkpoint.get('label_mapping', self.label_mapping)
        self.class_names = checkpoint.get('class_names', self.class_names)
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} ({self.training_phase})")
        return checkpoint
    
    def get_model_summary(self) -> str:
        """Get model training summary."""
        student_info = self.student_model.get_model_info()
        teacher_info = self.teacher_model.get_model_info()
        
        summary = f"""
Training Summary:
- Student Model: {student_info['model_name']}
  - Total Parameters: {student_info['total_parameters']:,}
  - Trainable Parameters: {student_info['trainable_parameters']:,}
  - Classes: {student_info['num_classes']}

- Teacher Model: {teacher_info['model_name']}
  - Total Parameters: {teacher_info['total_parameters']:,}
  - Trainable Parameters: {teacher_info['trainable_parameters']:,}

- Distillation Config:
  - Alpha: {self.distillation_loss.alpha}
  - Temperature: {self.distillation_loss.temperature}
  - Current Phase: {self.training_phase}
        """
        
        return summary.strip()

    def set_dataset_metadata(self, dataset) -> None:
        """Store dataset label mapping metadata for checkpoint export."""
        mapping = getattr(dataset, 'class_to_idx', None)
        classes = getattr(dataset, 'classes', None)

        if isinstance(mapping, dict):
            # Keep string keys for stable YAML/JSON interoperability.
            self.label_mapping = {str(k): int(v) for k, v in mapping.items()}
        else:
            self.label_mapping = {}

        if isinstance(classes, list):
            self.class_names = [str(name) for name in classes]
        elif self.label_mapping:
            # Derive class names ordered by class index.
            self.class_names = [
                name for name, _ in sorted(self.label_mapping.items(), key=lambda kv: kv[1])
            ]
        else:
            self.class_names = []
