import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
from typing import Dict, Tuple
from tqdm import tqdm

from models import StudentModel, TeacherModel, DistillationLoss, DistillationMetrics


class Validator:
    """
    Validation loop for knowledge distillation training.
    
    Evaluates student model performance with and without teacher guidance.
    """
    
    def __init__(
        self,
        student_model: StudentModel,
        teacher_model: TeacherModel,
        distillation_loss: DistillationLoss,
        device: torch.device,
        writer: SummaryWriter = None
    ):
        self.student_model = student_model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.distillation_loss = distillation_loss.to(device)
        self.device = device
        self.writer = writer
    
    def validate_epoch(
        self,
        val_loader,
        epoch: int,
        use_teacher: bool = True
    ) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            use_teacher: Whether to use teacher for distillation loss
            
        Returns:
            Dictionary of validation metrics
        """
        self.student_model.eval()
        self.teacher_model.eval()
        
        metrics = DistillationMetrics(num_classes=self.student_model.num_classes)
        epoch_start_time = time.time()
        
        # Progress bar
        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
        
        with torch.no_grad():
            for images, labels in pbar:
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                student_logits = self.student_model(images)
                
                if use_teacher:
                    # Calculate distillation loss
                    teacher_logits = self.teacher_model(images)
                    total_loss, loss_dict = self.distillation_loss(
                        student_logits, teacher_logits, labels
                    )
                else:
                    # Calculate only hard loss (student standalone performance)
                    hard_loss_fn = nn.CrossEntropyLoss()
                    total_loss = hard_loss_fn(student_logits, labels)
                    loss_dict = {
                        'total_loss': total_loss.item(),
                        'hard_loss': total_loss.item(),
                        'soft_loss': 0.0,
                        'alpha': 1.0,
                        'temperature': 1.0
                    }
                
                # Update metrics
                batch_size = images.size(0)
                metrics.update(student_logits, labels, loss_dict, batch_size)
                
                # Update progress bar
                current_metrics = metrics.get_metrics()
                pbar.set_postfix({
                    'loss': f"{current_metrics.get('total_loss', 0):.4f}",
                    'acc': f"{current_metrics.get('accuracy', 0):.4f}"
                })
        
        # Calculate epoch metrics
        epoch_metrics = metrics.get_metrics()
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        if self.writer:
            prefix = "Val/WithTeacher" if use_teacher else "Val/Standalone"
            self.writer.add_scalar(f'{prefix}/Loss', epoch_metrics['total_loss'], epoch)
            self.writer.add_scalar(f'{prefix}/Accuracy', epoch_metrics['accuracy'], epoch)
            self.writer.add_scalar(f'{prefix}/Time', epoch_time, epoch)
        
        mode_str = "with teacher" if use_teacher else "standalone"
        print(f"Validation {mode_str} completed in {epoch_time:.2f}s")
        print(f"Val Loss: {epoch_metrics['total_loss']:.4f}, "
              f"Val Acc: {epoch_metrics['accuracy']:.4f}")
        
        return epoch_metrics
    
    def validate_comprehensive(
        self,
        val_loader,
        epoch: int
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Comprehensive validation with and without teacher.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Tuple of (with_teacher_metrics, standalone_metrics)
        """
        print("Running comprehensive validation...")
        
        # Validate with teacher (distillation loss)
        with_teacher_metrics = self.validate_epoch(val_loader, epoch, use_teacher=True)
        
        # Validate standalone (student only)
        standalone_metrics = self.validate_epoch(val_loader, epoch, use_teacher=False)
        
        # Calculate performance gap
        performance_gap = with_teacher_metrics['accuracy'] - standalone_metrics['accuracy']
        
        print(f"Performance gap (with teacher - standalone): {performance_gap:+.4f}")
        
        # Log comprehensive metrics
        if self.writer:
            self.writer.add_scalar('Val/PerformanceGap', performance_gap, epoch)
            self.writer.add_scalar('Val/TeacherBenefit', 
                                  performance_gap / standalone_metrics['accuracy'], epoch)
        
        return with_teacher_metrics, standalone_metrics
    
    def evaluate_model(
        self,
        test_loader,
        model_name: str = "Student"
    ) -> Dict[str, float]:
        """
        Final model evaluation on test set.
        
        Args:
            test_loader: Test data loader
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary of test metrics
        """
        print(f"Evaluating {model_name} on test set...")
        
        self.student_model.eval()
        
        metrics = DistillationMetrics(num_classes=self.student_model.num_classes)
        
        # Additional metrics for detailed evaluation
        class_correct = [0] * self.student_model.num_classes
        class_total = [0] * self.student_model.num_classes
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.student_model(images)
                
                # Calculate loss (hard loss only for final evaluation)
                hard_loss_fn = nn.CrossEntropyLoss()
                total_loss = hard_loss_fn(logits, labels)
                loss_dict = {
                    'total_loss': total_loss.item(),
                    'hard_loss': total_loss.item(),
                    'soft_loss': 0.0,
                    'alpha': 1.0,
                    'temperature': 1.0
                }
                
                # Update metrics
                batch_size = images.size(0)
                metrics.update(logits, labels, loss_dict, batch_size)
                
                # Per-class accuracy
                _, predicted = torch.max(logits, 1)
                c = (predicted == labels).squeeze()
                for i in range(batch_size):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        # Calculate final metrics
        final_metrics = metrics.get_metrics()
        
        # Add per-class accuracy
        class_accuracies = []
        for i in range(self.student_model.num_classes):
            if class_total[i] > 0:
                class_acc = class_correct[i] / class_total[i]
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)
        
        final_metrics['class_accuracies'] = class_accuracies
        final_metrics['avg_class_accuracy'] = sum(class_accuracies) / len(class_accuracies)
        
        # Print detailed results
        print(f"\n{model_name} Test Results:")
        print(f"Overall Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Overall Loss: {final_metrics['total_loss']:.4f}")
        print(f"Average Class Accuracy: {final_metrics['avg_class_accuracy']:.4f}")
        print(f"Total Samples: {final_metrics['total_samples']}")
        
        return final_metrics
    
    def generate_inference_stats(
        self,
        sample_loader,
        num_batches: int = 10
    ) -> Dict[str, float]:
        """
        Generate inference statistics (speed, memory usage).
        
        Args:
            sample_loader: Sample data loader for timing
            num_batches: Number of batches to time
            
        Returns:
            Dictionary of inference statistics
        """
        print("Generating inference statistics...")
        
        self.student_model.eval()
        
        # Warm up
        with torch.no_grad():
            for i, (images, _) in enumerate(sample_loader):
                if i >= 2:  # Warm up with 2 batches
                    break
                images = images.to(self.device)
                _ = self.student_model(images)
        
        # Time inference
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        total_samples = 0
        with torch.no_grad():
            for i, (images, _) in enumerate(sample_loader):
                if i >= num_batches:
                    break
                images = images.to(self.device)
                _ = self.student_model(images)
                total_samples += images.size(0)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        # Calculate statistics
        total_time = end_time - start_time
        avg_time_per_batch = total_time / num_batches
        avg_time_per_sample = total_time / total_samples
        samples_per_second = total_samples / total_time
        
        stats = {
            'total_time': total_time,
            'avg_time_per_batch': avg_time_per_batch,
            'avg_time_per_sample': avg_time_per_sample,
            'samples_per_second': samples_per_second,
            'total_samples_timed': total_samples
        }
        
        print(f"Inference Statistics:")
        print(f"  Samples per second: {samples_per_second:.2f}")
        print(f"  Time per sample: {avg_time_per_sample*1000:.2f} ms")
        print(f"  Time per batch: {avg_time_per_batch*1000:.2f} ms")
        
        return stats
