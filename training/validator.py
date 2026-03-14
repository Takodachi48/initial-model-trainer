import json
import os
import time
from contextlib import nullcontext
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
try:
    import matplotlib
    if hasattr(matplotlib, "use"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None

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
        writer: SummaryWriter = None,
        use_amp: bool = True,
        amp_dtype: str = "float16",
        channels_last: bool = True
    ):
        self.student_model = student_model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.distillation_loss = distillation_loss.to(device)
        self.device = device
        self.writer = writer
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.channels_last = bool(channels_last and device.type == "cuda")
        self._amp_dtype = torch.float16 if str(amp_dtype).lower() == "float16" else torch.bfloat16

    def _autocast_ctx(self):
        if self.use_amp:
            return torch.cuda.amp.autocast(dtype=self._amp_dtype)
        return nullcontext()
    
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
        
        hard_loss_fn = nn.CrossEntropyLoss()
        with torch.inference_mode():
            for images, labels in pbar:
                # Move to device
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if self.channels_last and images.dim() == 4:
                    images = images.contiguous(memory_format=torch.channels_last)
                
                # Forward pass
                with self._autocast_ctx():
                    student_logits = self.student_model(images)
                
                if use_teacher:
                    # Calculate distillation loss
                    with self._autocast_ctx():
                        teacher_logits = self.teacher_model(images)
                        total_loss, loss_dict = self.distillation_loss(
                            student_logits, teacher_logits, labels
                        )
                else:
                    # Calculate only hard loss (student standalone performance)
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
        model_name: str = "Student",
        class_names: list = None,
        label_mapping: Dict[str, int] = None,
        results_dir: str = None,
        save_confusion_matrix: bool = True
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
        y_true = []
        y_pred = []
        
        hard_loss_fn = nn.CrossEntropyLoss()
        with torch.inference_mode():
            for images, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
                # Move to device
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if self.channels_last and images.dim() == 4:
                    images = images.contiguous(memory_format=torch.channels_last)
                
                # Forward pass
                with self._autocast_ctx():
                    logits = self.student_model(images)
                
                # Calculate loss (hard loss only for final evaluation)
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

                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(predicted.detach().cpu().tolist())
        
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

        if results_dir:
            os.makedirs(results_dir, exist_ok=True)

            def _sanitize_filename(name: str) -> str:
                return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip())

            def _unique_path(path: str) -> str:
                if not os.path.exists(path):
                    return path
                base, ext = os.path.splitext(path)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                return f"{base}_{timestamp}{ext}"

            model_tag = _sanitize_filename(model_name) or "model"
            label_mapping = label_mapping or {}
            class_names = class_names or []

            num_classes = self.student_model.num_classes
            labels = list(range(num_classes))
            display_labels = [str(i) for i in labels]
            if num_classes <= 30 and len(class_names) == num_classes:
                display_labels = class_names

            report = classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=class_names if len(class_names) == num_classes else None,
                output_dict=True,
                zero_division=0
            )

            metrics_payload = {
                "model_name": model_name,
                "accuracy": final_metrics["accuracy"],
                "avg_class_accuracy": final_metrics["avg_class_accuracy"],
                "macro_precision": report.get("macro avg", {}).get("precision", 0.0),
                "macro_recall": report.get("macro avg", {}).get("recall", 0.0),
                "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
                "weighted_f1": report.get("weighted avg", {}).get("f1-score", 0.0),
                "total_samples": final_metrics.get("total_samples", len(y_true)),
                "class_accuracies": final_metrics["class_accuracies"],
                "classification_report": report,
                "class_names": class_names,
                "label_mapping": label_mapping
            }

            metrics_path = _unique_path(os.path.join(results_dir, f"{model_tag}_metrics.json"))
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, indent=2)
            print(f"Saved test metrics JSON: {metrics_path}")

            if save_confusion_matrix and plt is not None:
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                fig, ax = plt.subplots(figsize=(8, 8))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
                disp.plot(
                    include_values=False if num_classes > 30 else True,
                    cmap="Blues",
                    ax=ax,
                    xticks_rotation=45
                )
                ax.set_title(f"{model_name} Confusion Matrix")
                fig.tight_layout()

                cm_path = _unique_path(os.path.join(results_dir, f"{model_tag}_confusion_matrix.png"))
                fig.savefig(cm_path, dpi=200)
                plt.close(fig)
                print(f"Saved confusion matrix image: {cm_path}")
            elif save_confusion_matrix and plt is None:
                print("Matplotlib not available; skipping confusion matrix image output.")
        
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
        with torch.inference_mode():
            for i, (images, _) in enumerate(sample_loader):
                if i >= 2:  # Warm up with 2 batches
                    break
                images = images.to(self.device, non_blocking=True)
                if self.channels_last and images.dim() == 4:
                    images = images.contiguous(memory_format=torch.channels_last)
                with self._autocast_ctx():
                    _ = self.student_model(images)
        
        # Time inference
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        total_samples = 0
        with torch.inference_mode():
            for i, (images, _) in enumerate(sample_loader):
                if i >= num_batches:
                    break
                images = images.to(self.device, non_blocking=True)
                if self.channels_last and images.dim() == 4:
                    images = images.contiguous(memory_format=torch.channels_last)
                with self._autocast_ctx():
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
