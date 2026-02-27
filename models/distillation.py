import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss combining hard and soft targets.
    
    Total loss = α * CrossEntropy(student, labels) 
                + (1 - α) * KLDiv(student, teacher) * T²
    
    The T² factor accounts for the gradient scaling when using temperature.
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        temperature: float = 4.0,
        reduction: str = "batchmean"
    ):
        """
        Initialize distillation loss.
        
        Args:
            alpha: Weight for hard label loss (0-1)
            temperature: Temperature for soft targets
            reduction: Reduction method for KL divergence
        """
        super().__init__()
        
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
        self.class_weights = None
        
        # Loss functions (will be updated with class weights if provided)
        self.hard_loss = nn.CrossEntropyLoss(reduction="mean")
        self.soft_loss = nn.KLDivLoss(reduction=reduction, log_target=False)
    
    def set_class_weights(self, class_weights: torch.Tensor):
        """
        Set class weights for handling class imbalance.
        
        Args:
            class_weights: Weight for each class
        """
        self.class_weights = class_weights
        self.hard_loss = nn.CrossEntropyLoss(
            weight=class_weights, 
            reduction="mean"
        )
        print(f"Class weights set for distillation loss: {class_weights.cpu().numpy()}")
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate distillation loss.
        
        Args:
            student_logits: Raw logits from student model
            teacher_logits: Raw logits from teacher model  
            labels: Ground truth labels
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Hard label loss (cross-entropy)
        hard_loss = self.hard_loss(student_logits, labels)
        
        # Soft target loss (KL divergence)
        # Scale logits by temperature
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        soft_loss = self.soft_loss(student_soft, teacher_soft)
        
        # Scale soft loss by temperature squared (gradient scaling)
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # Combine losses
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        # Return loss components for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
            'alpha': self.alpha,
            'temperature': self.temperature
        }
        
        return total_loss, loss_dict
    
    def get_loss_info(self) -> Dict[str, float]:
        """Get loss configuration information."""
        return {
            'alpha': self.alpha,
            'temperature': self.temperature,
            'reduction': self.reduction
        }


class DistillationMetrics:
    """
    Metrics for knowledge distillation training.
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.correct_predictions = 0
        self.total_samples = 0
        self.running_loss = 0.0
        self.running_hard_loss = 0.0
        self.running_soft_loss = 0.0
        self.batch_count = 0
    
    def update(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        loss_dict: Dict[str, float],
        batch_size: int
    ):
        """Update metrics with batch results."""
        # Calculate accuracy
        predictions = torch.argmax(student_logits, dim=1)
        correct = (predictions == labels).sum().item()
        
        # Update counters
        self.correct_predictions += correct
        self.total_samples += batch_size
        self.batch_count += 1
        
        # Update running losses
        self.running_loss += loss_dict['total_loss']
        self.running_hard_loss += loss_dict['hard_loss']
        self.running_soft_loss += loss_dict['soft_loss']
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate and return current metrics."""
        if self.batch_count == 0:
            return {}
        
        accuracy = self.correct_predictions / self.total_samples if self.total_samples > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'total_loss': self.running_loss / self.batch_count,
            'hard_loss': self.running_hard_loss / self.batch_count,
            'soft_loss': self.running_soft_loss / self.batch_count,
            'total_samples': self.total_samples,
            'correct_predictions': self.correct_predictions
        }
