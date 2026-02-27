import torch
import torch.nn as nn
import timm
from typing import Optional


class TeacherModel(nn.Module):
    """
    EfficientNet-B0 teacher model for knowledge distillation.
    
    Chosen EfficientNet-B0 because:
    - Strong baseline accuracy (~77% on ImageNet)
    - Efficient architecture with compound scaling
    - Good teacher for smaller student models
    - Provides rich soft targets for distillation
    """
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        model_name: str = "efficientnet_b0",
        freeze_all: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained EfficientNet
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Freeze all parameters if specified (teacher should not be updated)
        if freeze_all:
            self.freeze_all_params()
    
    def freeze_all_params(self):
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def get_model_info(self) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes
        }
    
    @torch.no_grad()
    def get_soft_targets(
        self, 
        x: torch.Tensor, 
        temperature: float = 4.0
    ) -> torch.Tensor:
        """
        Get soft targets for knowledge distillation.
        
        Args:
            x: Input images
            temperature: Temperature for softening logits
            
        Returns:
            Softened probability distribution
        """
        logits = self.forward(x)
        soft_targets = torch.softmax(logits / temperature, dim=1)
        return soft_targets
