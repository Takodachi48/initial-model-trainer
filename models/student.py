import torch
import torch.nn as nn
import timm
from typing import Optional


class StudentModel(nn.Module):
    """
    MobileNetV3-Small student model for deployment.
    
    Chosen MobileNetV3-Small because:
    - Extremely lightweight (~2.5M parameters)
    - Optimized for mobile/CPU inference
    - Good accuracy-speed tradeoff
    - Modern architecture with squeeze-and-excitation
    """
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        model_name: str = "mobilenetv3_small_100",
        freeze_backbone: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained MobileNetV3
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
            global_pool='',  # Remove default pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
        
        # Freeze backbone if specified
        if freeze_backbone:
            self.freeze_backbone_params()
    
    def _init_classifier(self):
        """Initialize classifier layers with proper weight initialization."""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def freeze_backbone_params(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_last_blocks(self, num_blocks: int = 1):
        """
        Unfreeze the last N blocks of the backbone for fine-tuning.
        
        MobileNetV3-Small structure:
        - First stages: conv_stem, blocks 0-10
        - Last blocks: blocks 11-12 (final layers)
        """
        # Get all backbone parameters
        params = list(self.backbone.parameters())
        
        # MobileNetV3-Small has 13 blocks total (0-12)
        # We'll unfreeze the last num_blocks blocks
        total_blocks = 13
        start_block = total_blocks - num_blocks
        
        # Find the start index for unfreezing
        param_count = 0
        unfreeze_start = len(params)  # Default: don't unfreeze anything
        
        for i, block in enumerate(self.backbone.blocks):
            if i >= start_block:
                # Found the first block to unfreeze
                break
            param_count += sum(1 for _ in block.parameters())
        
        # Unfreeze parameters from the calculated start
        for i, param in enumerate(params):
            if i >= param_count:
                param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Extract features
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def get_model_info(self) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim
        }
