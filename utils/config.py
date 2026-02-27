import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration class for training pipeline."""
    
    # Model configuration
    model: Dict[str, Any] = field(default_factory=dict)
    
    # Distillation configuration
    distillation: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    training: Dict[str, Any] = field(default_factory=dict)
    
    # Data configuration
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Logging configuration
    logging: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        return cls(
            model=config_dict.get('model', {}),
            distillation=config_dict.get('distillation', {}),
            training=config_dict.get('training', {}),
            data=config_dict.get('data', {}),
            logging=config_dict.get('logging', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            'model': self.model,
            'distillation': self.distillation,
            'training': self.training,
            'data': self.data,
            'logging': self.logging
        }
    
    def validate(self) -> bool:
        """Validate configuration."""
        required_keys = [
            'model.student.num_classes',
            'model.teacher.num_classes',
            'data.train_dir',
            'data.val_dir'
        ]
        
        for key in required_keys:
            if not self._get_nested_value(key):
                raise ValueError(f"Missing required configuration: {key}")
        
        # Validate model classes match
        student_classes = self._get_nested_value('model.student.num_classes')
        teacher_classes = self._get_nested_value('model.teacher.num_classes')
        if student_classes != teacher_classes:
            raise ValueError("Student and teacher model num_classes must match")
        
        # Validate distillation parameters
        alpha = self._get_nested_value('distillation.alpha', 0.7)
        temperature = self._get_nested_value('distillation.temperature', 4.0)
        
        if not 0 <= alpha <= 1:
            raise ValueError("Distillation alpha must be between 0 and 1")
        if temperature <= 0:
            raise ValueError("Distillation temperature must be positive")
        
        return True
    
    def _get_nested_value(self, key_path: str, default: Any = None) -> Any:
        """Get nested value from config using dot notation."""
        keys = key_path.split('.')
        value = self.to_dict()
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for key_path, value in updates.items():
            keys = key_path.split('.')
            config_dict = self.to_dict()
            
            # Navigate to the parent of the target key
            current = config_dict
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the final value
            current[keys[-1]] = value
        
        # Update this config object
        updated_config = Config.from_dict(config_dict)
        self.__dict__.update(updated_config.__dict__)


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config.from_dict(config_dict)
    config.validate()
    
    return config


def save_config(config: Config, config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object to save
        config_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)


def get_default_config() -> Config:
    """Get default configuration."""
    default_config_dict = {
        'model': {
            'student': {
                'name': 'mobilenetv3_small_100',
                'pretrained': True,
                'num_classes': 10,
                'freeze_backbone': True,
                'unfreeze_last_blocks': 1
            },
            'teacher': {
                'name': 'efficientnet_b0',
                'pretrained': True,
                'num_classes': 10,
                'freeze_all': True
            }
        },
        'distillation': {
            'alpha': 0.7,
            'temperature': 4.0
        },
        'training': {
            'phases': {
                'phase1': {
                    'epochs': 10,
                    'learning_rate': 0.001,
                    'freeze_backbone': True,
                    'weight_decay': 1e-4
                },
                'phase2': {
                    'epochs': 5,
                    'learning_rate': 0.0001,
                    'freeze_backbone': False,
                    'weight_decay': 1e-5
                }
            },
            'batch_size': 32,
            'num_workers': 4,
            'device': 'auto',
            'save_dir': 'checkpoints',
            'save_every': 5,
            'best_model_metric': 'val_accuracy'
        },
        'data': {
            'train_dir': 'data/train',
            'val_dir': 'data/val',
            'test_dir': 'data/test',
            'image_size': 224,
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'augmentations': {
                'horizontal_flip': 0.5,
                'rotation_degrees': 10,
                'color_jitter': {
                    'brightness': 0.1,
                    'contrast': 0.1,
                    'saturation': 0.1,
                    'hue': 0.05
                }
            }
        },
        'logging': {
            'log_dir': 'logs',
            'tensorboard': True,
            'print_every': 50
        }
    }
    
    return Config.from_dict(default_config_dict)
