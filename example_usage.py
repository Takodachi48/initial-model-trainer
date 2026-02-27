#!/usr/bin/env python3
"""
Example usage script for the CNN model trainer.

This script demonstrates how to use the training pipeline programmatically
and provides examples of common workflows.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import StudentModel, TeacherModel, DistillationLoss
from data import HerbDataset, create_data_loaders
from training import Trainer, Validator
from utils import load_config, get_device, Config


def create_sample_dataset(output_dir: str = "sample_data", num_classes: int = 3, samples_per_class: int = 20):
    """
    Create a sample dataset for testing purposes.
    
    Args:
        output_dir: Directory to save sample data
        num_classes: Number of classes to create
        samples_per_class: Number of samples per class
    """
    print(f"Creating sample dataset in {output_dir}...")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    
    splits = ['train', 'val', 'test']
    class_names = [f'herb_{i}' for i in range(num_classes)]
    
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create sample images
            for i in range(samples_per_class):
                # Create random RGB image
                img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                image = Image.fromarray(img_array)
                
                img_path = os.path.join(class_dir, f'sample_{i:03d}.jpg')
                image.save(img_path)
    
    print(f"Sample dataset created with {num_classes} classes, {samples_per_class} samples per class")
    return output_dir, class_names


def example_basic_training():
    """Example: Basic training with default configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Training")
    print("="*60)
    
    # Create sample dataset
    data_dir, class_names = create_sample_dataset()
    
    # Load default configuration
    config = load_config("config/default.yaml")
    
    # Update configuration for sample data
    config.update({
        'data.train_dir': os.path.join(data_dir, 'train'),
        'data.val_dir': os.path.join(data_dir, 'val'),
        'data.test_dir': os.path.join(data_dir, 'test'),
        'model.student.num_classes': len(class_names),
        'model.teacher.num_classes': len(class_names),
        'training.phases.phase1.epochs': 2,  # Reduce for demo
        'training.phases.phase2.epochs': 1,  # Reduce for demo
        'training.batch_size': 8,  # Small batch for demo
        'training.save_every': 1,  # Save every epoch for demo
    })
    
    # Setup device
    device = get_device('cpu')  # Use CPU for demo
    
    # Create models
    student_model = StudentModel(
        num_classes=len(class_names),
        pretrained=True,
        model_name="mobilenetv3_small_100",
        freeze_backbone=True
    )
    
    teacher_model = TeacherModel(
        num_classes=len(class_names),
        pretrained=True,
        model_name="efficientnet_b0",
        freeze_all=True
    )
    
    # Create distillation loss
    distillation_loss = DistillationLoss(alpha=0.7, temperature=4.0)
    
    # Create data loaders
    data_loaders, datasets = create_data_loaders(
        train_dir=config.data['train_dir'],
        val_dir=config.data['val_dir'],
        test_dir=config.data['test_dir'],
        batch_size=config.training['batch_size'],
        num_workers=0,  # Use 0 for demo to avoid multiprocessing issues
        image_size=config.data['image_size']
    )
    
    # Create trainer and validator
    trainer = Trainer(
        student_model=student_model,
        teacher_model=teacher_model,
        distillation_loss=distillation_loss,
        device=device,
        log_dir="demo_logs",
        save_dir="demo_checkpoints"
    )
    
    validator = Validator(
        student_model=student_model,
        teacher_model=teacher_model,
        distillation_loss=distillation_loss,
        device=device
    )
    
    print("Starting demo training...")
    print(trainer.get_model_summary())
    
    # Phase 1: Train classifier head only
    optimizer = trainer.setup_phase(
        phase="phase1",
        learning_rate=0.001,
        weight_decay=1e-4,
        freeze_backbone=True
    )
    
    # Train for a few epochs
    for epoch in range(2):
        print(f"\nEpoch {epoch} (Phase 1)")
        
        # Training
        train_metrics = trainer.train_epoch(
            train_loader=data_loaders['train'],
            optimizer=optimizer,
            epoch=epoch,
            print_every=1
        )
        
        # Validation
        val_metrics = validator.validate_epoch(
            val_loader=data_loaders['val'],
            epoch=epoch,
            use_teacher=True
        )
    
    print("Demo training completed!")
    print(f"Demo logs saved in: demo_logs")
    print(f"Demo checkpoints saved in: demo_checkpoints")


def example_inference():
    """Example: Model inference on sample images."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Model Inference")
    print("="*60)
    
    # Load trained model (if available)
    checkpoint_path = "demo_checkpoints/best_student_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print("No trained model found. Run example_basic_training() first.")
        return
    
    # Setup device
    device = get_device('cpu')
    
    # Load configuration
    config = load_config("config/default.yaml")
    
    # Create student model
    student_model = StudentModel(
        num_classes=3,  # Match the demo dataset
        pretrained=False,  # Don't need pretrained weights for inference
        model_name="mobilenetv3_small_100"
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student_model.load_state_dict(checkpoint['student_model_state_dict'])
    student_model.eval()
    
    # Create sample image for inference
    sample_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(sample_image)
    
    # Preprocess image
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        logits = student_model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Class probabilities: {probabilities.cpu().numpy().flatten()}")


def example_custom_configuration():
    """Example: Using custom configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Configuration")
    print("="*60)
    
    # Create custom configuration
    custom_config = Config.from_dict({
        'model': {
            'student': {
                'name': 'mobilenetv3_small_100',
                'pretrained': True,
                'num_classes': 5,
                'freeze_backbone': True
            },
            'teacher': {
                'name': 'efficientnet_b0',
                'pretrained': True,
                'num_classes': 5,
                'freeze_all': True
            }
        },
        'distillation': {
            'alpha': 0.8,  # More weight to hard labels
            'temperature': 3.0  # Lower temperature
        },
        'training': {
            'phases': {
                'phase1': {
                    'epochs': 5,
                    'learning_rate': 0.0005,
                    'freeze_backbone': True,
                    'weight_decay': 1e-4
                },
                'phase2': {
                    'epochs': 3,
                    'learning_rate': 0.00005,
                    'freeze_backbone': False,
                    'weight_decay': 1e-5
                }
            },
            'batch_size': 16,
            'num_workers': 2,
            'device': 'cpu',
            'save_dir': 'custom_checkpoints'
        },
        'data': {
            'image_size': 256,  # Larger image size
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'augmentations': {
                'horizontal_flip': 0.5,
                'rotation_degrees': 15,
                'color_jitter': {
                    'brightness': 0.2,
                    'contrast': 0.2,
                    'saturation': 0.2,
                    'hue': 0.1
                }
            }
        },
        'logging': {
            'log_dir': 'custom_logs',
            'tensorboard': True,
            'print_every': 25
        }
    })
    
    # Validate configuration
    try:
        custom_config.validate()
        print("Custom configuration is valid!")
        
        # Print configuration summary
        print("\nCustom Configuration Summary:")
        print(f"  Student model: {custom_config.model['student']['name']}")
        print(f"  Teacher model: {custom_config.model['teacher']['name']}")
        print(f"  Number of classes: {custom_config.model['student']['num_classes']}")
        print(f"  Distillation alpha: {custom_config.distillation['alpha']}")
        print(f"  Distillation temperature: {custom_config.distillation['temperature']}")
        print(f"  Image size: {custom_config.data['image_size']}")
        print(f"  Batch size: {custom_config.training['batch_size']}")
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")


def example_model_comparison():
    """Example: Compare different model architectures."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Model Architecture Comparison")
    print("="*60)
    
    device = get_device('cpu')
    
    # Different student model options
    student_models = [
        ('mobilenetv3_small_100', 'MobileNetV3-Small'),
        ('mobilenetv3_large_100', 'MobileNetV3-Large'),
        ('efficientnet_b0', 'EfficientNet-B0'),
    ]
    
    print("Comparing different student model architectures:")
    print("-" * 60)
    
    for model_name, display_name in student_models:
        try:
            # Create model
            student_model = StudentModel(
                num_classes=10,
                pretrained=True,
                model_name=model_name,
                freeze_backbone=True
            )
            
            # Get model info
            info = student_model.get_model_info()
            
            print(f"{display_name}:")
            print(f"  Total parameters: {info['total_parameters']:,}")
            print(f"  Trainable parameters: {info['trainable_parameters']:,}")
            print(f"  Parameter ratio: {info['trainable_parameters']/info['total_parameters']:.4f}")
            
            # Estimate model size (assuming float32)
            model_size_mb = info['total_parameters'] * 4 / (1024 * 1024)
            print(f"  Estimated model size: {model_size_mb:.2f} MB")
            print()
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    
    print("Teacher model (for reference):")
    teacher_model = TeacherModel(
        num_classes=10,
        pretrained=True,
        model_name="efficientnet_b0",
        freeze_all=True
    )
    
    teacher_info = teacher_model.get_model_info()
    teacher_size_mb = teacher_info['total_parameters'] * 4 / (1024 * 1024)
    
    print(f"  Total parameters: {teacher_info['total_parameters']:,}")
    print(f"  Estimated model size: {teacher_size_mb:.2f} MB")


def main():
    """Run all examples."""
    print("CNN Model Trainer - Example Usage")
    print("=" * 60)
    
    try:
        # Example 1: Basic training
        example_basic_training()
        
        # Example 2: Inference
        example_inference()
        
        # Example 3: Custom configuration
        example_custom_configuration()
        
        # Example 4: Model comparison
        example_model_comparison()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
