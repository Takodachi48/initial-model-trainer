#!/usr/bin/env python3
"""
Main training script for CNN model trainer with knowledge distillation.

Usage:
    python train.py --config config/default.yaml
    python train.py --config config/default.yaml --resume checkpoints/latest_student_model.pth
    python train.py --config config/default.yaml --data.train_dir custom/train --data.val_dir custom/val
"""

import argparse
import os
import sys
import time
from typing import Dict, Any

# Disable TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import StudentModel, TeacherModel, DistillationLoss
from data import create_data_loaders
from training import Trainer, Validator
from utils import load_config, get_device, clear_gpu_cache


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CNN model with knowledge distillation")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--override", 
        type=str, 
        nargs='*',
        default=[],
        help="Override configuration parameters (e.g., training.batch_size 64)"
    )
    
    parser.add_argument(
        "--data_only", 
        action="store_true",
        help="Only test data loading, don't train"
    )
    
    parser.add_argument(
        "--test_only", 
        action="store_true",
        help="Only run test evaluation, requires trained model"
    )
    
    return parser.parse_args()


def create_models(config) -> tuple:
    """Create student and teacher models."""
    print("Creating models...")
    
    # Student model
    student_config = config.model['student']
    student_model = StudentModel(
        num_classes=student_config['num_classes'],
        pretrained=student_config.get('pretrained', True),
        model_name=student_config.get('name', 'mobilenetv3_small_100'),
        freeze_backbone=student_config.get('freeze_backbone', True)
    )
    
    # Teacher model
    teacher_config = config.model['teacher']
    teacher_model = TeacherModel(
        num_classes=teacher_config['num_classes'],
        pretrained=teacher_config.get('pretrained', True),
        model_name=teacher_config.get('name', 'efficientnet_b0'),
        freeze_all=teacher_config.get('freeze_all', True)
    )
    
    print(f"Student model: {student_model.get_model_info()}")
    print(f"Teacher model: {teacher_model.get_model_info()}")
    
    return student_model, teacher_model


def load_teacher_checkpoint(teacher_model: TeacherModel, checkpoint_path: str, device: torch.device) -> None:
    """Load teacher model weights from checkpoint."""
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint.get('teacher_model_state_dict')
        or checkpoint.get('model_state_dict')
        or checkpoint.get('state_dict')
    )

    if state_dict is None:
        raise KeyError(
            f"Checkpoint does not contain teacher weights: {checkpoint_path}"
        )

    teacher_model.load_state_dict(state_dict)
    # Teacher should remain fixed during distillation.
    teacher_model.freeze_all_params()
    print(f"Loaded teacher checkpoint: {checkpoint_path}")


def create_distillation_loss(config) -> DistillationLoss:
    """Create distillation loss function."""
    distill_config = config.distillation
    
    distillation_loss = DistillationLoss(
        alpha=distill_config.get('alpha', 0.7),
        temperature=distill_config.get('temperature', 4.0)
    )
    
    print(f"Distillation loss config: {distillation_loss.get_loss_info()}")
    
    return distillation_loss


def create_data_loaders_from_config(config):
    """Create data loaders from configuration."""
    print("Creating data loaders...")
    
    data_config = config.data
    training_config = config.training
    
    # Create data loaders
    data_loaders, datasets = create_data_loaders(
        train_dir=data_config['train_dir'],
        val_dir=data_config['val_dir'],
        test_dir=data_config.get('test_dir'),
        batch_size=training_config['batch_size'],
        num_workers=training_config.get('num_workers', 4),
        image_size=data_config.get('image_size', 224),
        normalize_mean=data_config.get('normalize', {}).get('mean'),
        normalize_std=data_config.get('normalize', {}).get('std'),
        augmentations=data_config.get('augmentations', {})
    )
    
    # Print dataset info
    for split, dataset in datasets.items():
        print(f"{split.capitalize()} dataset: {len(dataset)} samples")
        if hasattr(dataset, 'get_class_distribution'):
            distribution = dataset.get_class_distribution()
            print(f"  Classes: {len(distribution)}")
            print(f"  Distribution: {distribution}")
    
    return data_loaders, datasets


def apply_config_overrides(config, overrides):
    """Apply configuration overrides from command line."""
    if not overrides:
        return config
    
    print("Applying configuration overrides...")
    
    overrides_dict = {}
    for i in range(0, len(overrides), 2):
        if i + 1 < len(overrides):
            key = overrides[i]
            value = overrides[i + 1]
            
            # Try to parse value as appropriate type
            try:
                # Try numeric
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Keep as string
                pass
            
            overrides_dict[key] = value
            print(f"  {key}: {value}")
    
    config.update(overrides_dict)
    return config


def train_model(config, resume_path=None):
    """Main training function."""
    print("Starting training...")
    
    # Setup device
    device = get_device(config.training.get('device', 'auto'))
    
    # Create models
    student_model, teacher_model = create_models(config)
    teacher_checkpoint = config.model.get('teacher', {}).get('checkpoint')
    load_teacher_checkpoint(teacher_model, teacher_checkpoint, device)
    
    # Create distillation loss
    distillation_loss = create_distillation_loss(config)
    
    # Create data loaders
    data_loaders, datasets = create_data_loaders_from_config(config)
    
    # Create trainer and validator
    class_imbalance_config = config.training.get('class_imbalance', {})
    
    trainer = Trainer(
        student_model=student_model,
        teacher_model=teacher_model,
        distillation_loss=distillation_loss,
        device=device,
        log_dir=config.logging.get('log_dir', 'logs'),
        save_dir=config.training.get('save_dir', 'checkpoints'),
        use_class_weights=class_imbalance_config.get('use_class_weights', False),
        use_balanced_sampler=class_imbalance_config.get('use_balanced_sampler', False)
    )
    
    # Setup class imbalance handling if needed
    if (class_imbalance_config.get('use_class_weights', False) or 
        class_imbalance_config.get('use_balanced_sampler', False)):
        trainer.setup_class_imbalance_handling(datasets['train'])
        
        # Recreate data loaders with balanced sampler if needed
        if class_imbalance_config.get('use_balanced_sampler', False):
            data_loaders, _ = create_data_loaders(
                train_dir=config.data['train_dir'],
                val_dir=config.data['val_dir'],
                test_dir=config.data.get('test_dir'),
                batch_size=config.training['batch_size'],
                num_workers=config.training.get('num_workers', 4),
                image_size=config.data.get('image_size', 224),
                normalize_mean=config.data.get('normalize', {}).get('mean'),
                normalize_std=config.data.get('normalize', {}).get('std'),
                augmentations=config.data.get('augmentations', {}),
                balanced_sampler=trainer.balanced_sampler
            )
    
    validator = Validator(
        student_model=student_model,
        teacher_model=teacher_model,
        distillation_loss=distillation_loss,
        device=device,
        writer=trainer.writer
    )
    
    # Print training summary
    print(trainer.get_model_summary())
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        checkpoint = trainer.load_checkpoint(resume_path)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    
    # Training phases
    phases_config = config.training['phases']
    
    for phase_name, phase_config in phases_config.items():
        print(f"\n{'='*50}")
        print(f"Starting {phase_name.upper()}")
        print(f"{'='*50}")
        
        # Setup phase
        optimizer = trainer.setup_phase(
            phase=phase_name,
            learning_rate=phase_config['learning_rate'],
            weight_decay=phase_config['weight_decay'],
            freeze_backbone=phase_config.get('freeze_backbone', True),
            unfreeze_blocks=config.model['student'].get('unfreeze_last_blocks', 1)
        )
        
        # Train for specified epochs
        phase_epochs = phase_config['epochs']
        save_every = config.training.get('save_every', 5)
        
        for epoch in range(start_epoch, start_epoch + phase_epochs):
            print(f"\nEpoch {epoch} ({phase_name})")
            print("-" * 30)
            
            # Training
            train_metrics = trainer.train_epoch(
                train_loader=data_loaders['train'],
                optimizer=optimizer,
                epoch=epoch,
                print_every=config.logging.get('print_every', 50)
            )
            
            # Validation
            with_teacher_metrics, standalone_metrics = validator.validate_comprehensive(
                val_loader=data_loaders['val'],
                epoch=epoch
            )
            
            # Combine metrics for checkpointing
            combined_metrics = {
                **train_metrics,
                'val_accuracy': with_teacher_metrics['accuracy'],
                'val_loss': with_teacher_metrics['total_loss'],
                'val_accuracy_standalone': standalone_metrics['accuracy']
            }
            
            # Check if this is the best model
            is_best = combined_metrics['val_accuracy'] > trainer.best_val_accuracy
            if is_best:
                trainer.best_val_accuracy = combined_metrics['val_accuracy']
            
            # Save checkpoint
            if (epoch % save_every == 0) or (epoch == start_epoch + phase_epochs - 1) or is_best:
                trainer.save_checkpoint(
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=combined_metrics,
                    is_best=is_best,
                    phase=phase_name
                )
            
            # Clear GPU cache
            if device.type == 'cuda':
                clear_gpu_cache()
        
        # Update start_epoch for next phase
        start_epoch = start_epoch + phase_epochs
    
    # Final evaluation on test set if available
    if 'test' in data_loaders:
        print(f"\n{'='*50}")
        print("FINAL TEST EVALUATION")
        print(f"{'='*50}")
        
        # Load best model for final evaluation
        best_model_path = os.path.join(trainer.save_dir, 'best_student_model.pth')
        if os.path.exists(best_model_path):
            trainer.load_checkpoint(best_model_path)
        
        test_metrics = validator.evaluate_model(
            test_loader=data_loaders['test'],
            model_name="Best Student Model"
        )
        
        # Generate inference statistics
        inference_stats = validator.generate_inference_stats(
            sample_loader=data_loaders['val'],
            num_batches=10
        )
        
        print(f"\nFinal Results:")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Inference Speed: {inference_stats['samples_per_second']:.2f} samples/sec")
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {trainer.best_val_accuracy:.4f}")
    print(f"Models saved in: {trainer.save_dir}")
    print(f"Logs saved in: {trainer.log_dir}")


def test_data_loading(config):
    """Test data loading only."""
    print("Testing data loading...")
    
    try:
        data_loaders, datasets = create_data_loaders_from_config(config)
        
        print("Data loading test successful!")
        
        # Test a few batches
        for split, loader in data_loaders.items():
            print(f"\nTesting {split} loader...")
            
            for i, (images, labels) in enumerate(loader):
                print(f"  Batch {i}: images {images.shape}, labels {labels.shape}")
                if i >= 2:  # Test first 3 batches
                    break
        
        print("\nData loading test completed successfully!")
        
    except Exception as e:
        print(f"Data loading test failed: {e}")
        raise


def test_model_only(config):
    """Test trained model only."""
    print("Testing trained model...")
    
    # Setup device
    device = get_device(config.training.get('device', 'auto'))
    
    # Load models
    student_model, teacher_model = create_models(config)
    teacher_checkpoint = config.model.get('teacher', {}).get('checkpoint')
    load_teacher_checkpoint(teacher_model, teacher_checkpoint, device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(config.training.get('save_dir', 'checkpoints'), 'best_student_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student_model.load_state_dict(checkpoint['student_model_state_dict'])
    
    # Create data loaders
    data_loaders, datasets = create_data_loaders_from_config(config)
    
    # Create validator
    distillation_loss = create_distillation_loss(config)
    validator = Validator(
        student_model=student_model,
        teacher_model=teacher_model,
        distillation_loss=distillation_loss,
        device=device
    )
    
    # Run evaluation
    if 'test' in data_loaders:
        test_metrics = validator.evaluate_model(
            test_loader=data_loaders['test'],
            model_name="Loaded Student Model"
        )
    else:
        val_metrics = validator.evaluate_model(
            test_loader=data_loaders['val'],
            model_name="Loaded Student Model"
        )
    
    print("Model testing completed!")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Apply command line overrides
    if args.override:
        config = apply_config_overrides(config, args.override)
    
    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    print(f"Configuration loaded from {args.config}")
    
    # Run appropriate mode
    if args.data_only:
        test_data_loading(config)
    elif args.test_only:
        test_model_only(config)
    else:
        train_model(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
