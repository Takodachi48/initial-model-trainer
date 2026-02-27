#!/usr/bin/env python3
"""
Two-stage training workflow script.

Stage 1: Train on your imbalanced data (3 classes)
Stage 2: Train on combined data (all classes)

Usage:
    python run_stages.py --stage 1
    python run_stages.py --stage 2
    python run_stages.py --all
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
    else:
        print("❌ Error!")
        if result.stderr:
            print(result.stderr)
        return False
    
    return True


def stage1_training():
    """Run Stage 1 training on your imbalanced data."""
    print("🚀 Starting Stage 1 Training")
    print("Training on your 3-class imbalanced data")
    
    # Check if data paths are configured
    config_path = "config/stage1.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    # Test data loading first
    print("\n📊 Testing data loading...")
    if not run_command(
        "python train.py --config config/stage1.yaml --data_only",
        "Data loading test for Stage 1"
    ):
        return False
    
    # Run training
    print("\n🏋️ Starting training...")
    if not run_command(
        "python train.py --config config/stage1.yaml",
        "Stage 1 training"
    ):
        return False
    
    # Test the trained model
    print("\n🧪 Testing trained model...")
    if not run_command(
        "python train.py --config config/stage1.yaml --test_only",
        "Stage 1 model testing"
    ):
        return False
    
    print("\n✅ Stage 1 completed successfully!")
    print("Model saved in: stage1_checkpoints/")
    print("Logs saved in: stage1_logs/")
    
    return True


def stage2_training():
    """Run Stage 2 training on combined data."""
    print("🚀 Starting Stage 2 Training")
    print("Training on combined data with all classes")
    
    # Check if stage 1 model exists
    stage1_model = "stage1_checkpoints/best_student_model.pth"
    if not os.path.exists(stage1_model):
        print(f"❌ Stage 1 model not found: {stage1_model}")
        print("Please run Stage 1 first or ensure the model exists.")
        return False
    
    # Check if data paths are configured
    config_path = "config/stage2.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    # Test data loading first
    print("\n📊 Testing data loading...")
    if not run_command(
        "python train.py --config config/stage2.yaml --data_only",
        "Data loading test for Stage 2"
    ):
        return False
    
    # Run training with resume from Stage 1
    print("\n🏋️ Starting training...")
    if not run_command(
        f"python train.py --config config/stage2.yaml --resume {stage1_model}",
        "Stage 2 training with transfer learning"
    ):
        return False
    
    # Test the trained model
    print("\n🧪 Testing trained model...")
    if not run_command(
        "python train.py --config config/stage2.yaml --test_only",
        "Stage 2 model testing"
    ):
        return False
    
    print("\n✅ Stage 2 completed successfully!")
    print("Model saved in: stage2_checkpoints/")
    print("Logs saved in: stage2_logs/")
    
    return True


def compare_results():
    """Compare results between Stage 1 and Stage 2."""
    print("\n📊 Comparing Results")
    print("="*60)
    
    # Load Stage 1 results
    stage1_path = "stage1_checkpoints/best_student_model.pth"
    stage2_path = "stage2_checkpoints/best_student_model.pth"
    
    if os.path.exists(stage1_path) and os.path.exists(stage2_path):
        import torch
        
        stage1_checkpoint = torch.load(stage1_path, map_location='cpu')
        stage2_checkpoint = torch.load(stage2_path, map_location='cpu')
        
        stage1_acc = stage1_checkpoint['metrics'].get('val_accuracy', 0)
        stage2_acc = stage2_checkpoint['metrics'].get('val_accuracy', 0)
        
        print(f"Stage 1 (3 classes): {stage1_acc:.4f}")
        print(f"Stage 2 (10 classes): {stage2_acc:.4f}")
        print(f"Difference: {stage2_acc - stage1_acc:+.4f}")
        
        print("\n📈 Model Information:")
        stage1_info = stage1_checkpoint['model_info']
        stage2_info = stage2_checkpoint['model_info']
        
        print(f"Stage 1 - Classes: {stage1_info['num_classes']}, "
              f"Parameters: {stage1_info['total_parameters']:,}")
        print(f"Stage 2 - Classes: {stage2_info['num_classes']}, "
              f"Parameters: {stage2_info['total_parameters']:,}")
    else:
        print("❌ Cannot compare - missing checkpoint files")


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check Python packages
    try:
        import torch
        import torchvision
        import timm
        print("✅ Required packages installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Check configuration files
    configs = ["config/stage1.yaml", "config/stage2.yaml"]
    for config in configs:
        if not os.path.exists(config):
            print(f"❌ Configuration file missing: {config}")
            return False
    
    print("✅ Configuration files found")
    
    # Check data directories (basic check)
    data_dirs = ["your_data", "combined_data", "existing_dataset"]
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"⚠️  Data directory not found: {data_dir}")
            print("    Make sure to update paths in config files")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Two-stage training workflow")
    parser.add_argument(
        "--stage", 
        type=int, 
        choices=[1, 2],
        help="Which stage to run (1 or 2)"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run both stages sequentially"
    )
    parser.add_argument(
        "--compare", 
        action="store_true",
        help="Compare results between stages"
    )
    parser.add_argument(
        "--check", 
        action="store_true",
        help="Check prerequisites only"
    )
    
    args = parser.parse_args()
    
    print("🤖 CNN Model Trainer - Two-Stage Workflow")
    print("="*60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix issues above.")
        return
    
    if args.check:
        print("\n✅ All prerequisites met!")
        return
    
    # Run requested stages
    success = True
    
    if args.stage == 1 or args.all:
        success &= stage1_training()
    
    if args.stage == 2 or args.all:
        success &= stage2_training()
    
    # Compare results if requested or if running all stages
    if (args.compare or args.all) and success:
        compare_results()
    
    if success:
        print("\n🎉 Workflow completed successfully!")
    else:
        print("\n❌ Workflow completed with errors. Please check messages above.")


if __name__ == "__main__":
    main()
