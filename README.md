# CNN Model Trainer - Herb Image Classification

A PyTorch-based training pipeline for herb image classification using knowledge distillation with MobileNetV3 (student) and EfficientNet (teacher).

## Architecture

- **Student Model**: MobileNetV3-Small (deployment-ready, CPU-friendly)
- **Teacher Model**: EfficientNet-B0 (training-only, provides soft targets)
- **Training Strategy**: Knowledge distillation with configurable training

## Features

- Knowledge distillation loss combining hard and soft targets
- Configurable training with flexible hyperparameters
- Configurable data augmentations
- Comprehensive metrics and logging
- Model checkpointing

## Quick Start

```bash
# Option 1: Install all dependencies
pip install -r requirements.txt

# Option 2: If pip install -r requirements.txt doesn't work
pip install torch>=2.0.0 torchvision>=0.15.0
pip install timm>=0.9.0 Pillow>=9.0.0
pip install numpy>=1.21.0 tqdm>=4.64.0
pip install matplotlib>=3.5.0 scikit-learn>=1.1.0
pip install tensorboard>=2.10.0

# Option 3: For Windows PowerShell (if pip not recognized)
python -m pip install -r requirements.txt

# 1) Train teacher first (creates checkpoints_teacher/best_teacher_model.pth)
python train_teacher.py --config config/default.yaml

# 2) Distill student using trained teacher
python train.py --config config/default.yaml \
  --override model.teacher.checkpoint checkpoints_teacher/best_teacher_model.pth
```

## Training Workflow

This project uses a two-step workflow:
1. Train a teacher model with `train_teacher.py`
2. Train the student model with `train.py` using `model.teacher.checkpoint`

The student distillation step does not automatically train the teacher.

## Project Structure

```
├── models/
│   ├── __init__.py
│   ├── student.py          # MobileNetV3 student model
│   ├── teacher.py          # EfficientNet teacher model
│   └── distillation.py     # Knowledge distillation loss
├── data/
│   ├── __init__.py
│   └── dataset.py          # Data pipeline and augmentations
├── training/
│   ├── __init__.py
│   ├── trainer.py          # Training loop
│   └── validator.py        # Validation loop
├── config/
│   └── default.yaml        # Default configuration
├── train_teacher.py        # Train teacher model (supervised)
├── train.py                # Distill and train student model
└── requirements.txt
```

## Configuration

Edit `config/default.yaml` to modify:
- Model parameters
- Training hyperparameters
- Data paths
- Distillation settings (alpha, temperature)
- Teacher checkpoint path (`model.teacher.checkpoint`) for student distillation
