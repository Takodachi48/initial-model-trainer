# CNN Model Trainer - User Guide

## Herb Image Classification with Knowledge Distillation

A comprehensive guide for training and deploying herb classification models using MobileNetV3-Small (student) and EfficientNet-B0 (teacher) with knowledge distillation.

---

## TLDR: Quick Setup & Run Commands

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Prepare Dataset Structure

#### Option A: Use the Automated Split Script
```bash
# If you have all images in class folders:
# raw_data/
#   class1/
#   class2/
#   class3/

python split_dataset.py --input raw_data --output-root for_training --dataset-name default
```

#### Option B: Manual Dataset Structure
```
your_data/
├── train/
│   ├── class1/
│   ├── class2/
│   └── class3/
├── val/
│   ├── class1/
│   ├── class2/
│   └── class3/
└── test/ (optional)
    ├── class1/
    ├── class2/
    └── class3/
```

### 3. Update Configuration
Edit `config/default.yaml`:
```yaml
model:
  student:
    num_classes: 10  # Set to your number of classes
  teacher:
    num_classes: 10  # Set to your number of classes

data:
  train_dir: "for_training/default/train"  # If using split script
  val_dir: "for_training/default/val"      # If using split script
  test_dir: "for_training/default/test"    # If using split script
  # Or use "your_data/train", "your_data/val", "your_data/test" for manual setup
```

### 4. Run Training (in order)
```bash
# Step 1: Train teacher model
python train_teacher.py --config config/default.yaml

# Step 2: Train student model with knowledge distillation
python train.py --config config/default.yaml \
    --override model.teacher.checkpoint checkpoints_teacher/best_teacher_model.pth
```

### 5. Test Your Model
```bash
# Test teacher
python train_teacher.py --config config/default.yaml --test_only

# Test student
python train.py --config config/default.yaml --test_only
```

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Dataset Preparation](#dataset-preparation)
4. [Configuration](#configuration)
5. [Training](#training)
6. [Class Imbalance Handling](#class-imbalance-handling)
7. [Model Evaluation](#model-evaluation)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
```
your_data/
├── train/
│   ├── herb_class1/
│   ├── herb_class2/
│   └── herb_class3/
├── val/
│   ├── herb_class1/
│   ├── herb_class2/
│   └── herb_class3/
└── test/ (optional)
    ├── herb_class1/
    ├── herb_class2/
    └── herb_class3/
```

### 3. Update Configuration
Edit `config/default.yaml`:
```yaml
model:
  student:
    num_classes: 3
  teacher:
    num_classes: 3
    checkpoint: null  # Set this during student distillation

data:
  train_dir: "your_data/train"
  val_dir: "your_data/val"
  test_dir: "your_data/test"
```

### 4. Start Training
```bash
# Train teacher first
python train_teacher.py --config config/default.yaml

# Distill student using the trained teacher
python train.py --config config/default.yaml \
    --override model.teacher.checkpoint checkpoints_teacher/best_teacher_model.pth
```

---

## 🔧 Installation

### System Requirements
- Python 3.8+
- PyTorch 2.0+
- 4GB+ RAM
- 2GB+ disk space

### Install Dependencies
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

# Option 4: Install manually one by one (if above fail)
python -m pip install torch>=2.0.0
python -m pip install torchvision>=0.15.0
python -m pip install timm>=0.9.0
python -m pip install Pillow>=9.0.0
python -m pip install numpy>=1.21.0
python -m pip install tqdm>=4.64.0
python -m pip install matplotlib>=3.5.0
python -m pip install scikit-learn>=1.1.0
python -m pip install tensorboard>=2.10.0
```

### Verify Installation
```bash
python train.py --config config/default.yaml --data_only
```

---

## 📁 Dataset Preparation

### ⚠️ CRITICAL: Data Split Requirements

**IMPORTANT: Use DIFFERENT images for train, val, and test splits!**

- ❌ **NEVER** use the same images in multiple splits
- ❌ **NEVER** use test images during training
- ✅ **ALWAYS** ensure each image appears in ONLY ONE split
- ✅ **ALWAYS** maintain class distribution across splits

### Folder Structure
```
dataset_name/
├── train/          # 70-80% of data - UNIQUE IMAGES ONLY
│   ├── Ocimum_basilicum/        # Scientific name for basil
│   │   ├── basil_001.jpg  ← NOT in val or test
│   │   ├── herb_photo.jpg ← NOT in val or test
│   │   └── ...
│   ├── Mentha_spicata/          # Scientific name for mint
│   └── Rosmarinus_officinalis/  # Scientific name for rosemary
├── val/            # 10-15% of data - UNIQUE IMAGES ONLY
│   ├── Ocimum_basilicum/
│   │   ├── basil_101.jpg  ← NOT in train or test
│   │   ├── plant_202.jpg  ← NOT in train or test
│   │   └── ...
│   ├── Mentha_spicata/
│   └── Rosmarinus_officinalis/
└── test/           # 10-15% of data - UNIQUE IMAGES ONLY
    ├── Ocimum_basilicum/
    │   ├── basil_201.jpg  ← NOT in train or val
    │   ├── sample_img.jpg ← NOT in train or val
    │   └── ...
    ├── Mentha_spicata/
    └── Rosmarinus_officinalis/
```

### Data Split Ratios
| Dataset Size | Train | Val | Test | Total Images |
|-------------|-------|-----|------|--------------|
| < 1000      | 70%   | 15% | 15%  | 100%         |
| 1000-10000  | 80%   | 10% | 10%  | 100%         |
| > 10000     | 85%   | 10% | 5%   | 100%         |

### Example: 267 Total Images
```
Total: 267 images
├── Train: 187 images (70%) - UNIQUE IMAGES
├── Val:   40 images (15%)  - UNIQUE IMAGES  
└── Test:  40 images (15%)  - UNIQUE IMAGES

Per class (34, 110, 123 distribution):
├── Capsicum_frutescens (34): Train 24, Val 5, Test 5
├── Citrus_microcarpa (110): Train 77, Val 17, Test 16
└── Manihot_esculenta (123): Train 86, Val 18, Test 19
```

### How to Split Correctly

#### Option 1: Manual Split
1. **List all images** for each class
2. **Randomly shuffle** each class list
3. **Split by ratio** (70/15/15)
4. **Move images** to appropriate folders
5. **Verify no duplicates** across splits

#### Option 2: Script Split
```python
import os
import shutil
import random
from sklearn.model_selection import train_test_split

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split dataset ensuring no image appears in multiple splits"""
    
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # Get all images for this class
        images = [f for f in os.listdir(class_path) 
                 if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Shuffle randomly
        random.shuffle(images)
        
        # Calculate split sizes
        total = len(images)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        test_size = total - train_size - val_size
        
        # Split the list
        train_imgs = images[:train_size]
        val_imgs = images[train_size:train_size + val_size]
        test_imgs = images[train_size + val_size:]
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)
        
        # Copy images to respective splits
        for img in train_imgs:
            shutil.copy2(os.path.join(class_path, img), 
                        os.path.join(output_dir, 'train', class_name, img))
        
        for img in val_imgs:
            shutil.copy2(os.path.join(class_path, img), 
                        os.path.join(output_dir, 'val', class_name, img))
        
        for img in test_imgs:
            shutil.copy2(os.path.join(class_path, img), 
                        os.path.join(output_dir, 'test', class_name, img))
        
        print(f"{class_name}: {total} total → Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

# Usage
split_dataset("raw_herb_images", "for_training/stage_1")
```

### Verification Checklist
- [ ] **No duplicate images** across train/val/test
- [ ] **Class distribution maintained** in each split
- [ ] **Correct ratios** (70/15/15 or 80/10/10)
- [ ] **All images accounted for** (train + val + test = total)
- [ ] **Proper folder structure** created

### Why This Matters

#### Using Same Images in Multiple Splits Causes:
- **Inflated accuracy** - Model has already seen test images
- **Overfitting detection failure** - Can't identify real overfitting
- **Unreliable evaluation** - Test results don't reflect real performance
- **Poor generalization** - Model fails on new, unseen data

#### Proper Splitting Ensures:
- **Realistic performance** - Test on truly unseen data
- **Overfitting detection** - Validation catches training issues
- **Reliable metrics** - Accuracy reflects real-world performance
- **Better generalization** - Model learns to handle new data

### Common Mistakes to Avoid

❌ **Wrong**: Same image in train and test
```
train/basil/img001.jpg  ← Same image
test/basil/img001.jpg   ← Same image (WRONG!)
```

✅ **Correct**: Different images in each split
```
train/basil/img001.jpg  ← Unique to train
test/basil/img101.jpg   ← Unique to test
```

❌ **Wrong**: Random copying without tracking
```
# Copying randomly without ensuring uniqueness
cp *.jpg ../train/
cp *.jpg ../val/  # Same files! (WRONG!)
cp *.jpg ../test/ # Same files! (WRONG!)
```

✅ **Correct**: Systematic split with tracking
```
# Split systematically ensuring no overlap
python split_dataset.py --input raw_data --output-root for_training --dataset-name default
```

### Naming Conventions
- **Folder names (Classes)**: Use **SCIENTIFIC NAMES** (e.g., `Ocimum_basilicum`, `Mentha_spicata`, `Rosmarinus_officinalis`)
- **Image names**: Any valid name (e.g., `basil_001.jpg`, `IMG_2024_001.jpg`, `herb_photo.jpg`)
- **Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

#### Scientific Name Examples
```
✅ RECOMMENDED: Scientific names for folders
├── Ocimum_basilicum/        # Sweet basil
├── Mentha_spicata/          # Spearmint  
├── Rosmarinus_officinalis/  # Rosemary
├── Salvia_officinalis/      # Common sage
├── Thymus_vulgaris/         # Common thyme
├── Origanum_vulgare/        # Oregano
└── Capsicum_frutescens/     # Chili pepper

❌ AVOID: Common names or inconsistent formatting
├── basil/                   # Too generic
├── Sweet Basil/             # Contains space
├── mint/                    # Too generic
├── rosemary/                # Common name OK but scientific preferred
└── herb_1/                  # Not descriptive
```

#### Scientific Name Guidelines
- **Use Latin scientific names** when possible
- **Genus_species format** (e.g., `Ocimum_basilicum`)
- **Use underscores** instead of spaces
- **Capitalize Genus**, lowercase species
- **No special characters** or accents
- **Be consistent** across all classes

#### Benefits of Scientific Names
- **Universally recognized** across languages
- **Precise identification** of plant species
- **Professional standard** for botanical work
- **Avoids confusion** between common names
- **Better for research** and documentation

### CSV Alternative
Use CSV files instead of folder structure:
```csv
image_path,label,split
class1/img001.jpg,class1,train
class1/img002.jpg,class1,train
class1/img101.jpg,class1,val
class1/img201.jpg,class1,test
```

**Remember: Each image must appear in EXACTLY ONE split!**

---

## ⚙️ Configuration

### Configuration Files
- `config/default.yaml` - Default configuration

### Key Configuration Options

#### Model Settings
```yaml
model:
  student:
    name: "mobilenetv3_small_100"  # CPU-friendly
    num_classes: 3
    freeze_backbone: true
  teacher:
    name: "efficientnet_b0"        # Strong teacher
    num_classes: 3
    checkpoint: null  # Set to checkpoints_teacher/best_teacher_model.pth
```

#### Training Settings
```yaml
training:
  phases:
    phase1:
      epochs: 10
      learning_rate: 0.001
      freeze_backbone: true
      weight_decay: 0.0001
    phase2:
      epochs: 5
      learning_rate: 0.0001
      freeze_backbone: false
      weight_decay: 0.00001
  batch_size: 32
  device: "auto"  # auto, cpu, cuda
```

#### Class Imbalance Handling
```yaml
training:
  class_imbalance:
    use_class_weights: true      # Weighted loss
    use_balanced_sampler: true   # Balanced batches
```

#### Data Augmentation
```yaml
data:
  augmentations:
    horizontal_flip: 0.7
    rotation_degrees: 20
    color_jitter:
      brightness: 0.3
      contrast: 0.3
```

---

## 🏋️ Training

### Basic Training Commands

#### Test Data Loading
```bash
python train.py --config config/default.yaml --data_only
```

#### Train Teacher
```bash
python train_teacher.py --config config/default.yaml
```

#### Distill Student
```bash
python train.py --config config/default.yaml \
    --override model.teacher.checkpoint checkpoints_teacher/best_teacher_model.pth
```

#### Resume Teacher Training
```bash
python train_teacher.py --config config/default.yaml \
    --resume checkpoints_teacher/latest_teacher_model.pth
```

#### Resume Student Distillation
```bash
python train.py --config config/default.yaml \
    --override model.teacher.checkpoint checkpoints_teacher/best_teacher_model.pth \
    --resume checkpoints/latest_student_model.pth
```

#### Test Trained Teacher
```bash
python train_teacher.py --config config/default.yaml --test_only
```

#### Test Trained Model
```bash
python train.py --config config/default.yaml --test_only
```

#### Override Configuration
```bash
python train.py --config config/default.yaml \
    --override training.batch_size 32 \
    --override model.student.num_classes 5
```

### Monitoring Training

#### TensorBoard
```bash
tensorboard --logdir logs
```

#### Progress Indicators
- **Training loss**: Should decrease steadily
- **Validation accuracy**: Main performance metric
- **Class accuracy**: Per-class performance
- **Inference speed**: Samples per second

---

## ⚖️ Class Imbalance Handling

### When to Use
- **Imbalanced datasets** (e.g., 34 vs 110 samples)
- **Minority classes** need more attention
- **Real-world data** is rarely balanced

### Available Methods

#### 1. Weighted Loss
```yaml
training:
  class_imbalance:
    use_class_weights: true
```
- Higher weight for minority classes
- Lower weight for majority classes
- Automatic calculation based on class distribution

#### 2. Balanced Sampling
```yaml
training:
  class_imbalance:
    use_balanced_sampler: true
```
- Equal representation in each batch
- Prevents model bias toward majority classes
- More stable training

### Example Results
```
Class distribution: [24, 86, 77]
Class weights: [2.60, 0.72, 0.81]
```
- Minority class (24 samples): Weight 2.60
- Majority classes (86-77 samples): Weight ~0.75

---

## 📊 Model Evaluation

### Performance Metrics

#### Accuracy Metrics
- **Overall Accuracy**: Total correct predictions
- **Class Accuracy**: Per-class performance
- **Average Class Accuracy**: Mean of all class accuracies

#### Loss Metrics
- **Training Loss**: Model learning progress
- **Validation Loss**: Generalization performance
- **Distillation Loss**: Teacher-student alignment

#### Inference Metrics
- **Samples per second**: Processing speed
- **Time per sample**: Latency
- **Model size**: Memory footprint

### Evaluation Commands

#### Test on Validation Set
```bash
python train.py --config config/default.yaml --test_only
```

#### Test Teacher Model
```bash
python train_teacher.py --config config/default.yaml --test_only
```

#### View Detailed Results
```python
# Load and analyze results
import torch
checkpoint = torch.load('checkpoints/best_student_model.pth')
print(checkpoint['metrics'])
```

#### Per-Class Analysis
The evaluation shows per-class accuracy to identify:
- Well-performing classes
- Classes needing more data
- Confusion between similar herbs

---

## 🚀 Deployment

### Model Files
```
checkpoints/
├── best_student_model.pth    # Best performing model
├── latest_student_model.pth  # Most recent model
└── checkpoint_epoch_X.pth     # Periodic saves
```

### Model Information
```python
import torch
checkpoint = torch.load('checkpoints/best_student_model.pth')
model_info = checkpoint['model_info']

print(f"Model: {model_info['model_name']}")
print(f"Parameters: {model_info['total_parameters']:,}")
print(f"Classes: {model_info['num_classes']}")
```

### Inference Example
```python
import torch
from torchvision import transforms
from PIL import Image

# Load model
device = torch.device('cpu')
model = StudentModel(num_classes=3, pretrained=False)
checkpoint = torch.load('checkpoints/best_student_model.pth', 
                       map_location=device)
model.load_state_dict(checkpoint['student_model_state_dict'])
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Predict
image = Image.open('herb_image.jpg')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
```

### Performance Optimization
- **CPU inference**: ~3-5 samples/second
- **Batch processing**: Process multiple images together
- **Model quantization**: Further reduce size and improve speed

---

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
```bash
# ImportError: cannot import name 'X'
# Solution: Check __init__.py files
python -c "from models import StudentModel; print('OK')"
```

#### Configuration Errors
```bash
# TypeError: '<=' not supported between instances of 'float' and 'str'
# Solution: Use decimal notation instead of scientific
weight_decay: 0.0001  # Not 1e-4
```

#### Data Loading Issues
```bash
# FileNotFoundError or empty dataset
# Solution: Check data paths and structure
python train.py --config config/default.yaml --data_only
```

#### Memory Issues
```bash
# CUDA out of memory
# Solution: Reduce batch size
python train.py --config config/default.yaml \
    --override training.batch_size 8
```

#### TensorFlow Warnings
```bash
# TensorFlow oneDNN warnings
# Solution: Set environment variable
set TF_ENABLE_ONEDNN_OPTS=0
# Or add to script (already done)
```

### Performance Issues

#### Low Accuracy
- **Check data quality**: Ensure correct labels
- **Increase epochs**: More training time
- **Adjust learning rate**: Try different values
- **Add more data**: Especially for poor-performing classes

#### Slow Training
- **Reduce batch size**: Less memory usage
- **Use fewer workers**: Reduce CPU overhead
- **Enable GPU**: If available
- **Reduce image size**: Smaller inputs process faster

#### Overfitting
- **Add augmentation**: More data variety
- **Reduce epochs**: Stop earlier
- **Increase regularization**: Higher weight decay
- **Use dropout**: Already included in model

### Getting Help

#### Debug Mode
```bash
# Enable detailed logging
python train.py --config config/default.yaml \
    --override logging.print_every 1
```

#### Check Results
```bash
# View training logs
tensorboard --logdir logs
tensorboard --logdir logs_teacher

# Check model files
ls -la checkpoints/
ls -la checkpoints_teacher/
```

#### Reset Training
```bash
# Remove all results and start fresh
rm -rf checkpoints/ logs/ checkpoints_teacher/ logs_teacher/
python train_teacher.py --config config/default.yaml
python train.py --config config/default.yaml \
    --override model.teacher.checkpoint checkpoints_teacher/best_teacher_model.pth
```

---

## 📚 Advanced Usage

### Custom Models
```python
# Create custom student model
student_model = StudentModel(
    num_classes=10,
    model_name="efficientnet_b0",  # Different architecture
    freeze_backbone=False
)
```

### Custom Loss Functions
```python
# Modify distillation loss
class CustomDistillationLoss(DistillationLoss):
    def forward(self, student_logits, teacher_logits, labels):
        # Custom loss calculation
        return custom_loss, loss_dict
```

### Custom Data Augmentation
```python
# Add custom transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=30),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
```

---

## 🎯 Best Practices

### Data Preparation
1. **Consistent naming**: Use clear class names
2. **Quality over quantity**: Better images than more images
3. **Balanced splits**: Maintain class distribution
4. **Proper validation**: Never use test data for training

### Training Strategy
1. **Start simple**: Use default configurations
2. **Monitor progress**: Use TensorBoard
3. **Save frequently**: Don't lose progress
4. **Test thoroughly**: Evaluate on unseen data

### Model Deployment
1. **Validate performance**: Test on real data
2. **Optimize inference**: Measure speed requirements
3. **Monitor drift**: Track performance over time
4. **Update regularly**: Retrain with new data

---

## 📞 Support

### Resources
- **GitHub Issues**: Report bugs and request features
- **Documentation**: This guide and code comments
- **Examples**: `example_usage.py` for reference

### Community
- **Discussions**: Share experiences and tips
- **Contributions**: Welcome improvements and fixes

---

## 📄 License

This project is open source. See LICENSE file for details.

---

**Happy Training! 🌿**

Remember: Good models start with good data. Focus on data quality first, then optimize training parameters.
