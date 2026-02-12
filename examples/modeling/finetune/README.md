# Fine-tuning Scripts

Clean, production-ready training scripts using HuggingFace Transformers Trainer API.

## Available Models

### Vision Transformers (ViT)
- **ViT-Base**: `train_vit_base.py` - google/vit-base-patch16-224-in21k
- **ViT-Large**: `train_vit_large.py` - google/vit-large-patch16-224-in21k

### ResNet
- **ResNet-18**: `train_resnet18.py` - microsoft/resnet-18
- **ResNet-34**: `train_resnet34.py` - microsoft/resnet-34
- **ResNet-50**: `train_resnet50.py` - microsoft/resnet-50

### ConvNeXT
- **ConvNeXT-Base**: `train_convnext_base.py` - facebook/convnext-base-224
- **ConvNeXT-Large**: `train_convnext_large.py` - facebook/convnext-large-224

### MobileNet
- **MobileNetV2-Large**: `train_mobilenet_large.py` - google/mobilenet_v2_1.0_224
- **MobileNetV2-Small**: `train_mobilenet_small.py` - google/mobilenet_v2_0.75_160

### EfficientNet
- **EfficientNet-B0**: `train_efficientnet.py` - google/efficientnet-b0

## Features

✅ Clean, modular code following best practices  
✅ HuggingFace Trainer API for robust training  
✅ Automatic checkpointing and best model selection  
✅ Validation split from training data  
✅ Test set evaluation with confusion matrix  
✅ Proper data augmentation (train vs val/test)  
✅ Mixed precision training (FP16) for large models  
✅ TensorBoard logging  

## Usage

```bash
# Install dependencies
pip install transformers datasets torch torchvision scikit-learn matplotlib

# Run training - pick any model variant
cd examples/modeling/finetune
python train_vit_base.py
python train_resnet18.py
python train_convnext_base.py
# etc...
```

## Training Output

Each script will:
1. Download and prepare CIFAR-10 dataset
2. Train the model with automatic checkpointing
3. Evaluate on test set
4. Generate confusion matrix visualization
5. Save the fine-tuned model

## Customization

Edit these key parameters in each script:

```python
TrainingArguments(
    output_dir="model_name_cifar10",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    num_train_epochs=10,
    ...
)
```

## Hardware Requirements

### Small Models (4-8GB VRAM)
- ResNet-18
- MobileNetV2 (both variants)
- EfficientNet-B0

### Medium Models (8-12GB VRAM)
- ResNet-34/50
- ViT-Base
- ConvNeXT-Base (with FP16)

### Large Models (16GB+ VRAM)
- ViT-Large (FP16 + gradient accumulation)
- ConvNeXT-Large (FP16 + gradient accumulation)

Adjust `per_device_train_batch_size` and `gradient_accumulation_steps` based on your GPU memory.

## Model Comparison

All models trained on CIFAR-10 can be compared for:
- **Accuracy**: Test set performance
- **Speed**: Training time per epoch
- **Efficiency**: Parameters vs performance
- **Memory**: VRAM requirements
