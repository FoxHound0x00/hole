"""
Unified training script using Hydra configuration.

Usage:
    # Train with default config (ResNet50)
    python train.py
    
    # Train specific model
    python train.py model=convnext_base
    
    # Override parameters
    python train.py model=mixer_b16 training.epochs=20 training.batch_size=32
    
    # Train multiple models
    python train.py -m model=resnet18,resnet34,resnet50
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    resnet18, resnet34, resnet50,
    mobilenet_v3_small, mobilenet_v3_large,
    vit_b_32, vit_l_32,
    convnext_base, convnext_large,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights,
    ViT_B_32_Weights, ViT_L_32_Weights,
    ConvNeXt_Base_Weights, ConvNeXt_Large_Weights
)
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


def get_model(cfg: DictConfig, num_classes: int):
    """Load model based on configuration."""
    model_name = cfg.model.name
    architecture = cfg.model.architecture
    
    # Handle timm models
    if cfg.model.get('requires_timm', False):
        import timm
        model = timm.create_model(cfg.model.weights, pretrained=cfg.model.pretrained, num_classes=num_classes)
        return model
    
    # Map model names to constructors and weights
    model_map = {
        'resnet18': (resnet18, ResNet18_Weights.IMAGENET1K_V1),
        'resnet34': (resnet34, ResNet34_Weights.IMAGENET1K_V1),
        'resnet50': (resnet50, ResNet50_Weights.IMAGENET1K_V1),
        'mobilenet_v3_small': (mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1),
        'mobilenet_v3_large': (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V1),
        'vit_b_32': (vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1),
        'vit_l_32': (vit_l_32, ViT_L_32_Weights.IMAGENET1K_V1),
        'convnext_base': (convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1),
        'convnext_large': (convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1),
    }
    
    model_fn, weights = model_map[model_name]
    model = model_fn(weights=weights if cfg.model.pretrained else None)
    
    # Modify final layer for num_classes
    if architecture == 'resnet':
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'mobilenet_v3':
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif architecture == 'vit':
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif architecture == 'convnext':
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    
    return model


def get_transforms(cfg: DictConfig, train: bool = True):
    """Get data transforms based on configuration."""
    transform_cfg = cfg.dataset.train_transforms if train else cfg.dataset.test_transforms
    
    transform_list = []
    
    if 'resize' in transform_cfg:
        transform_list.append(transforms.Resize(transform_cfg.resize))
    
    if train and transform_cfg.get('random_horizontal_flip', False):
        transform_list.append(transforms.RandomHorizontalFlip())
    
    transform_list.append(transforms.ToTensor())
    
    if 'normalize' in transform_cfg:
        norm = transform_cfg.normalize
        transform_list.append(transforms.Normalize(norm.mean, norm.std))
    
    return transforms.Compose(transform_list)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    """Main training function."""
    print("=" * 80)
    print(f"Training {cfg.model.name} on {cfg.dataset.name}")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    
    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    train_transform = get_transforms(cfg, train=True)
    test_transform = get_transforms(cfg, train=False)
    
    train_dataset = datasets.CIFAR10(
        root=cfg.dataset.data_dir, 
        train=True, 
        download=True, 
        transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=cfg.dataset.data_dir, 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers
    )
    
    # Model
    model = get_model(cfg, cfg.dataset.num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    # Training loop
    accumulation_steps = cfg.training.gradient_accumulation_steps
    
    for epoch in range(cfg.training.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % cfg.training.log_interval == 0:
                print(f"Epoch [{epoch+1}/{cfg.training.epochs}], "
                      f"Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {running_loss/cfg.training.log_interval:.3f}, "
                      f"Acc: {100.*correct/total:.2f}%")
                running_loss = 0.0
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{cfg.training.epochs}] Test Accuracy: {test_acc:.2f}%\n")
    
    # Save model
    save_dir = Path(cfg.training.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{cfg.model.name}_cifar10.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()





