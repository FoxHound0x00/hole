# Model: ViT-L/32 Fine-tuning with Transformers Trainer
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.models import vit_l_32, ViT_L_32_Weights
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
from sklearn.metrics import accuracy_score

# Config
EPOCHS = 30
BATCH_SIZE = 64  # Smaller for larger model
EVAL_BATCH_SIZE = 64
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 5

# Custom Dataset wrapper
class CIFARDataset(Dataset):
    def __init__(self, cifar_dataset):
        self.dataset = cifar_dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {"pixel_values": image, "labels": label}

# Data transforms with proper augmentation
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10
print("Loading CIFAR-10 dataset...")
train_dataset_raw = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset_raw = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Split train into train + validation
train_size = int(0.9 * len(train_dataset_raw))
val_size = len(train_dataset_raw) - train_size
train_dataset_raw, val_dataset_raw = torch.utils.data.random_split(
    train_dataset_raw, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Wrap datasets
train_dataset = CIFARDataset(train_dataset_raw)
val_dataset = CIFARDataset(val_dataset_raw)
test_dataset = CIFARDataset(test_dataset_raw)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# Model wrapper
class ViTForCIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vit_l_32(weights=ViT_L_32_Weights.IMAGENET1K_V1)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, 10)
        
    def forward(self, pixel_values, labels=None):
        logits = self.model(pixel_values)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

print("Initializing ViT-L/32 model...")
model = ViTForCIFAR()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

training_args = TrainingArguments(
    output_dir="./vit_l_32_checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    remove_unused_columns=False,
    report_to="none",
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
)

print(f"\nTraining ViT-L/32 on CIFAR-10 with Transformers Trainer...")
print(f"LR: {LR}, Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}")
print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE} epochs\n")

trainer.train()

print("\nEvaluating on test set...")
test_results = trainer.evaluate(test_dataset)
print(f"Test Results: {test_results}")

torch.save(model.model.state_dict(), 'vit_l_32_cifar10.pth')
print("Model saved to vit_l_32_cifar10.pth")
