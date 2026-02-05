# Model: MLP-Mixer-L/16 Fine-tuning with Transformers Trainer
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import timm
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
from sklearn.metrics import accuracy_score

EPOCHS = 30
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 5

class CIFARDataset(Dataset):
    def __init__(self, cifar_dataset):
        self.dataset = cifar_dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {"pixel_values": image, "labels": label}

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

print("Loading CIFAR-10...")
train_dataset_raw = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset_raw = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_size = int(0.9 * len(train_dataset_raw))
val_size = len(train_dataset_raw) - train_size
train_dataset_raw, val_dataset_raw = torch.utils.data.random_split(
    train_dataset_raw, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

train_dataset = CIFARDataset(train_dataset_raw)
val_dataset = CIFARDataset(val_dataset_raw)
test_dataset = CIFARDataset(test_dataset_raw)

class MixerForCIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('mixer_l16_224.goog_in21k_ft_in1k', pretrained=True, num_classes=10)
    def forward(self, pixel_values, labels=None):
        logits = self.model(pixel_values)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

model = MixerForCIFAR()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

training_args = TrainingArguments(
    output_dir="./mixer_l16_checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
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

print(f"\nTraining MLP-Mixer-L/16...")
trainer.train()

test_results = trainer.evaluate(test_dataset)
print(f"Test Results: {test_results}")

torch.save(model.model.state_dict(), 'mixer_l16_cifar10.pth')
print("Model saved!")
