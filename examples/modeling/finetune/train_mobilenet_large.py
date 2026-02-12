"""
MobileNetV2 fine-tuning on CIFAR-10 using HuggingFace Transformers
Model: google/mobilenet_v2_1.0_224
"""

from datasets import load_dataset
from transformers import AutoImageProcessor, MobileNetV2ForImageClassification
from transformers import TrainingArguments, Trainer
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomHorizontalFlip,
    RandomResizedCrop, Resize, ToTensor
)
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))


def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples


def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


if __name__ == "__main__":
    # Load CIFAR-10 dataset
    train_ds, test_ds = load_dataset('cifar10', split=['train', 'test'])
    
    # Split training into train + validation
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']

    print(f"Train dataset: {train_ds}")
    print(f"Validation dataset: {val_ds}")
    print(f"Test dataset: {test_ds}")

    # Create label mappings
    id2label = {id: label for id, label in enumerate(train_ds.features['label'].names)}
    label2id = {label: id for id, label in id2label.items()}
    print(f"Labels: {list(id2label.values())}")
    
    # Initialize processor
    processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["shortest_edge"]

    # Define transforms
    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])

    _val_transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])
    
    # Set transforms
    train_ds.set_transform(train_transforms)
    val_ds.set_transform(val_transforms)
    test_ds.set_transform(val_transforms)
    
    # Load model
    model = MobileNetV2ForImageClassification.from_pretrained(
        'google/mobilenet_v2_1.0_224',
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    # Training arguments
    metric_name = "accuracy"
    args = TrainingArguments(
        output_dir="mobilenet_v2_cifar10",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=15,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_dir='logs',
        logging_steps=100,
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    # Train
    print("\n" + "="*50)
    print("Starting MobileNetV2 training...")
    print("="*50 + "\n")
    trainer.train()
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50 + "\n")
    outputs = trainer.predict(test_ds)
    print(f"Test metrics: {outputs.metrics}")

    # Confusion matrix
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)
    labels = train_ds.features['label'].names
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig("mobilenet_v2_confusion_matrix.png", dpi=150)
    print("Confusion matrix saved to mobilenet_v2_confusion_matrix.png")
    
    # Save model
    trainer.save_model("./mobilenet_v2_cifar10_finetuned")
    print("\nModel saved to ./mobilenet_v2_cifar10_finetuned")
