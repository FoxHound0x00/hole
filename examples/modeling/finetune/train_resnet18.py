"""
ResNet fine-tuning on CIFAR-10 using HuggingFace Transformers
"""

from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers import TrainingArguments, Trainer
from torchvision.transforms import (CenterCrop, Compose, Normalize, RandomHorizontalFlip,
                                    RandomResizedCrop, Resize, ToTensor)
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import accuracy_score


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
    # load cifar10 (only small portion for demonstration purposes) 
    train_ds, test_ds = load_dataset('cifar10', split=['train[:5000]', 'test[:2000]'])
    # split up training into training + validation
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']

    print(f"train dataset : {train_ds}")
    print(f"train dataset features : {train_ds.features}")

    id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
    label2id = {label:id for id,label in id2label.items()}
    print(id2label)
    
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["shortest_edge"]

    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    _val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )
    
    # Set the transforms
    train_ds.set_transform(train_transforms)
    val_ds.set_transform(val_transforms)
    test_ds.set_transform(val_transforms)
    
    train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)
        

    batch = next(iter(train_dataloader))
    for k,v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
    
    model = ResNetForImageClassification.from_pretrained('microsoft/resnet-18',
                                                  id2label=id2label,
                                                  label2id=label2id,
                                                  ignore_mismatched_sizes=True)
    
    metric_name = "accuracy"
    args = TrainingArguments(
        f"resnet18_cifar10",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_dir='logs',
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    trainer.train()
    outputs = trainer.predict(test_ds)
    print(outputs.metrics)

    trainer.save_model("./resnet18_cifar10_finetuned")
