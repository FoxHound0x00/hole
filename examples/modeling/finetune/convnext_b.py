# Model: ConvNeXt_Base_Weights.IMAGENET1K_V1
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

# Config
EPOCHS = 10
BATCH_SIZE = 16  # Reduced for memory constraints
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACCUMULATION_STEPS = 4  # Effective batch size = 16 * 4 = 64

# Data transforms
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Model
model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 10)  # CIFAR-10 has 10 classes
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training
print(f"Training ConvNeXt-Base on {DEVICE}")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss / ACCUMULATION_STEPS
        loss.backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * ACCUMULATION_STEPS
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], "
                  f"Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.2f}%")
            running_loss = 0.0
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] Test Accuracy: {100.*correct/total:.2f}%\n")

# Save model
torch.save(model.state_dict(), 'convnext_b_cifar10.pth')
print("Model saved to convnext_b_cifar10.pth")

