'''
Created on Dec. 7, 2025

@author: Loosoon
'''
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets, transforms, models
from datetime import datetime
import time

current_time = datetime.now()
print(f"--------- Start runing model, ResNet-152, at the time of: {current_time} ---------")

# Set seed for random generators
torch.manual_seed(42)
generator = torch.Generator().manual_seed(42)

data_dir = "dataset/PlantVillage_20"

image_size = 224                      # Required for ResNet
batch_size = 32
num_workers = 0
epochs = 50
learning_rate = 1e-4

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std), # ImageNet normalization
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

base_dataset = datasets.ImageFolder(root=data_dir, transform=None)
full_train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
full_val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

num_classes = len(base_dataset.classes)
num_samples = len(base_dataset.targets)

total_len = len(base_dataset)
if total_len == 0:
    raise RuntimeError(f"No images found in {data_dir}.")

# Split (80% for training, 20% for validation)
train_len = int(0.8 * total_len)
val_len = total_len - train_len
train_subset_indices, val_subset_indices = random_split(range(total_len), [train_len, val_len], generator=generator)

# random_split on a range returns a Subset-like object; extract indices
train_indices = list(train_subset_indices)
val_indices = list(val_subset_indices)

# Create Subset objects on the datasets that have transforms applied:
train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_val_dataset, val_indices)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


class ResNet152Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet152(weights="IMAGENET1K_V1")
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

model = ResNet152Classifier(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

current_time = datetime.now()
print(f"Time of starting training: {current_time}")

for epoch in range(1, epochs + 1):

    t0 = time.time()

    # ----- Training -----
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for imgs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)
        train_total += imgs.size(0)
        train_correct += (outputs.argmax(dim=1) == labels).sum().item()

    train_loss /= train_total
    train_acc = 100 * train_correct / train_total

    t1 = time.time()
    print(f"Time cost for training epoch {epoch} is {t1-t0:.1f}s")

    # ----- Validation -----
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            val_total += imgs.size(0)
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()

    val_loss /= val_total
    val_acc = 100 * val_correct / val_total

    t2 = time.time()
    print(f"Time cost for validation epoch {epoch} is {t2-t1:.1f}s")

    print(f"Epoch {epoch}/{epochs} "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

current_time = datetime.now()
print(f"\nTraining complete at {current_time}")




