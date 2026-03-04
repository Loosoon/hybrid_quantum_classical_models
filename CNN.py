'''
Created on Dec. 14, 2025

@author: Loosoon
'''
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets, transforms
import math
from datetime import datetime
import time


current_time = datetime.now()
print(f"--------- Start runing model, CNN, at the time of: {current_time} ---------")

# Set seed for random generators
torch.manual_seed(42)
generator = torch.Generator().manual_seed(42)

data_dir = "dataset/PlantVillage_20"
image_size = 128
batch_size = 32
epochs = 100
num_workers = 0

# Training hyperparams
initial_lr = 1e-4    # reduced LR for hybrid nets is usually safer
weight_decay = 1e-5
grad_clip_val = 1.0  # prevents exploding gradients

# load dataset
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

base_dataset = datasets.ImageFolder(root=data_dir, transform=None)
full_train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
full_val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

num_classes = len(base_dataset.classes)   # dataset's number of classes
num_samples = len(base_dataset.targets)

print(f"Total number of classes: {num_classes}, Total number of samples: {num_samples}")

# Split (80/20)
total_len = len(base_dataset)
if total_len == 0:
    raise RuntimeError(f"No images found in {data_dir}. Check path and folder structure (ImageFolder expects class subfolders).")

train_len = int(0.8 * total_len)
val_len = total_len - train_len
train_subset_indices, val_subset_indices = random_split(range(total_len), [train_len, val_len], generator=generator)

# random_split on a range returns a Subset-like object; extract indices
train_indices = list(train_subset_indices)
val_indices = list(val_subset_indices)

# create Subset objects on the datasets that have transforms applied:
train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_val_dataset, val_indices)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


class ClassicalCNN(nn.Module):
    def __init__(self, num_classes: int, image_size: int = 128):
        super().__init__()

        # convolutional backbone
        self.conv_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 128 -> 64

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Compute flattened dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feats = self.conv_backbone(dummy)
            flat_dim = feats.view(1, -1).shape[1]

        self.fc_features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.ReLU(),
        )

        # Classical head
        self.classifier = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.conv_backbone(x)
        x = self.fc_features(x)
        logits = self.classifier(x)
        return logits


model = ClassicalCNN(
    num_classes=num_classes,
    image_size=image_size
)


# Training utils
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

best_val_acc = -math.inf

current_time = datetime.now()
print(f"Time of starting training: {current_time}")

train_losses, val_losses = [], []
train_accs, val_accs = [], []

# Training loop
for epoch in range(1, epochs + 1):
    t0 = time.time()
    
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total * 100
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    t1 = time.time()
    print(f"Time cost for training epoch {epoch} is {t1-t0:.1f}s")

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * imgs.size(0)
            val_total += imgs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss /= val_total
    val_acc = val_correct / val_total * 100
    
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    t2 = time.time()
    print(f"Time cost for validation epoch {epoch} is {t2-t1:.1f}s")

    scheduler.step(val_loss)

    print(f"Epoch {epoch}/{epochs}  Train Loss {train_loss:.4f}  Train Acc {train_acc:.2f}%  "
          f"Val Loss {val_loss:.4f}  Val Acc {val_acc:.2f}%")

    # save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc

current_time = datetime.now()
print(f"\nTraining complete at {current_time}. Best Val Acc: {best_val_acc:.2f}%")




