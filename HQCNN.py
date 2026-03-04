'''
Created on Feb. 26, 2026

@author: Loosoon
'''
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets, transforms
from collections import Counter
import math
from datetime import datetime
import time

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.utils import algorithm_globals


current_time = datetime.now()
print(f"--------- Start runing model, QCNN, at the time of: {current_time} ---------")

# Set seed for random generators
torch.manual_seed(42)
generator = torch.Generator().manual_seed(42)
algorithm_globals.random_seed = 42

data_dir = "dataset/PlantVillage_20"
image_size = 128
batch_size = 8
n_qubits = 6         # number of qubits in QNN input
n_reps = 1           # repetitions for ansatz
epochs = 260
num_workers = 0


# Training hyperparams
initial_lr = 1e-4
weight_decay = 1e-5
grad_clip_val = 1.0

# Transforms and dataset loading, 
# And add data augmentation, including RandomHorizontalFlip, RandomRotation and ColorJitter
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

# Class counts / names
base_targets = base_dataset.targets
class_names = base_dataset.classes
full_counts = Counter(base_targets)

def print_counts(counts, names):
    for cls_idx in sorted(counts.keys()):
        print(f"{names[cls_idx]}: {counts[cls_idx]}")

print("Full dataset counts:")
print(f"Total number of classes: {num_classes}, total number of samples: {num_samples}")
print_counts(full_counts, class_names)

# train/val counts
train_targets = [base_targets[i] for i in train_indices]
val_targets = [base_targets[i] for i in val_indices]
print("\nTrain counts:")
print_counts(Counter(train_targets), class_names)
print("\nVal counts:")
print_counts(Counter(val_targets), class_names)

# QNN creation
def create_estimator_qnn(n_qubits: int):
    """
    Creates an EstimatorQNN with n_qubits inputs.
    """
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=n_reps)
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=n_reps)

    qc = QuantumCircuit(n_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    estimator = Estimator()  # CPU primitive (statevector)
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,   # parameter objects for input features
        weight_params=ansatz.parameters,       # parameter objects for trainable weights
        input_gradients=True,
        estimator=estimator,
    )
    return qnn

qnn = create_estimator_qnn(n_qubits=n_qubits)

# Hybrid PyTorch Module
class HybridQCNN(nn.Module):
    def __init__(self, qnn, n_qubits: int, num_classes: int, image_size: int = 128):
        super().__init__()
        self.n_qubits = n_qubits
        self.num_classes = num_classes

        # simple but slightly deeper CNN backbone
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
            nn.AdaptiveAvgPool2d((4,4)),  # reduce spatial dims deterministically
        )

        # compute flattened dim dynamically
        with torch.no_grad():
            example = torch.zeros(1, 3, image_size, image_size)
            feats = self.conv_backbone(example)
            flat_dim = feats.view(1, -1).shape[1]

        # classical head -> reduce dimension to n_qubits
        self.fc_classical = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_qubits),
        )

        # QNN wrapped as torch module (TorchConnector)
        self.qnn_torch = TorchConnector(qnn)  # maps (batch, n_qubits) -> (batch, qnn_output_dim)

        # post-QNN head (map QNN outputs to class logits)
        self.post_qnn = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x -> (batch, 3, H, W)
        # classical feature extraction
        x = self.conv_backbone(x)
        x = self.fc_classical(x)  # -> (batch, n_qubits)
        
        # Scale inputs for quantum encoding
        x = torch.tanh(x) * math.pi

        # TorchConnector runs QNN using CPU-based Qiskit primitives.
        # If the rest of the model is on GPU, move the small qnn input to CPU,
        # run qnn, then move outputs back to original device.
        model_device = next(self.parameters()).device
        if model_device.type == "cuda":
            # move the classical features to cpu (float32)
            q_in = x.detach().cpu().to(torch.float32)
        else:
            q_in = x.to(torch.float32)

        # The TorchConnector will accept (batch, n_qubits) tensor and returns (batch, out_dim).
        q_out = self.qnn_torch(q_in)  # often returns shape (batch,) or (batch,1)

        # Normalize q_out shape to (batch, 1)
        if q_out.dim() == 1:
            q_out = q_out.unsqueeze(-1)
        elif q_out.dim() == 2 and q_out.shape[1] != 1:
            # If qnn returns multiple observables, you could reduce or keep them.
            # Here we keep only the first observable for simplicity.
            q_out = q_out[:, :1]

        # Move q_out to the model device for the rest of the head
        if model_device.type == "cuda":
            q_out = q_out.to(model_device)

        logits = self.post_qnn(q_out)
        return logits

# Instantiate model
model = HybridQCNN(qnn=qnn, n_qubits=n_qubits, num_classes=num_classes, image_size=image_size)

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
    train_acc = correct / total
    
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
    val_acc = val_correct / val_total
    
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    t2 = time.time()
    print(f"Time cost for validation epoch {epoch} is {t2-t1:.1f}s")

    scheduler.step(val_loss)

    print(f"Epoch {epoch}/{epochs}  Train Loss {train_loss:.4f}  Train Acc {train_acc:.4f}  "
          f"Val Loss {val_loss:.4f}  Val Acc {val_acc:.4f}")

    # show the best
    if val_acc > best_val_acc:
        best_val_acc = val_acc

current_time = datetime.now()
print(f"\nTraining complete at {current_time} - Best Val Acc: {best_val_acc:.4f}")



