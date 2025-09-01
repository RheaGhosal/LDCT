import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from collections import Counter
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler

from models import ResNetClassifier, UNetDenoiser

# ----- CONFIG -----

BATCH_SIZE = 8
EPOCHS = 52
DOSE_LEVEL = 5
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = '/storage/LDCT/denoised_dataset'
MODEL_PATH = '/storage/LDCT/saved_models/classifier.pt'
CHECKPOINT_PATH = '/storage/LDCT/saved_models/classifier_checkpoint.pt'
DENOISER_PATH = '/storage/LDCT/saved_models/denoiser.pt'

# ----- TRANSFORM -----

transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))
])

# ----- LOAD DENOISER -----

print("🔹 Loading denoiser...")
denoiser = UNetDenoiser().to(DEVICE)
denoiser.load_state_dict(torch.load(DENOISER_PATH, map_location=DEVICE))
denoiser.eval()
print("✅ Denoiser loaded.")

# ----- DATASET CLASS -----

class LDCTClassifierOnDenoiserDataset(Dataset):
    def __init__(self, split='train', dose_level=5, transform=None, denoiser=None):
        self.split = split
        self.dose_level = dose_level
        self.transform = transform
        self.denoiser = denoiser

        # Find noisy images for the split
        split_dir = os.path.join(DATA_ROOT, split)

        self.image_files = []
        self.mask_files = []

        for root, _, files in os.walk(split_dir):
            if 'image.npz' in files and 'mask.npz' in files:
                self.image_files.append(os.path.join(root, 'image.npz'))
                self.mask_files.append(os.path.join(root, 'mask.npz'))

        assert len(self.image_files) > 0, f"No images found in {split_dir}."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the noisy (low-dose) input image
        img_np = np.load(self.image_files[idx])["image"].astype(np.float32)
        mask_np = np.load(self.mask_files[idx])["mask"].astype(np.float32)

        img_np = np.nan_to_num(img_np, nan=0.0, posinf=0.0, neginf=0.0)
        img_np = np.clip(img_np, 0.0, None)

        maxval = np.max(img_np)
        if maxval > 0:
            img_np /= maxval

        noisy_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # Pass through denoiser
        with torch.no_grad():
            denoised_tensor = self.denoiser(noisy_tensor).squeeze(0).cpu()

        if self.transform:
            denoised_tensor = self.transform(denoised_tensor)

        label = 1 if mask_np.sum() > 0 else 0

        return denoised_tensor, torch.tensor(label, dtype=torch.float32)

# ----- LOAD DATA -----

train_set = LDCTClassifierOnDenoiserDataset(
    split='train',
    dose_level=DOSE_LEVEL,
    transform=transform,
    denoiser=denoiser
)

# Compute sample weights for oversampling
labels = [train_set[i][1].item() for i in range(len(train_set))]
class_counts = Counter(labels)
weights = [1.0 / class_counts[label] for label in labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    sampler=sampler
)

print(f"✅ Training on {len(train_set)} samples.")
print("Training set class counts:", class_counts)

# Calculate pos_weight for BCE
weight_for_1 = class_counts[0] / (class_counts[1] + 1e-6)
print(f"Calculated pos_weight for BCE: {weight_for_1:.3f}")

class_weights = torch.tensor([1.0, weight_for_1], dtype=torch.float32).to(DEVICE)

# ----- MODEL -----

classifier = ResNetClassifier().to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

# ----- RESUME LOGIC -----

start_epoch = 0

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"🔄 Resuming from epoch {start_epoch}")
elif os.path.exists(MODEL_PATH):
    print(f"🔄 Loading model weights from {MODEL_PATH}")
    classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print("ℹ️ Starting training from scratch.")

# ----- TRAIN LOOP -----

classifier.train()
for epoch in range(start_epoch, EPOCHS):
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for denoised_img, labels in pbar:
        denoised_img = denoised_img.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = classifier(denoised_img).view(-1)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}")

    torch.save({
        "epoch": epoch,
        "model_state_dict": classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, CHECKPOINT_PATH)

torch.save(classifier.state_dict(), MODEL_PATH)
print(f"✅ Classifier saved to: {MODEL_PATH}")
