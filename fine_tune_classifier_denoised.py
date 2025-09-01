import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from collections import Counter

from models import ResNetClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- CONFIG ----------
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4

DATA_ROOT = "/storage/LDCT/denoised_dataset"
MODEL_SAVE_PATH = "/storage/LDCT/saved_models/classifier_finetuned.pt"
CHECKPOINT_PATH = "/storage/LDCT/saved_models/classifier_finetuned_checkpoint.pt"

# ---------- DATASET ----------

class LDCTDenoisedClassifierDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.image_files = []
        self.mask_files = []

        for root, _, files in os.walk(DATA_ROOT):
            if 'image.npz' in files and 'mask.npz' in files:
                self.image_files.append(os.path.join(root, 'image.npz'))
                self.mask_files.append(os.path.join(root, 'mask.npz'))

        assert len(self.image_files) > 0, f"No denoised images found in {DATA_ROOT}"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_np = np.load(self.image_files[idx])["image"].astype(np.float32)
        mask_np = np.load(self.mask_files[idx])["mask"].astype(np.float32)

        img_np = np.clip(img_np, 0.0, 1.0)

        image_tensor = torch.from_numpy(img_np).unsqueeze(0)

        label = 1 if mask_np.sum() > 0 else 0

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, torch.tensor(label, dtype=torch.float32)

# ---------- TRANSFORMS ----------

transform = transforms.Normalize((0.5,), (0.5,))

# ---------- LOAD DATA ----------

dataset = LDCTDenoisedClassifierDataset(transform=transform)

# Compute class weights
labels = [dataset[i][1].item() for i in range(len(dataset))]
counts = Counter(labels)
weights = [1.0 / counts[label] for label in labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler
)

print(f"✅ Loaded {len(dataset)} denoised samples.")
print("Class counts:", counts)

weight_for_1 = counts[0] / (counts[1] + 1e-6)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight_for_1).to(DEVICE))

# ---------- MODEL ----------

classifier = ResNetClassifier().to(DEVICE)

# Load your existing classifier weights (optional)
PRETRAINED_PATH = "/storage/LDCT/saved_models/classifier.pt"

if os.path.exists(PRETRAINED_PATH):
    print(f"Loading pretrained weights from {PRETRAINED_PATH}")
    classifier.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))

optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

start_epoch = 0

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"🔄 Resuming from epoch {start_epoch}")
else:
    print("ℹ️ Starting fine-tuning from scratch.")


# ---------- TRAIN LOOP ----------

 
for epoch in range(start_epoch, EPOCHS):

    classifier.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for imgs, labels in pbar:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = classifier(imgs).view(-1)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}")

# ---------- SAVE ----------
#torch.save(classifier.state_dict(), MODEL_SAVE_PATH)
    torch.save({
    "epoch": epoch,
    "model_state_dict": classifier.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, CHECKPOINT_PATH)

print(f"✅ Checkpoint saved at epoch {epoch+1}")
print(f"✅ Fine-tuned classifier saved to: {MODEL_SAVE_PATH}")

