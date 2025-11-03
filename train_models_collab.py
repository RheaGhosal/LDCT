import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models import ResNetClassifier, UNetDenoiser
from dataset import LDCTStrokeDataset
from dataset_denoiser import LDCTDenoiserDataset

# Paths
saved_model_dir = "saved_models"
os.makedirs(saved_model_dir, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 5
DOSE_LEVEL = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Classifier Training ===

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set_cls = LDCTStrokeDataset(
    split='train',
    dose_level=DOSE_LEVEL,
    transform=transform
)

train_loader_cls = DataLoader(train_set_cls, batch_size=BATCH_SIZE, shuffle=True)

classifier = ResNetClassifier().to(DEVICE)
criterion_cls = nn.BCEWithLogitsLoss()
optimizer_cls = optim.Adam(classifier.parameters(), lr=1e-4)

print("✅ Starting Classifier Training...")
for epoch in range(EPOCHS):
    classifier.train()
    running_loss = 0.0
    for inputs, labels in train_loader_cls:
        inputs, labels = inputs.to(DEVICE), labels.float().to(DEVICE)

        optimizer_cls.zero_grad()
        outputs = classifier(inputs).squeeze()
        loss = criterion_cls(outputs, labels)
        loss.backward()
        optimizer_cls.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Classifier Loss: {running_loss:.4f}")
    torch.save(classifier.state_dict(), f"{saved_model_dir}/classifier_epoch_{epoch+1}.pt")

torch.save(classifier.state_dict(), f"{saved_model_dir}/classifier.pt")
print("✅ Classifier model saved.")

# === Denoiser Training ===

# USE THE DENOISER DATASET instead of classification dataset
train_set_denoise = LDCTDenoiserDataset(split='train', dose_level=DOSE_LEVEL)
train_loader_denoise = DataLoader(train_set_denoise, batch_size=BATCH_SIZE, shuffle=True)

denoiser = UNetDenoiser().to(DEVICE)
criterion_denoise = nn.MSELoss()
optimizer_denoise = optim.Adam(denoiser.parameters(), lr=1e-4)

# Check for checkpoint to resume training
checkpoint_path = f"{saved_model_dir}/denoiser_checkpoint.pt"
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    denoiser.load_state_dict(checkpoint["model_state_dict"])
    optimizer_denoise.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"✅ Resuming denoiser training from epoch {start_epoch}")
else:
    print("✅ Starting Denoiser Training from scratch...")

for epoch in range(start_epoch, EPOCHS):
    denoiser.train()
    running_loss = 0.0
    for noisy_img, clean_img in train_loader_denoise:
        noisy_img = noisy_img.to(DEVICE)
        clean_img = clean_img.to(DEVICE)

        optimizer_denoise.zero_grad()
        outputs = denoiser(noisy_img)
        loss = criterion_denoise(outputs, clean_img)
        loss.backward()
        optimizer_denoise.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader_denoise)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Denoiser Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": denoiser.state_dict(),
        "optimizer_state_dict": optimizer_denoise.state_dict(),
    }, checkpoint_path)

# Save final denoiser model
torch.save(denoiser.state_dict(), f"{saved_model_dir}/denoiser.pt")
print("✅ Denoiser model saved.")

