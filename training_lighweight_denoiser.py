import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from models import UNetDenoiser
from dataset_denoiser import LDCTDenoiserDataset

# Paths
saved_model_dir = "saved_models"
os.makedirs(saved_model_dir, exist_ok=True)

checkpoint_path = os.path.join(saved_model_dir, "denoiser_checkpoint.pt")

# Training settings
batch_size = 8
num_epochs = 5
learning_rate = 1e-4
dose_level = 5  # adjust as desired

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
full_dataset = LDCTDenoiserDataset(split='train', dose_level=dose_level)

# Optionally subset for speed during testing:
# subset_size = min(len(full_dataset), 200)
# dataset = Subset(full_dataset, list(range(subset_size)))
dataset = full_dataset

print(f"Training on {len(dataset)} samples.")

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = UNetDenoiser().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# === Check for existing checkpoint ===
start_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
else:
    print("ℹ️ Starting training from scratch.")

# === Training Loop ===
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for noisy_img, clean_img in pbar:
        noisy_img = noisy_img.to(device)
        clean_img = clean_img.to(device)

        output = model(noisy_img)
        loss = criterion(output, clean_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")

    # Save checkpoint after each epoch
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)

# Save final model as simpler single file
torch.save(model.state_dict(), os.path.join(saved_model_dir, "denoiser.pt"))
print("✅ Denoiser training complete.")

