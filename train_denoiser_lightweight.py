import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_denoiser import LDCTDenoiserDataset
from models import UNetDenoiser

# ----- CONFIG -----
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVED_MODELS_DIR = '/storage/LDCT/saved_models'
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

dose_levels = [1, 2, 5, 10, 40]  # omit 20

# ----- TRAIN -----
for dose in dose_levels:
    print(f"\n🧪 Training Denoiser for Dose {dose}")

    model = UNetDenoiser().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    model_path = os.path.join(SAVED_MODELS_DIR, f"denoiser_dose{dose}.pt")
    checkpoint_path = os.path.join(SAVED_MODELS_DIR, f"denoiser_dose{dose}_checkpoint.pt")

    # Load checkpoint if exists
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("🔁 Resuming from checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        print("📖 Starting from scratch")

    # Dataset & Loader
    dataset = LDCTDenoiserDataset(split='train', dose_level=dose)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Training Loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS} | Dose {dose}")
        for noisy, clean, mask in pbar:
            noisy, clean, mask = noisy.to(DEVICE), clean.to(DEVICE), mask.to(DEVICE)
            output = model(noisy)

            loss_mse = criterion(output, clean)
            region_mse = criterion(output * mask, clean * mask)
            loss = loss_mse + 10.0 * region_mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"✅ Epoch {epoch+1} | Avg Loss: {running_loss/len(loader):.6f}")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, checkpoint_path)

    # Save final model
    torch.save(model.state_dict(), model_path)
    print(f"💾 Saved: {model_path}")

