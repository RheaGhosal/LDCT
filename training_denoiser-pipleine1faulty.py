import os
import numpy as np
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
LEARNING_RATE = 1e-4
dose_levels = [1, 5, 10, 20, 40]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVED_MODELS_DIR = '/storage/LDCT/saved_models'
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# ----- TRAINING LOOP FOR ALL DOSE LEVELS -----

for dose_level in dose_levels:
    print(f"\n=== Training Denoiser for dose {dose_level} ===")

    # Paths specific to this dose level
    MODEL_PATH = os.path.join(SAVED_MODELS_DIR, f"denoiser_dose{dose_level}.pt")
    CHECKPOINT_PATH = os.path.join(SAVED_MODELS_DIR, f"denoiser_dose{dose_level}_checkpoint.pt")

    # ----- LOAD DATASET -----

    train_set = LDCTDenoiserDataset(split='train', dose_level=dose_level)
    loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print(f"✅ Loaded training set with {len(train_set)} samples.")

    # ----- MODEL -----

    model = UNetDenoiser().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    start_epoch = 0

    # Load checkpoint if exists
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"🔄 Resuming from epoch {start_epoch}")
    elif os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"🔄 Loaded model from {MODEL_PATH}")
    else:
        print("ℹ️ Starting training from scratch.")

    # ----- TRAIN LOOP -----

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Dose {dose_level}]")

        for noisy, clean, mask in pbar:
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)
            mask = mask.to(DEVICE)

            output = model(noisy)

            # ----- REGION-WEIGHTED LOSS -----

            loss_mse = criterion(output, clean)
            region_mse = criterion(output * mask, clean * mask)
            loss = loss_mse + 10.0 * region_mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(loader)
        print(f"✅ Epoch {epoch+1} complete. Avg loss: {avg_loss:.6f}")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"💾 Checkpoint saved after epoch {epoch+1}")

    # ----- SAVE FINAL MODEL -----

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"🎉 Denoiser saved to: {MODEL_PATH}")
