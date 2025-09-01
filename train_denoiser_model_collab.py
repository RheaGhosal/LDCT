import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

from models import UNetDenoiser
from dataset import LDCTStrokeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basic preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load training set (no labels used for denoising)
train_set = LDCTStrokeDataset(split='train', dose_level=10, transform=transform)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

denoiser = UNetDenoiser().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)

os.makedirs("saved_models", exist_ok=True)

print("Training Denoiser...")
for epoch in range(5):  # You can increase this if needed
    running_loss = 0.0
    for noisy_imgs, _ in tqdm(train_loader):
        noisy_imgs = noisy_imgs.to(device)
        optimizer.zero_grad()
        output = denoiser(noisy_imgs)
        loss = criterion(output, noisy_imgs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/5], Loss: {running_loss:.4f}")
    torch.save(denoiser.state_dict(), f"saved_models/denoiser_epoch_{epoch+1}.pt")

# Optionally save final model under standard name
torch.save(denoiser.state_dict(), "saved_models/denoiser.pt")
print("✅ Denoiser model saved.")

