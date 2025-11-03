import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import LDCTStrokeDataset
from models import UNetDenoiser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- CONFIG -----
DOSE_LEVEL = 5
BATCH_SIZE = 8
SAVE_ROOT = "/storage/LDCT/denoised_dataset"

# Add this transform:
transform = transforms.ToTensor()

# ----- LOAD DENOISER -----
denoiser = UNetDenoiser().to(DEVICE)
denoiser.load_state_dict(torch.load("/storage/LDCT/saved_models/denoiser.pt", map_location=DEVICE))
denoiser.eval()

# ----- DATASET -----
dataset = LDCTStrokeDataset(
    split='train',
    dose_level=DOSE_LEVEL,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ----- GENERATE DENOISED IMAGES -----
os.makedirs(SAVE_ROOT, exist_ok=True)

for i, (noisy_img, label) in enumerate(tqdm(loader, desc="Denoising train set")):
    noisy_img = noisy_img.to(DEVICE)
    with torch.no_grad():
        denoised = denoiser(noisy_img)

    denoised_np = denoised.cpu().numpy()  # shape: [B, 1, H, W]

    for j in range(denoised_np.shape[0]):
        idx = i * BATCH_SIZE + j
        out_dir = os.path.join(SAVE_ROOT, f"sample_{idx:05d}")
        os.makedirs(out_dir, exist_ok=True)

        # save denoised image
        np.savez_compressed(os.path.join(out_dir, "image.npz"),
                            image=denoised_np[j, 0])

        # save mask (label) as mask.npz
        np.savez_compressed(os.path.join(out_dir, "mask.npz"),
                            mask=np.ones_like(denoised_np[j, 0]) if label[j] == 1 else np.zeros_like(denoised_np[j, 0]))
