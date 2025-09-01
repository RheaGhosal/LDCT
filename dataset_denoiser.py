import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob

class LDCTDenoiserDataset(Dataset):
    def __init__(self, split='train', data_root='dataset', dose_level=5, transform=None):
        self.split = split
        self.dose_level = dose_level
        self.transform = transform

        split_dir = os.path.join(
            '/storage/LDCT/',
            data_root,
            split
        )

        self.image_files = glob(os.path.join(split_dir, '**/image.npz'), recursive=True)
        assert len(self.image_files) > 0, f"No images found in {split_dir}."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
    # load noisy image (was originally your image file)
        noisy_np = np.load(self.image_files[idx])["image"].astype(np.float32)

    # load clean image and mask
        clean_data = np.load(self.image_files[idx])
        clean_np = clean_data["image"].astype(np.float32)

        if "mask" in clean_data:
            mask_np = clean_data["mask"].astype(np.float32)
        else:
            mask_np = np.zeros_like(clean_np, dtype=np.float32)

    # Clean NaNs or infs
        noisy_np = np.nan_to_num(noisy_np, nan=0.0, posinf=0.0, neginf=0.0)
        clean_np = np.nan_to_num(clean_np, nan=0.0, posinf=0.0, neginf=0.0)
        mask_np = np.nan_to_num(mask_np, nan=0.0, posinf=0.0, neginf=0.0)

    # Rescale
        if np.max(noisy_np) > 0:
            noisy_np /= np.max(noisy_np)
        if np.max(clean_np) > 0:
            clean_np /= np.max(clean_np)

        noisy_tensor = torch.from_numpy(noisy_np).unsqueeze(0).float()
        clean_tensor = torch.from_numpy(clean_np).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()

        return noisy_tensor, clean_tensor, mask_tensor
