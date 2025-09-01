import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random

class LDCTStrokeDataset(Dataset):
    def __init__(self, split='train', dose_level=5, transform=None, data_dir='/storage/LDCT/dataset'):
        self.split = split
        self.dose_level = dose_level
        self.transform = transform
        self.data_dir = os.path.join(data_dir, split)
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for root, _, files in os.walk(self.data_dir):
            if 'image.npz' in files and 'mask.npz' in files:
                image_path = os.path.join(root, 'image.npz')
                mask_path = os.path.join(root, 'mask.npz')

                mask_npz = np.load(mask_path)
                # Figure out the correct key
                if 'mask' in mask_npz:
                    mask = mask_npz['mask']
                elif 'arr_0' in mask_npz:
                    mask = mask_npz['arr_0']
                else:
                    continue  # skip if no usable data

                label = 1 if mask.sum() > 0 else 0
                samples.append((image_path, label))

        random.shuffle(samples)
        return samples

    def apply_poisson_noise(self, img):
        img_np = np.array(img).astype(np.float32) / 255.0
        scaled = img_np * self.dose_level
        noisy = np.random.poisson(scaled).astype(np.float32) / self.dose_level
        noisy = np.clip(noisy, 0., 1.)
        return Image.fromarray((noisy * 255).astype(np.uint8))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        img_npz = np.load(image_path)
        key = 'arr_0' if 'arr_0' in img_npz else list(img_npz.files)[0]
        img_array = img_npz[key].astype(np.float32) / 255.0

        image = Image.fromarray((img_array * 255).astype(np.uint8)).convert('L')
        image = self.apply_poisson_noise(image)

        if self.transform:
            image = self.transform(image)

        return image, label

