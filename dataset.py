
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random

class LDCTStrokeDataset(Dataset):
    def __init__(self, split='train', dose_level=5, transform=None, data_dir='data/processed'):
        self.split = split
        self.dose_level = dose_level
        self.transform = transform
        self.data_dir = os.path.join(data_dir, split)
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for label in ['stroke', 'no_stroke']:
            label_dir = os.path.join(self.data_dir, label)
            for fname in os.listdir(label_dir):
                if fname.endswith('.png'):
                    samples.append((os.path.join(label_dir, fname), 1 if label == 'stroke' else 0))
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
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')
        image = self.apply_poisson_noise(image)

        if self.transform:
            image = self.transform(image)

        return image, label
