import os
import shutil

root = "/storage/LDCT/denoised_dataset"
splits_names = ["train", "val", "test"]

# Find all folders in the root
samples = sorted(os.listdir(root))

# Exclude any folders already named train, val, or test
samples = [s for s in samples if s not in splits_names]

print(f"Found {len(samples)} sample folders.")

# Define splits
n_total = len(samples)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

splits = {
    "train": samples[:n_train],
    "val": samples[n_train:n_train + n_val],
    "test": samples[n_train + n_val:]
}

for split, split_samples in splits.items():
    split_dir = os.path.join(root, split)
    os.makedirs(split_dir, exist_ok=True)
    
    for s in split_samples:
        src = os.path.join(root, s)
        dst = os.path.join(split_dir, s)
        if not os.path.exists(dst):
            shutil.move(src, dst)

print("✅ Done restructuring denoised dataset.")

