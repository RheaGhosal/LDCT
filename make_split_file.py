# make_split_file.py
import os, json
from sklearn.model_selection import train_test_split

# point this to the top-level folder where patient subfolders are stored
DATA_ROOT = "/storage/LDCT/data/dataset"

# Case 1: if dataset has folders like dataset/patient123/*.png
patients = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]

# Case 2: if files are like dataset/patient123_slice01.png
# patients = sorted(set(fname.split("_")[0] for fname in os.listdir(DATA_ROOT) if fname.endswith(".png")))

print(f"Found {len(patients)} patients")

# make reproducible split
train_ids, test_ids = train_test_split(patients, test_size=0.20, random_state=12345)
train_ids, val_ids  = train_test_split(train_ids, test_size=0.125, random_state=12345)  
# (0.125 × 0.80 ≈ 0.10 total)

split_dict = {
    "train": sorted(train_ids),
    "val":   sorted(val_ids),
    "test":  sorted(test_ids)
}

with open("patient_splits.json", "w") as f:
    json.dump(split_dict, f, indent=2)

print("Wrote patient_splits.json")

