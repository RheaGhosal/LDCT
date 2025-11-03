import os
import argparse
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from collections import Counter
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score

from models import ResNetClassifier, UNetDenoiser

# ----- CONFIG -----

BATCH_SIZE = 8
EPOCHS = 52
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = '/storage/LDCT/denoised_dataset'
SAVED_MODELS_DIR = '/storage/LDCT/saved_models'
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# ----- SEEDING -----

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----- TRANSFORM -----

transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))
])

# ----- DATASET CLASS -----

class LDCTClassifierOnDenoiserDataset(Dataset):
    def __init__(self, split='train', dose_level=5, transform=None, denoiser=None):
        self.split = split
        self.dose_level = dose_level
        self.transform = transform
        self.denoiser = denoiser

        split_dir = os.path.join(DATA_ROOT, split)
        self.image_files, self.mask_files = [], []

        for root, _, files in os.walk(split_dir):
            if 'image.npz' in files and 'mask.npz' in files:
                self.image_files.append(os.path.join(root, 'image.npz'))
                self.mask_files.append(os.path.join(root, 'mask.npz'))

        assert len(self.image_files) > 0, f"No images found in {split_dir}."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_np = np.load(self.image_files[idx])["image"].astype(np.float32)
        mask_np = np.load(self.mask_files[idx])["mask"].astype(np.float32)

        img_np = np.nan_to_num(img_np, nan=0.0, posinf=0.0, neginf=0.0)
        img_np = np.clip(img_np, 0.0, None)

        maxval = np.max(img_np)
        if maxval > 0:
            img_np /= maxval

        noisy_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # Pass through denoiser
        with torch.no_grad():
            denoised_tensor = self.denoiser(noisy_tensor).squeeze(0).cpu()

        if self.transform:
            denoised_tensor = self.transform(denoised_tensor)

        label = 1 if mask_np.sum() > 0 else 0
        return denoised_tensor, torch.tensor(label, dtype=torch.float32)

# ----- MAIN -----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dose", type=int, default=5, help="Dose level (1,2,5,10,20,40)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outcsv", type=str, default="results_classifier.csv", help="CSV to log results")
    args = parser.parse_args()

    seed_everything(args.seed)

    # Classifier always saved with fixed names
    MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "classifier.pt")
    CHECKPOINT_PATH = os.path.join(SAVED_MODELS_DIR, "classifier_checkpoint.pt")

    # Denoiser dose-specific
    DENOISER_PATH = os.path.join(SAVED_MODELS_DIR, f"denoiser_dose{args.dose}.pt")
    DENOISER_CKPT_PATH = os.path.join(SAVED_MODELS_DIR, f"denoiser_dose{args.dose}_checkpoint.pt")

    # ----- LOAD DENOISER -----
    print(f"🔹 Loading denoiser for dose {args.dose}...")
    denoiser = UNetDenoiser().to(DEVICE)
    if os.path.exists(DENOISER_PATH):
        denoiser.load_state_dict(torch.load(DENOISER_PATH, map_location=DEVICE))
    elif os.path.exists(DENOISER_CKPT_PATH):
        ckpt = torch.load(DENOISER_CKPT_PATH, map_location=DEVICE)
        denoiser.load_state_dict(ckpt["model_state_dict"])
    else:
        raise FileNotFoundError(f"No denoiser found for dose {args.dose}")
    denoiser.eval()
    print("✅ Denoiser loaded.")

    # ----- LOAD TRAIN DATA -----
    train_set = LDCTClassifierOnDenoiserDataset(
        split='train',
        dose_level=args.dose,
        transform=transform,
        denoiser=denoiser
    )

    labels = [train_set[i][1].item() for i in range(len(train_set))]
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)

    print(f"✅ Training on {len(train_set)} samples.")
    print("Training set class counts:", class_counts)

    weight_for_1 = class_counts[0] / (class_counts[1] + 1e-6)
    print(f"Calculated pos_weight for BCE: {weight_for_1:.3f}")

    class_weights = torch.tensor([1.0, weight_for_1], dtype=torch.float32).to(DEVICE)

    # ----- MODEL -----
    classifier = ResNetClassifier().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

    # ----- RESUME LOGIC -----
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        classifier.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"🔄 Resuming from epoch {start_epoch}")
    elif os.path.exists(MODEL_PATH):
        classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"📦 Loaded model from {MODEL_PATH}")
    else:
        print("🚀 Starting fresh training.")

    # ----- TRAIN LOOP -----
    classifier.train()
    for epoch in range(start_epoch, EPOCHS):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Dose {args.dose}]")

        for denoised_img, labels in pbar:
            denoised_img = denoised_img.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = classifier(denoised_img).view(-1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, CHECKPOINT_PATH)

    torch.save(classifier.state_dict(), MODEL_PATH)
    print(f"✅ Classifier saved to: {MODEL_PATH}")

    # ----- EVALUATE ON TEST -----
    test_set = LDCTClassifierOnDenoiserDataset(
        split="test",
        dose_level=args.dose,
        transform=transform,
        denoiser=denoiser
    )
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    classifier.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = classifier(imgs).view(-1)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    sens = recall_score(y_true, y_pred)
    spec = recall_score(y_true, y_pred, pos_label=0)
    prec = precision_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    print(f"[Dose {args.dose} | Seed {args.seed}] Test Acc={acc:.3f}, AUC={auc:.3f}, Sens={sens:.3f}, Spec={spec:.3f}")

    # ----- SAVE RESULTS TO CSV -----
    row = {
        "dose": args.dose,
        "seed": args.seed,
        "acc": acc, "auc": auc, "sens": sens, "spec": spec,
        "prec": prec, "f1": f1
    }
    exists = os.path.exists(args.outcsv)
    with open(args.outcsv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":
    main()

