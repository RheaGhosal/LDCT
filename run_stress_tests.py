import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from dataset import LDCTStrokeDataset
from models import ResNetClassifier, UNetDenoiser  # make sure UNetDenoiser is in models.py
import os
import pandas as pd

RESULTS_FILE = "stress_results.csv"

# -----------------------
# Preprocessing helper
# -----------------------
def preprocess_batch(batch, device):
    data, target = batch

    # Convert image to float tensor
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float32)
    data = data.float().to(device)

    # Ensure shape [B, 1, H, W] for grayscale
    if data.ndim == 3:  # [B, H, W]
        data = data.unsqueeze(1)

    # Convert labels to tensor
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.long)
    target = target.to(device)

    return data, target


# -----------------------
# Training + Evaluation
# -----------------------
def train_classifier(model, loader, optimizer, criterion, device):
    model.train()
    for batch in loader:
        data, target = preprocess_batch(batch, device)

        optimizer.zero_grad()
        output = model(data)

        # Binary classification with BCEWithLogitsLoss
        loss = criterion(output.squeeze(), target.float())
        loss.backward()
        optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            data, target = preprocess_batch(batch, device)
            output = model(data)
            preds = (torch.sigmoid(output) > 0.5).long().squeeze()
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / total if total > 0 else 0


# -----------------------
# Pipeline 1 (Direct Classification)
# -----------------------
def run_pipeline1(train_loader, test_loader, device):
    model = ResNetClassifier(in_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(3):  # keep short for stress tests
        train_classifier(model, train_loader, optimizer, criterion, device)

    acc = evaluate(model, test_loader, device)
    return acc


# -----------------------
# Pipeline 2 (Denoising + Classification)
# -----------------------
def run_pipeline2(train_loader, test_loader, device):
    denoiser = UNetDenoiser(in_channels=1, out_channels=1).to(device)
    classifier = ResNetClassifier(in_channels=1).to(device)

    optimizer_d = optim.Adam(denoiser.parameters(), lr=1e-4)
    optimizer_c = optim.Adam(classifier.parameters(), lr=1e-4)
    criterion_d = nn.MSELoss()
    criterion_c = nn.BCEWithLogitsLoss()

    # Train denoiser for 3 epochs
    denoiser.train()
    for epoch in range(3):
        for batch in train_loader:
            noisy, target = preprocess_batch(batch, device)
            clean = noisy  # placeholder (since we don’t have HDCT pairs here)

            optimizer_d.zero_grad()
            output = denoiser(noisy)
            loss = criterion_d(output, clean)
            loss.backward()
            optimizer_d.step()

    # Train classifier on denoised images
    classifier.train()
    for epoch in range(3):
        for batch in train_loader:
            noisy, target = preprocess_batch(batch, device)
            with torch.no_grad():
                denoised = denoiser(noisy)

            optimizer_c.zero_grad()
            output = classifier(denoised)
            loss = criterion_c(output.squeeze(), target.float())
            loss.backward()
            optimizer_c.step()

    # Evaluate
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            noisy, target = preprocess_batch(batch, device)
            denoised = denoiser(noisy)
            output = classifier(denoised)
            preds = (torch.sigmoid(output) > 0.5).long().squeeze()
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / total if total > 0 else 0

def save_result(lam, acc1, acc2):
    """Append results to CSV file."""
    df = pd.DataFrame([{
        "lambda": lam,
        "Pipeline1_Accuracy": acc1,
        "Pipeline2_Accuracy": acc2
    }])
    if os.path.exists(RESULTS_FILE):
        df.to_csv(RESULTS_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(RESULTS_FILE, mode="w", header=True, index=False)
#------------------------
# Main stress-test loop
# -----------------------
from torchvision import transforms

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on", device)

    lam_values = [1, 5, 10, 20, 40]
    completed = set()
    if os.path.exists(RESULTS_FILE):
        prev = pd.read_csv(RESULTS_FILE)
        completed = set(prev["lambda"].tolist())
    results_p1, results_p2 = [], []

    # Add transform here (convert PIL → tensor)
    transform = transforms.Compose([
        transforms.ToTensor()  # converts to [C,H,W] float32 tensor in [0,1]
    ])

    for lam in lam_values:
        if lam in completed:
            print(f"Skipping λ={lam}, already completed.")
            continue
        print(f"\n=== Stress test with λ = {lam} ===")

        train_ds = LDCTStrokeDataset(split="train", dose_level=lam, transform=transform)
        test_ds = LDCTStrokeDataset(split="test", dose_level=lam, transform=transform)

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

        acc1 = run_pipeline1(train_loader, test_loader, device)
        acc2 = run_pipeline2(train_loader, test_loader, device)

        print(f"Pipeline 1 Accuracy @ λ={lam}: {acc1:.3f}")
        print(f"Pipeline 2 Accuracy @ λ={lam}: {acc2:.3f}")
        save_result(lam, acc1, acc2)
        results_p1.append((lam, acc1))
        results_p2.append((lam, acc2))

    # Plot
    lams, accs1 = zip(*results_p1)
    _, accs2 = zip(*results_p2)
    plt.plot(lams, accs1, marker="o", label="Pipeline 1 (Direct)")
    plt.plot(lams, accs2, marker="s", label="Pipeline 2 (Denoise+Classify)")
    plt.xlabel("Dose level (λ)")
    plt.ylabel("Accuracy")
    plt.title("Stress Test Results")
    plt.legend()
    plt.savefig("stress_test_results.png")
    print("Saved plot -> stress_test_results.png")
    print("\nAll stress tests finished. Results saved to", RESULTS_FILE)


if __name__ == "__main__":
    main()

