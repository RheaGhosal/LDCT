# run_stress_tests.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Dummy CNN (replace with your ResNet / U-Net imports later)
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),  # assuming 128x128 input
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ---------------------------
# Noise injection (stress test)
# ---------------------------
def add_poisson_noise(images, lam):
    noisy = np.random.poisson(images * lam) / float(lam)
    return np.clip(noisy, 0, 1).astype(np.float32)   # <-- force float32

# ---------------------------
# Training loop
# ---------------------------
def train_model(model, train_loader, test_loader, device, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

# ---------------------------
# Main stress test runner
# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Dummy data (replace with real dataset loader later)
    X = np.random.rand(200, 1, 128, 128).astype(np.float32)  # <-- force float32
    y = np.random.randint(0, 2, size=(200,)).astype(np.int64)  # <-- force int64 for labels

    # Split
    train_X, test_X = X[:150], X[150:]
    train_y, test_y = y[:150], y[150:]

    results = {}
    for lam in [1, 5, 10, 20, 40]:
        print(f"\n=== Stress test with λ = {lam} ===")

        # Apply noise
        noisy_train = add_poisson_noise(train_X, lam)
        noisy_test = add_poisson_noise(test_X, lam)

        # Torch datasets (cast explicitly to torch.float32)
        train_ds = TensorDataset(torch.tensor(noisy_train, dtype=torch.float32),
                                 torch.tensor(train_y, dtype=torch.long))
        test_ds = TensorDataset(torch.tensor(noisy_test, dtype=torch.float32),
                                torch.tensor(test_y, dtype=torch.long))

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=16)

        # Model
        model = SimpleCNN().to(device)
        acc = train_model(model, train_loader, test_loader, device)
        results[lam] = acc
        print(f"Accuracy @ λ={lam}: {acc:.3f}")

    # Plot results
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xlabel("λ (Dose level)")
    plt.ylabel("Accuracy")
    plt.title("Stress Test: Accuracy vs Dose Level")
    plt.savefig("stress_test_results.png")
    print("\nSaved plot -> stress_test_results.png")

if __name__ == "__main__":
    main()

