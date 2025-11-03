import os
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset import LDCTStrokeDataset
from dataset_denoiser import LDCTDenoiserDataset

from models import ResNetClassifier, UNetDenoiser
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def evaluate_classifier(model, dataloader, device, threshold=0.02):
    model.eval()
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits.view(-1))

            y_scores.extend(probs.cpu().numpy())
            y_pred.extend((probs > threshold).int().cpu().numpy())
            y_true.extend(y.cpu().numpy())

    unique_labels = np.unique(y_true)
    unique_preds = np.unique(y_pred)
    print(f"Unique true labels: {unique_labels}")
    print(f"Unique predictions: {unique_preds}")

    if len(unique_labels) <= 1:
        print("⚠️ Only one class in test data. Metrics may be invalid.")
        auc = float('nan')
    else:
        auc = roc_auc_score(y_true, y_scores)

    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)

    return accuracy, auc, sensitivity


def evaluate_denoiser(denoiser, dataloader, device):
    denoiser.eval()

    mse_list = []
    psnr_list = []    
    ssim_list = []

    with torch.no_grad():
        for noisy, clean, mask in tqdm(dataloader, desc="Evaluating Denoiser"):
            noisy = noisy.to(device)
            clean = clean.to(device)

            output = denoiser(noisy)

            output_np = output.squeeze().cpu().numpy()
            clean_np = clean.squeeze().cpu().numpy()

            mse = np.mean((output_np - clean_np) ** 2)
            psnr = peak_signal_noise_ratio(clean_np, output_np, data_range=1.0)
            ssim = structural_similarity(clean_np, output_np, data_range=1.0)

            mse_list.append(mse)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

    avg_mse = np.mean(mse_list)
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    return avg_mse, avg_psnr, avg_ssim

def run_pipeline_1(dose_level, model_path, batch_size, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_set = LDCTStrokeDataset(
        split='val',
        dose_level=dose_level,
        transform=transform
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )

    model = ResNetClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    acc, auc, sens = acc, auc, sens = evaluate_classifier(model, test_loader, device, threshold=0.02)


    return acc, auc, sens

def run_pipeline_2(dose_level, denoiser_path, classifier_path, batch_size, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # load validation set with real labels
    val_set = LDCTStrokeDataset(
        split='val',
        dose_level=dose_level,
        transform=transform
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False
    )

    # load models
    denoiser = UNetDenoiser().to(device)
    denoiser.load_state_dict(torch.load(denoiser_path, map_location=device))

    classifier = ResNetClassifier().to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))

    # Evaluate denoiser quality
    denoiser_dataset = LDCTDenoiserDataset(
        split='val',
        dose_level=dose_level
    )
    denoiser_loader = DataLoader(
        denoiser_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    mse, psnr, ssim = evaluate_denoiser(denoiser, denoiser_loader, device)

    # Run classification on denoised images
    classifier.eval()
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for noisy_img, label in tqdm(val_loader, desc="Running Pipeline 2 Classification"):
            noisy_img = noisy_img.to(device)
            label = label.to(device)

            denoised_img = denoiser(noisy_img)
            denoised_img = torch.clamp(denoised_img, 0, 1)
            denoised_img = (denoised_img - 0.5) / 0.5  # match classifier input normalization

            if isinstance(denoised_img, torch.Tensor):
              # Skip ToTensor in transform — manually normalize instead
               denoised_img = (denoised_img - 0.5) / 0.5
            else:
                denoised_img = transform(denoised_img)

            logits = classifier(denoised_img)
            
            probs = torch.sigmoid(logits.view(-1))
           # print("Sample probs:", probs[:10].cpu().numpy())

            #y_pred.extend((probs > 0.5).int().cpu().numpy())
            threshold = 0.02
            y_pred.extend((probs > threshold).int().cpu().numpy())

            y_scores.extend(probs.cpu().numpy())
            y_true.extend(label.cpu().numpy())

    unique_labels = np.unique(y_true)
    unique_preds = np.unique(y_pred)
    print(f"Unique true labels: {unique_labels}")
    print(f"Unique predictions: {unique_preds}")

    if len(unique_labels) <= 1:
        print("⚠️ Only one class in test data. Metrics may be invalid.")
        auc = float('nan')
    else:
        auc = roc_auc_score(y_true, y_scores)

    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)

    return acc, auc, sens, mse, psnr, ssim

if __name__ == "__main__":
    model_paths = {
        'classifier': 'saved_models/classifier.pt',
    }

    dose_levels = [1,5,10,20,40]

    metrics_p1 = {'accuracy': [], 'auc': [], 'sensitivity': []}
    metrics_p2 = {'accuracy': [], 'auc': [], 'sensitivity': [], 'psnr': [], 'ssim': []}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dose in dose_levels:
        print(f"\n=== Running Pipeline 1 for dose {dose} ===")
        acc1, auc1, sens1 = run_pipeline_1(
            dose_level=dose,
            model_path=model_paths['classifier'],
            batch_size=4,
            device=device
        )
        print(f"Pipeline 1 - Dose {dose} | Acc: {acc1:.3f} | AUC: {auc1:.3f} | Sensitivity: {sens1:.3f}")

        metrics_p1['accuracy'].append(acc1)
        metrics_p1['auc'].append(auc1)
        metrics_p1['sensitivity'].append(sens1)

        gc.collect()
        torch.cuda.empty_cache()

        print(f"\n=== Running Pipeline 2 for dose {dose} ===")
        denoiser_path = f"saved_models/denoiser_dose{dose}.pt"
        acc2, auc2, sens2, mse, psnr, ssim = run_pipeline_2(
            dose_level=dose,
            denoiser_path=denoiser_path,
            classifier_path=model_paths['classifier'],
            batch_size=1,
            device=device
        )
        print(f"Pipeline 2 - Dose {dose} | Acc: {acc2:.3f} | AUC: {auc2:.3f} | Sensitivity: {sens2:.3f}")
        print(f"  Denoising → MSE: {mse:.6f} | PSNR: {psnr:.3f} | SSIM: {ssim:.3f}")

        metrics_p2['accuracy'].append(acc2)
        metrics_p2['auc'].append(auc2)
        metrics_p2['sensitivity'].append(sens2)
        metrics_p2['psnr'].append(psnr)
        metrics_p2['ssim'].append(ssim)

        gc.collect()
        torch.cuda.empty_cache()

    # Plot classification accuracy
    plt.figure(figsize=(10,5))
    plt.plot(dose_levels, metrics_p1['accuracy'], label="Pipeline 1 Accuracy", marker='o')
    plt.plot(dose_levels, metrics_p2['accuracy'], label="Pipeline 2 Accuracy", marker='o')
    plt.xlabel("Dose Level (λ)")
    plt.ylabel("Accuracy")
    plt.title("Classification Accuracy vs Dose Level")
    plt.legend()
    plt.grid()
    plt.savefig("classification_accuracy_vs_dose.png")
    plt.show()

    # Plot denoising metrics
    plt.figure(figsize=(10,5))
    plt.plot(dose_levels, metrics_p2['psnr'], label="Pipeline 2 PSNR", marker='o')
    plt.plot(dose_levels, metrics_p2['ssim'], label="Pipeline 2 SSIM", marker='o')
    plt.xlabel("Dose Level (λ)")
    plt.ylabel("Metric Value")
    plt.title("Denoising Performance vs Dose Level")
    plt.legend()
    plt.grid()
    plt.savefig("denoising_metrics_vs_dose.png")
    plt.show()
