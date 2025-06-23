
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from dataset import LDCTStrokeDataset  # assumes you have a custom Dataset class
from models import ResNetClassifier, UNetDenoiser  # custom models

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits.squeeze())

            y_scores.extend(probs.cpu().numpy())
            y_pred.extend((probs > 0.5).int().cpu().numpy())
            y_true.extend(y.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    sensitivity = recall_score(y_true, y_pred)

    return accuracy, auc, sensitivity

def run_experiment(pipeline=1, dose_level=5, batch_size=32, model_paths=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_set = LDCTStrokeDataset(split='val', dose_level=dose_level, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if pipeline == 1:
        model = ResNetClassifier()
        model.load_state_dict(torch.load(model_paths['classifier']))
        model.to(device)
        return evaluate_model(model, test_loader, device)

    elif pipeline == 2:
        denoiser = UNetDenoiser()
        classifier = ResNetClassifier()
        denoiser.load_state_dict(torch.load(model_paths['denoiser']))
        classifier.load_state_dict(torch.load(model_paths['classifier']))
        denoiser.to(device)
        classifier.to(device)

        def pipeline2_inference(loader):
            classifier.eval()
            denoiser.eval()
            y_true, y_pred, y_scores = [], [], []

            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    y = y.to(device)
                    x_denoised = denoiser(x)
                    logits = classifier(x_denoised)
                    probs = torch.sigmoid(logits.squeeze())

                    y_scores.extend(probs.cpu().numpy())
                    y_pred.extend((probs > 0.5).int().cpu().numpy())
                    y_true.extend(y.cpu().numpy())

            accuracy = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_scores)
            sensitivity = recall_score(y_true, y_pred)

            return accuracy, auc, sensitivity

        return pipeline2_inference(test_loader)

# Example usage
if __name__ == "__main__":
    model_paths = {
        'classifier': 'saved_models/classifier.pt',
        'denoiser': 'saved_models/denoiser.pt'
    }

    for dose in [1, 5, 10, 20, 40]:
        acc1, auc1, sens1 = run_experiment(pipeline=1, dose_level=dose, model_paths=model_paths)
        acc2, auc2, sens2 = run_experiment(pipeline=2, dose_level=dose, model_paths=model_paths)
        print(f"Dose Level {dose}:")
        print(f"  Pipeline 1 - Accuracy: {acc1:.3f}, AUC: {auc1:.3f}, Sensitivity: {sens1:.3f}")
        print(f"  Pipeline 2 - Accuracy: {acc2:.3f}, AUC: {auc2:.3f}, Sensitivity: {sens2:.3f}")
