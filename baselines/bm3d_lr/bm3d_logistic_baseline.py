# bm3d_logistic_baseline.py
import os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from bm3d import bm3d
from tqdm import tqdm

NPZ_PATH = "LDCT/dataset/lambda10.npz"

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = d["images"].astype(np.float32)  # (N,H,W) in [0,1]
    y = d["labels"].astype(np.int64)
    return X, y

X, y = load_npz(NPZ_PATH)

print("Applying BM3D denoising …")
deno = np.array([bm3d(x, sigma_psd=0.1) for x in tqdm(X)], dtype=np.float32)

# simple scalar features
means = deno.mean(axis=(1,2))
stds  = deno.std(axis=(1,2))
meds  = np.median(deno, axis=(1,2))
Xf = np.stack([means, stds, meds], axis=1)

Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.3, random_state=42, stratify=y)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr, ytr)
probs = clf.predict_proba(Xte)[:, 1]
preds = (probs > 0.5).astype(int)

acc  = accuracy_score(yte, preds)
sens = recall_score(yte, preds)              # positive class recall
spec = recall_score(yte, preds, pos_label=0) # negative class recall
auc  = roc_auc_score(yte, probs)

print(f"Accuracy:   {acc:.3f}")
print(f"Sensitivity:{sens:.3f}")
print(f"Specificity:{spec:.3f}")
print(f"AUC:        {auc:.3f}")
# -------------------------------------------------------
# 7. Save metrics to a text file for later retrieval
# -------------------------------------------------------
os.makedirs("results", exist_ok=True)
with open("results/bm3d_lr_lambda10.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Sensitivity: {sens:.4f}\n")
    f.write(f"Specificity: {spec:.4f}\n")
    f.write(f"AUC: {auc:.4f}\n")
print("Saved results to results/bm3d_lr_lambda10.txt")

