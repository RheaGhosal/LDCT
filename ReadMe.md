## LDCT Stroke Classification
** AI-driven stroke diagnosis from simulated low-dose CT (LDCT) scans
This repository reproduces the full pipeline described in the paper “Simulating Low-Dose CT for Stroke Diagnosis”, including realistic dose simulation, UNet denoising, ResNet classification, and quantitative evaluation.

## Project Overview
Low-dose CT (LDCT) imaging reduces patient radiation exposure but introduces photon noise that obscures subtle stroke patterns.
This project simulates low-dose conditions from clean CT data, trains a deep learning denoiser, and evaluates stroke classification under realistic dose scenarios.

---

##  Repository Structure


```text
LDCT/
├─ app.py                        # Streamlit demo
├─ models.py                     # UNetDenoiser, ResNetClassifier (1-channel)
├─ dataset.py                    # I/O, transforms, dose simulation helpers
├─ split_dataset.py              # Patient-level 70/15/15 split
├─ train_denoiser_model.py       # Train UNet on (noisy → clean)
├─ train_models.py               # Train ResNet classifier
├─ run_experiments.py            # End-to-end eval across dose levels
├─ requirements.txt
├─ saved_models/                 # (created) model checkpoints
│  ├─ denoiser.pt
│  └─ classifier.pt
├─ dataset/                      # Clean (high-dose) CT slices (prepared)
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ denoised_dataset/             # Materialized UNet outputs (optional)
   ├─ train/
   ├─ val/
   └─ test
```

##   Dataset Preparation

Base Dataset

The clean CT images are derived from the RSNA Intracranial Hemorrhage Detection dataset on Kaggle, focusing on non-contrast brain CTs.

Download and extract the dataset:##  Target Dataset Layouts

### Clean / High-Dose (prepared)
Each **series UID** folder contains ordered slice subfolders with a single `image.npz` (key `arr_0`) and an optional `metadata.json`.

```text
dataset/
└─ test/
   └─ 2.25.116613827163187469445304378258728959191/
      ├─ 00000/
      │  └─ image.npz
      ├─ 00001/
      │  └─ image.npz
      ├─ 00002/
      │  └─ image.npz
      ├─ ...
      ├─ 00080/
      │  └─ image.npz
      └─ metadata.json
```
## Low-Dose Simulation (per paper)

The paper simulates photon-limited conditions using Poisson statistics at dose factors λ ∈ {1, 5, 10, 20, 40}. Conceptually:


I_LD(x, y) = (1 / λ) · Poisson(λ · I_clean(x, y))

In code (see dataset.py), low-dose variants are generated on-the-fly for training and explicitly during experiments.

## Environment
```text
cd /storage/LDCT
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

```

##Training
 A) Train the UNet Denoiser (noisy → clean)

Creates saved_models/denoiser.pt and can populate denoised_dataset/ when evaluated.

```text
python train_denoiser_model.py \
  --epochs 50 \
  --batch_size 8

```


B) Train the ResNet Classifier (stroke vs non-stroke)

Creates saved_models/classifier.pt. The classifier is 1-channel and trained on noisy or clean images depending on flags set in the script.

```text
python train_models.py \
  --epochs 40 \
  --batch_size 32

```

## Generate Denoised Slices (optional materialization)

While run_experiments.py can denoise on-the-fly, you may also materialize denoised slices into denoised_dataset/ to browse or reuse:

```text
# Example helper pattern (if exposed in your script):
# python train_denoiser_model.py --export_denoised --export_split {train,val,test} --out denoised_dataset
# Otherwise, run the denoiser pass embedded in run_experiments.py (next section).


```
The persisted structure mirrors dataset/:

```text
denoised_dataset/test/sample_08295/image.npz
denoised_dataset/test/sample_08295/mask.npz   # if masks/aux are produced

```

## Experiments & Evaluation

Run the full benchmark across dose levels for both pipelines:

Pipeline 1: Direct classification on low-dose images

Pipeline 2: Denoise (UNet) → classification (ResNet)


```text
python run_experiments.py
```

You’ll see per-dose metrics (example of typical output):


```text
Pipeline 1 - Dose 10 | Acc: 0.836 | AUC: 0.897 | Sensitivity: 0.712
Pipeline 2 - Dose 10 | Acc: 0.773 | AUC: 0.500 | Sensitivity: 0.000
  Denoising → MSE: 13879.6 | PSNR: 4.418 | SSIM: 0.077
```

run_experiments.py loads saved_models/classifier.pt and saved_models/denoiser.pt if present.

## Metrics (reported)

*  Classification: Accuracy, AUC (ROC), Sensitivity/Recall, Specificity, F1, (optionally) ECE
* Image Quality (denoiser): PSNR, SSIM, MSE
* Stress tests (as in paper): motion/ring artifacts and robustness deltas (if enabled)

## Reproducibility

Patient-level splits: 70/15/15 (seed = 42) via split_dataset.py

Determinism helpers:

```text
import random, numpy as np, torch
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
```


