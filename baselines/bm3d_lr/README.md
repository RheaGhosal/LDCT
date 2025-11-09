# BM3D + Logistic Regression (Classical Baseline)

Pilot baseline used in the paper’s “Classical Baselines” paragraph:
BM3D denoising → logistic regression on simple intensity features.

## Steps
1) Build λ=10 dataset into a single file:
```bash
python baselines/bm3d_lr/build_lambda_npz.py \
  --root ./dataset \
  --out LDCT_Project/data/dataset/lambda10.npz \
  --lambda_val 10 --resize 128


## Run Baseline
python baselines/bm3d_lr/bm3d_logistic_baseline.py


