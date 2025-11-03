Low-Dose CT (LDCT) Stroke Classification

A Dual-Pipeline Deep Learning Framework for Portable Neuroimaging

This repository implements a complete experimental framework for stroke classification from simulated low-dose CT (LDCT) brain scans, exploring the impact of dose reduction and denoising on diagnostic reliability.

Developed as part of an ongoing research project on AI-assisted portable neuroimaging, the code supports both direct and two-stage (denoise → classify) learning pipelines and reproduces the experimental setup described in the author’s paper draft “Low-Dose CT for Stroke Diagnosis: A Dual-Pipeline Deep Learning Framework for Portable Neuroimaging.”

 Research Overview

Low-dose CT imaging reduces radiation exposure but introduces severe noise, which can hinder stroke diagnosis.
This project evaluates two complementary deep learning strategies:

Pipeline 1 — Direct Classification:
A ResNet-18 classifier trained directly on simulated LDCT slices.

Pipeline 2 — Denoise + Classify:
A U-Net denoiser trained to restore LDCT images, followed by classification using the same ResNet-18 model.

The framework quantifies performance across multiple dose levels (λ = 1, 5, 10, 20, 40) and under artifact stress tests such as motion and ring artifacts.
Repository
LDCT/
├── dataset.py               # Handles loading, preprocessing, and noise simulation
├── split_dataset.py         # Splits dataset into train/val/test ensuring patient-level separation
├── train_models.py          # Trains ResNet-18 classifier on noisy LDCT images (Pipeline 1)
├── train_denoiser_model.py  # Trains U-Net denoiser (Pipeline 2, stage 1)
├── run_experiments.py       # Runs experiments across dose levels and stress tests
├── models.py                # Contains UNet and ResNet-18 architectures
├── utils/                   # Helper functions for metrics, visualization, and configs
├── data/                    # Dataset directory (see below)
└── saved_models/            # Directory for trained models (created automatically)


Project Overview

Portable CT scanners operate at reduced radiation doses, leading to high noise and degraded image quality.
This project evaluates two complementary pipelines:

Pipeline 1 (Direct Classification):

Trains a ResNet-18 classifier directly on noisy LDCT images.

Pipeline 2 (Denoise + Classify):

Uses a U-Net denoiser to restore images, then classifies with the same ResNet-18 model.

Both are benchmarked under dose levels λ ∈ {1, 5, 10, 20, 40} and artifact stress tests (motion, ring
Dataset
Source

This project uses the RSNA Intracranial Hemorrhage Detection dataset (public, de-identified):

 https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection

Preprocessing Steps

Convert DICOM to grayscale PNGs (128×128).

Normalize pixel values to [0, 1].

Label slices as hemorrhage (1) or non-hemorrhage (0).

Split by patient ID into train/validation/test (no cross-patient leakage).

Augment training data with rotations, flips, and affine transforms.

Simulate LDCT by adding Poisson noise:

​
ILD​(x,y)=1/λ​Poisson(λI(x,y))
where λ controls dose level (smaller λ ⇒ higher noise

Set Up Instructions - 
# 1. Clone the repository
git clone git@github.com:RheaGhosal/LDCT.git
cd LDCT

# 2. Create and activate environment
conda create -n ldct python=3.10 -y
conda activate ldct

# 3. Install dependencies
pip install -r requirements.txt

 Training and Evaluation
 Train the ResNet Classifier (Pipeline 1)
python train_models.py --dose 20 --epochs 50 --batch_size 32


Trains a ResNet-18 classifier directly on noisy LDCT images.

 Train the U-Net Denoiser (Pipeline 2 – Stage 1)
python train_denoiser_model.py --dose 20 --epochs 50 --batch_size 16


Trains the denoiser to reconstruct high-dose images from simulated low-dose inputs.

 Run the Full Experiment Suite
python run_experiments.py


This script evaluates both pipelines at multiple dose levels (λ ∈ {1, 5, 10, 20, 40}) and outputs performance metrics and image quality scores.


Metrics and Outputs
Classification Metrics

Accuracy

AUC (ROC)

Sensitivity / Specificity

F1-Score

Expected Calibration Error (ECE)

Image Quality (for denoiser)

PSNR (Peak Signal-to-Noise Ratio)

SSIM (Structural Similarity Index)

MSE (Mean Squared Error)

 Representative Results
Dose	Pipeline	Accuracy	AUC	Sensitivity	PSNR	SSIM
1×	Direct	0.564	0.728	0.883	—	—
5×	Direct	0.840	0.866	0.734	—	—
10×	Direct	0.836	0.897	0.712	—	—
1×	Denoise→Classify	0.773	0.500	0.000	4.418	0.077
5×	Denoise→Classify	0.678	0.603	0.117	4.978	0.360
20×	Denoise→Classify	0.755	0.441	0.054	4.978	0.360

 Observation:
The denoiser substantially improves visual quality but sometimes suppresses critical contrast cues used by the classifier — revealing a key trade-off between perceptual enhancement and diagnostic accuracy.

 Folder Structure After Training
LDCT/
├── dataset/                   # Original data
├── denoised_dataset/          # Denoised outputs (auto-created)
├── saved_models/
│   ├── denoiser.pt
│   └── classifier.pt
└── results/
    ├── metrics_dose10.json
    ├── plots/
    └── confusion_matrix.png

 Citation / Attribution

If referencing this work in your portfolio or application:

Ghosal, Rhea. Low-Dose CT for Stroke Diagnosis: A Dual-Pipeline Deep Learning Framework for Portable Neuroimaging.
Independent research project (2025). Code and experiments available at https://github.com/RheaGhosal/LDCT
.

 Acknowledgments

Ronok Ghosal – for dataset preparation and baseline modeling.

Mentor: E. Lou, Ph.D. – for guidance on medical imaging design and validation.

Data Source: RSNA Intracranial Hemorrhage Detection Challenge.




Both are benchmarked under dose levels λ ∈ {1, 5, 10, 20, 40} and artifact stress tests (motion, ring)
