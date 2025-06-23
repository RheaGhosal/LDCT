
# Low-Dose CT Stroke Classification

This repository implements a dual-pipeline deep learning framework for classifying strokes from simulated low-dose CT (LDCT) brain images. The goal is to support portable, low-radiation neuroimaging in ambulatory and emergency environments.

## Project Structure

- `dataset.py`: Loads preprocessed CT images, applies adaptive Poisson noise based on dose level.
- `models.py`: Contains two models:
  - `ResNetClassifier`: Binary classifier for stroke detection.
  - `UNetDenoiser`: Deep learning denoising model for restoring LDCT images.
- `run_experiments.py`: Benchmarks both pipelines (direct classification and denoising+classification) across different dose levels.

## Methodology

- **Data**: RSNA Intracranial Hemorrhage Detection Dataset from Kaggle.
- **Low-Dose Simulation**: Poisson noise added to high-dose CT using:
  ```
  I_LD(x, y) ~ (1/λ) * Poisson(λ * I(x, y))
  ```
- **Pipelines**:
  1. **Pipeline 1**: Noisy LDCT → ResNet-18 Classifier.
  2. **Pipeline 2**: Noisy LDCT → U-Net Denoiser → ResNet-18 Classifier.

## Usage

```bash
# Install dependencies
pip install torch torchvision scikit-learn numpy pillow tqdm

# Run experiments
python run_experiments.py
```

Make sure to place your trained model weights in `saved_models/`:
- `classifier.pt`
- `denoiser.pt`

## Citation

If you use this work, please cite:
```
Rhea Ghosal, Ronok Ghosal, Eileen Lou.
Simulating Low-Dose CT for Stroke Diagnosis: A Dual-Pipeline Deep Learning Framework for Portable Neuroimaging. 2025.
```
