Repository Structure
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
