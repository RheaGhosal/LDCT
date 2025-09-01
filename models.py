import torch
import torch.nn as nn
import torchvision.models as models
import os

class ResNetClassifier(nn.Module):
    def __init__(self, in_channels=1, weights_path='/storage/LDCT/resnet18-f37072fd.pth'):
        super().__init__()
        
        # Print the path to confirm the weights file location
        print("Weights path:", weights_path)
        print("File exists:", os.path.exists(weights_path))
        
        # Load ResNet18 without pre-trained weights (pretrained=False)
        self.model = models.resnet18(pretrained=False)  # NO pre-trained weights, to avoid downloading
       
        # Load custom pre-trained weights manually
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path))
        else:
            raise FileNotFoundError(f"Weights file not found at: {weights_path}")
        
        # Modify the first convolutional layer for grayscale input
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the final fully connected layer for binary classification
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNetDenoiser(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d1 = self.up1(e3)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        return torch.sigmoid(self.out(d2))

