import torch
import torch.nn as nn
import torchvision.models as tvm
import os

class ResNetClassifier(nn.Module):
    def __init__(self, in_channels=1, weights_path=None):
        super().__init__()
        # Build a plain resnet18 (we’ll load fine-tuned weights later in the app)
        backbone = tvm.resnet18(weights=None)

        if in_channels == 1:
            # replace first conv to accept 1-channel
            conv1 = backbone.conv1
            new_conv = nn.Conv2d(
                1, conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=False
            )
            # init from RGB by averaging into the Parameter
            with torch.no_grad():
                avg = conv1.weight.detach().mean(dim=1, keepdim=True)  # [out,1,k,k]
                new_conv.weight.copy_(avg)
            backbone.conv1 = new_conv

        # binary head (one logit)
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.model = backbone

        # Optional backbone init if a path was provided
        if weights_path:
            try:
                if os.path.exists(weights_path):
                    state = torch.load(weights_path, map_location='cpu')
                    self.model.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"[WARN] Could not load initial weights from {weights_path}: {e}")

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

