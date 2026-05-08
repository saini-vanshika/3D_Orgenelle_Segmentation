"""
3D U-Net Architecture for FIB-SEM Organelle Segmentation.
"""

import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    """3D double convolution block with residual connection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
        )
        self.shortcut = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x) + self.shortcut(x))


class UNet3D(nn.Module):
    """
    3D U-Net with encoder-decoder architecture and skip connections.

    Args:
        in_ch:          Number of input channels (1 for grayscale FIB-SEM)
        out_ch:         Number of output classes
        base_features:  Feature channels at first encoder level (doubles each level)
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 10, base_features: int = 32):
        super().__init__()

        f = base_features  # 32 → 64 → 128 → bottleneck 128

        # Encoder
        self.enc1 = DoubleConv3D(in_ch, f)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv3D(f, f * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv3D(f * 2, f * 4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = DoubleConv3D(f * 4, f * 4)

        # Decoder
        self.upconv3 = nn.ConvTranspose3d(f * 4, f * 4, 2, stride=2)
        self.dec3 = DoubleConv3D(f * 8, f * 2)

        self.upconv2 = nn.ConvTranspose3d(f * 2, f * 2, 2, stride=2)
        self.dec2 = DoubleConv3D(f * 4, f)

        self.upconv1 = nn.ConvTranspose3d(f, f, 2, stride=2)
        self.dec1 = DoubleConv3D(f * 2, f)

        # Output
        self.outc = nn.Conv3d(f, out_ch, 1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        x = self.pool1(e1)

        e2 = self.enc2(x)
        x = self.pool2(e2)

        e3 = self.enc3(x)
        x = self.pool3(e3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        x = self.upconv3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)

        return self.outc(x)
