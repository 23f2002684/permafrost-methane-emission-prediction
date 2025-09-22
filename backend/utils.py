import numpy as np
import torch
import torch.nn as nn


# UNet variant matching checkpoint-style names (no BatchNorm, explicit conv names)
class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Encoder
        self.enc1_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=True)
        self.enc1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)

        self.enc2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.enc2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)

        self.enc3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.enc3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)

        self.enc4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.enc4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1_conv1 = nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1, bias=True)
        self.dec1_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_conv1 = nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1, bias=True)
        self.dec2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3_conv1 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1, bias=True)
        self.dec3_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)

        # Final
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        # Encoder
        e1 = self.relu(self.enc1_conv1(x))
        e1 = self.relu(self.enc1_conv2(e1))

        p1 = self.pool(e1)
        e2 = self.relu(self.enc2_conv1(p1))
        e2 = self.relu(self.enc2_conv2(e2))

        p2 = self.pool(e2)
        e3 = self.relu(self.enc3_conv1(p2))
        e3 = self.relu(self.enc3_conv2(e3))

        p3 = self.pool(e3)
        e4 = self.relu(self.enc4_conv1(p3))
        e4 = self.relu(self.enc4_conv2(e4))

        # Decoder
        d1 = torch.cat([self.up1(e4), e3], dim=1)
        d1 = self.relu(self.dec1_conv1(d1))
        d1 = self.relu(self.dec1_conv2(d1))

        d2 = torch.cat([self.up2(d1), e2], dim=1)
        d2 = self.relu(self.dec2_conv1(d2))
        d2 = self.relu(self.dec2_conv2(d2))

        d3 = torch.cat([self.up3(d2), e1], dim=1)
        d3 = self.relu(self.dec3_conv1(d3))
        d3 = self.relu(self.dec3_conv2(d3))

        # Return logits; sigmoid in app.py
        return self.final_conv(d3)


def physics_filter(pred_mask: np.ndarray):
    """
    Accepts logits, probabilities in [0,1], or uint8 masks in {0,255}/{0,1}.
    Converts to a binary mask and returns a simple risk score and label.
    """
    arr = np.asarray(pred_mask)

    if arr.dtype == np.uint8:
        prob = (arr > 127).astype(np.float32) if arr.max() > 1 else arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32)
        if np.isfinite(arr).all() and 0.0 <= arr.min() <= 255.0 and 0.0 <= arr.max() <= 255.0:
            prob = np.clip(arr / 255.0, 0.0, 1.0)
        elif np.isfinite(arr).all() and 0.0 <= arr.min() and arr.max() <= 1.0:
            prob = arr
        else:
            prob = 1.0 / (1.0 + np.exp(-arr))

    bin_mask = (prob > 0.5).astype(np.uint8)

    eruption_pixels = int(bin_mask.sum())
    total_pixels = int(bin_mask.size)
    eruption_ratio = (eruption_pixels / total_pixels) if total_pixels > 0 else 0.0

    if eruption_ratio > 0.35:
        risk_level = "High"
        extra_message = "Extensive methane-unstable permafrost detected."
    elif eruption_ratio > 0.15:
        risk_level = "Medium"
        extra_message = "Moderate methane instability detected."
    else:
        risk_level = "Low"
        extra_message = "Minimal methane eruption risk."

    risk_score = eruption_ratio * 100.0
    return risk_score, risk_level, extra_message
