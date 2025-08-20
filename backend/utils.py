import numpy as np
import torch
import torch.nn as nn

#UNet Model Architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = CBR(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d1 = torch.cat([self.up1(e4), e3], dim=1)
        d1 = self.dec1(d1)
        d2 = torch.cat([self.up2(d1), e2], dim=1)
        d2 = self.dec2(d2)
        d3 = torch.cat([self.up3(d2), e1], dim=1)
        d3 = self.dec3(d3)

        return torch.sigmoid(self.final(d3))


#Physics-Based Analysis Function
def physics_filter(pred_mask):
    # Ensure binary mask is in 0/1 form
    mask = (pred_mask / 255).astype(np.uint8) if pred_mask.max() > 1 else pred_mask

    #Calculate eruption-prone pixel ratio
    eruption_pixels = np.sum(mask)
    total_pixels = mask.size
    eruption_ratio = eruption_pixels / total_pixels if total_pixels > 0 else 0

    #Risk scoring based on thresholds
    if eruption_ratio > 0.35:
        risk_level = "High"
        extra_message = "Extensive methane-unstable permafrost detected."
    elif eruption_ratio > 0.15:
        risk_level = "Medium"
        extra_message = "Moderate methane instability detected."
    else:
        risk_level = "Low"
        extra_message = "Minimal methane eruption risk."

    #Risk score (0-100)
    risk_score = eruption_ratio * 100

    return risk_score, risk_level, extra_message
