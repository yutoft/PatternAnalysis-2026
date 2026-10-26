"""Baseline U-Net adapted from my Demo 2 implementation.

Architecture reference:
Ronneberger et al. (2015), U-Net: Convolutional Networks
for Biomedical Image Segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two convolutions with batch normalisation and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    """Baseline U-Net for multiclass HipMRI segmentation"""

    def __init__(
        self,
        in_channels=1,
        out_channels=6,
        features=(64, 128, 256, 512),
    ):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder blocks
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(
            features[-1], features[-1] * 2
        )

        # Upsample then combine with encoder features
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature,
                    kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # One output score per class at each pixel
        self.final_conv = nn.Conv2d(
            features[0], out_channels, kernel_size=1
        )

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for index in range(0, len(self.ups), 2):
            x = self.ups[index](x)
            skip = skip_connections[index // 2]

            # Match spatial dimensions if input sizes require it
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            x = torch.cat((skip, x), dim=1)
            x = self.ups[index + 1](x)

        # Return logits
        # loss function handles probabilities
        return self.final_conv(x)


def check_model():
    """Check output shape for HipMRI-sized input."""
    torch.manual_seed(42)

    model = UNET(in_channels=1, out_channels=6)
    model.eval()

    images = torch.randn(1, 1, 256, 128)

    with torch.no_grad():
        outputs = model(images)

    print("Input shape:", images.shape)
    print("Output shape:", outputs.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    assert outputs.shape == (1, 6, 256, 128)
    assert torch.isfinite(outputs).all()
    print("Baseline forward-pass checks passed.")


if __name__ == "__main__":
    check_model()