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

def conv_norm_act(in_channels, out_channels, kernel_size=3):

    # Convolution followed by instance norm and Leaky ReLU
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(negative_slope=0.01),
    )


class ContextModule(nn.Module):
    """Residual context block adapted from Isensee et al."""

    def __init__(self, channels, dropout=0.3):
        super().__init__()

        # Pre-activation: normalise and activate before convolution
        self.block = nn.Sequential(
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(
                channels, channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.Dropout2d(p=dropout),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(
                channels, channels,
                kernel_size=3, padding=1, bias=False
            ),
        )

    def forward(self, x):
        # Add input back to learned features
        return x + self.block(x)


class ImprovedDown(nn.Module):
    """Change channel count, then extract features with residual block."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        # Stride 2 reduces resolution without max pooling
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.context = ContextModule(out_channels)

    def forward(self, x):
        return self.context(self.conv(x))


class ImprovedUp(nn.Module):
    """Upsample, combine skip features, then localise anatomy."""

    def __init__(
        self, in_channels, skip_channels, out_channels, final=False
    ):
        super().__init__()

        self.up_conv = conv_norm_act(in_channels, skip_channels)

        if final:
            # Final level keeps combined features for segmentation
            self.localisation = conv_norm_act(
                skip_channels * 2, out_channels
            )
        else:
            # Spatial convolution then reduce channel count
            self.localisation = nn.Sequential(
                conv_norm_act(skip_channels * 2, skip_channels),
                conv_norm_act(
                    skip_channels, out_channels, kernel_size=1
                ),
            )

    def forward(self, x, skip):
        # Repeat features to match encoder resolution
        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = self.up_conv(x)

        # Combine deeper features with encoder detail
        x = torch.cat((skip, x), dim=1)
        return self.localisation(x)


class ImprovedUNet(nn.Module):
    """2D adaptation of Isensee et al. (2018), Section 2.1/Figure 1.

    Uses 2D convolutions, instance normalisation, channel dropout
    and three segmentation heads combined before loss calculation.
    """

    def __init__(self, in_channels=1, out_channels=6):
        super().__init__()

        # Encoder widths: 16, 32, 64, 128, 256
        self.down1 = ImprovedDown(in_channels, 16, stride=1)
        self.down2 = ImprovedDown(16, 32, stride=2)
        self.down3 = ImprovedDown(32, 64, stride=2)
        self.down4 = ImprovedDown(64, 128, stride=2)
        self.down5 = ImprovedDown(128, 256, stride=2)

        # Decoder reduces channels after combining skip features
        self.up4 = ImprovedUp(256, 128, 64)
        self.up3 = ImprovedUp(64, 64, 32)
        self.up2 = ImprovedUp(32, 32, 16)
        self.up1 = ImprovedUp(16, 16, 32, final=True)

        # Predict class scores at three decoder resolutions
        self.seg3 = nn.Conv2d(32, out_channels, kernel_size=1)
        self.seg2 = nn.Conv2d(16, out_channels, kernel_size=1)
        self.seg1 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encode and keep features for skip connections
        e1 = self.down1(x)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)
        e5 = self.down5(e4)

        # Decode back to original resolution
        d4 = self.up4(e5, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        # Combine segmentation scores from coarse to fine
        scores = self.seg3(d3)
        scores = F.interpolate(
            scores, size=d2.shape[2:], mode="nearest"
        )
        scores = scores + self.seg2(d2)

        scores = F.interpolate(
            scores, size=d1.shape[2:], mode="nearest"
        )
        scores = scores + self.seg1(d1)

        # Return logits for same loss function used by baseline
        return scores

def check_model():

    # Check output shape for HipMRI-sized input
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

def check_improved_model():
    torch.manual_seed(42)

    model = ImprovedUNet(in_channels=1, out_channels=6)
    model.train()

    images = torch.randn(1, 1, 256, 128)
    masks = torch.randint(0, 6, (1, 256, 128))

    outputs = model(images)

    print("Improved input shape:", images.shape)
    print("Improved output shape:", outputs.shape)
    print("Improved parameters:",
          sum(p.numel() for p in model.parameters()))

    assert outputs.shape == (1, 6, 256, 128)
    assert torch.isfinite(outputs).all()

    # Check loss can send gradients through model
    loss = F.cross_entropy(outputs, masks)
    loss.backward()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"No gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), (
                f"Non-finite gradient: {name}"
            )

    print("Improved forward and backward checks passed.")

if __name__ == "__main__":
    check_model()
    check_improved_model()