# Adapted from https://github.com/milesial/Pytorch-UNet
# Original code licensed under GNU GPL v3.0
# Modifications for the AI4MI project


import torch.nn as nn
from .unet_sections import * 

class _UNetCore(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, base_c=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # Encoder
        self.inc   = DoubleConv(n_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, (base_c * 16) // factor)

        # Decoder 
        self.up1   = Up(base_c * 16, base_c * 4, bilinear)  # concat 8C + 8C -> 16C, out 4C
        self.up2   = Up(base_c * 8, base_c * 2, bilinear)  # concat 4C + 4C -> 8C,  out 2C
        self.up3   = Up(base_c * 4, base_c, bilinear)  # concat 2C + 2C -> 4C,  out 1C
        self.up4   = Up(base_c * 2, base_c, bilinear)  # concat 1C + 1C -> 2C,  out 1C

        self.outc  = OutConv(base_c, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)

class UNet(nn.Module):
    """Drop-in compatible with ENet(in_dim, out_dim, **kwargs)."""
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        bilinear      = kwargs.get("bilinear", True)
        base_channels = kwargs.get("base_channels", 64)
        self.model    = _UNetCore(n_channels=in_dim, n_classes=out_dim,
                                  bilinear=bilinear, base_c=base_channels)
        print(f"> Initialized UNet ({in_dim=}->{out_dim=}, bilinear={bilinear}, base_c={base_channels})")

    def forward(self, x):
        return self.model(x)
