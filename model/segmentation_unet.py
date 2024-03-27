# https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, use_batchnorm = True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if use_batchnorm :
            self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(mid_channels),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(out_channels),
                                             nn.ReLU(inplace=True))
        else :
            self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                             nn.LayerNorm([mid_channels, int(20480/mid_channels),int(20480/mid_channels)]),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1,bias=False),
                                             nn.LayerNorm([mid_channels, int(20480 / mid_channels), int(20480 / mid_channels)]),
                                             nn.ReLU(inplace=True))
    def forward(self, x):
        return self.double_conv(x)




class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, use_batchnorm = True ):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_batchnorm = use_batchnorm)

    def forward(self, x1, x2):

        # [1] x1
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # [2] concat
        x = torch.cat([x2, x1], dim=1) # concatenation
        # [3] out conv
        x = self.conv(x)
        return x

class Up_conv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, kernel_size=2, use_batchnorm=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x1):

        # [1] x1
        x = self.up(x1)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Segmentation_Head(nn.Module):

    def __init__(self,  n_classes, bilinear=False, use_batchnorm=True,
                 kernel_size = 2):
        super(Segmentation_Head, self).__init__()

        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up1 = (Up(1280, 640 // factor, bilinear, use_batchnorm))
        self.up2 = (Up(640, 320 // factor, bilinear, use_batchnorm))
        self.up3 = (Up_conv(320, kernel_size = kernel_size)) # 64 -> 154
        self.outc = (OutConv(160, n_classes))

    def forward(self, x16_out, x32_out, x64_out):

        x = self.up1(x16_out,x32_out)  # 1,640,32,32 -> 640*32
        x = self.up2(x, x64_out)    # 1,320,64,64
        x3_out = self.up3(x)        # 1,160,128,128
        logits = self.outc(x3_out)  # 1,4, 128,128
        return logits
