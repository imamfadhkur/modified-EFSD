import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):
    return k // 2 if p is None else p

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))

class CBS(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=None):
        super(CBS, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class PCB(nn.Module):
    def __init__(self, in_channels):
        super(PCB, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.block(x)

class BCB(nn.Module):
    def __init__(self, in_channels):
        super(BCB, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x)

class ASFF(nn.Module):
    def __init__(self, channels):
        super(ASFF, self).__init__()
        self.reduce = nn.Conv2d(channels, channels, 1)
        self.weight = nn.Parameter(torch.ones(3), requires_grad=True)
        self.expand = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x1, x2, x3):
        size = x3.shape[-2:]
        x1 = F.interpolate(x1, size=size, mode='nearest')
        x2 = F.interpolate(x2, size=size, mode='nearest')

        w = torch.softmax(self.weight, dim=0)
        fused = w[0]*self.reduce(x1) + w[1]*self.reduce(x2) + w[2]*self.reduce(x3)
        return self.expand(fused)


class AFPN(nn.Module):
    def __init__(self, in_channels):  # expects [C3, C4, C5]
        super(AFPN, self).__init__()
        c3, c4, c5 = in_channels

        # Reduce + CA + CBS
        self.reduce3 = nn.Conv2d(c3, 128, 1)
        self.ca3 = ChannelAttention(128)
        self.cbs3 = CBS(128, 128)

        self.reduce4 = nn.Conv2d(c4, 256, 1)
        self.ca4 = ChannelAttention(256)
        self.cbs4 = CBS(256, 256)

        self.reduce5 = nn.Conv2d(c5, 512, 1)
        self.ca5 = ChannelAttention(512)
        self.cbs5 = CBS(512, 512)

        # ASFF 2 and 3
        self.asff2 = ASFF(256)
        self.asff3 = ASFF(512)

        # PCB x3 + BCB x3 (twice)
        self.pcb2 = nn.Sequential(PCB(256), PCB(256), PCB(256))
        self.bcb2 = nn.Sequential(BCB(256), BCB(256), BCB(256))

        self.pcb3 = nn.Sequential(PCB(512), PCB(512), PCB(512))
        self.bcb3 = nn.Sequential(BCB(512), BCB(512), BCB(512))

    def forward(self, c3, c4, c5):
        x3 = self.cbs3(self.ca3(self.reduce3(c3)))
        x4 = self.cbs4(self.ca4(self.reduce4(c4)))
        x5 = self.cbs5(self.ca5(self.reduce5(c5)))

        f2 = self.asff2(x3, x4, x5)
        f3 = self.asff3(x3, x4, x5)

        out2 = self.bcb2(self.pcb2(f2))
        out3 = self.bcb3(self.pcb3(f3))

        return out2, out3
