import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_utils import *

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # g: 디코더 신호, l: 인코더 스킵 커넥션 신호
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # 🚀 [핵심 수정] g1의 크기를 x1(스킵 커넥션)의 크기에 맞게 키워줍니다.
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1) # 👈 이제 크기가 똑같아서 더하기가 가능합니다!
        psi = self.psi(psi)
        return x * psi

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes , bilinear = False):
        super(UNet,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.in_conv = UNetConvBlock(self.n_channels , 64)
        self.Down1 = Down(64 , 128)
        self.Down2 = Down(128, 256)
        self.Down3 = Down(256, 512)
        self.Down4 = Down(512, 512)

        self.Ag1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.Ag2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.Ag3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.Ag4 = AttentionGate(F_g=64, F_l=64, F_int=32)

        self.Up1 = Up(512 + 512, 256 , self.bilinear)
        self.Up2 = Up(256 + 256, 128 , self.bilinear)
        self.Up3 = Up(128 + 128 , 64 , self.bilinear)
        self.Up4 = Up(64 + 64, 64 , self.bilinear)
        self.out_conv = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.in_conv(x)
        x2 = self.Down1(x1)
        x3 = self.Down2(x2)
        x4 = self.Down3(x3)
        x5 = self.Down4(x4)

        s4 = self.Ag1(g=x5, x=x4)
        x = self.Up1(x5, s4)

        s3 = self.Ag2(g=x, x=x3)
        x = self.Up2(x, s3)

        s2 = self.Ag3(g=x, x=x2)
        x = self.Up3(x, s2)

        s1 = self.Ag4(g=x, x=x1)
        x = self.Up4(x, s1)

        return self.out_conv(x)


if __name__ == '__main__':
    UNet(3,3)
