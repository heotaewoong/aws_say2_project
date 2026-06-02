import torch
import torch.nn as nn

class SkipNormalLinkAE(nn.Module):
    def __init__(self):
        super(SkipNormalLinkAE, self).__init__()
        
        # --- Encoder: 레이어를 개별 정의하여 중간 출력을 보존 ---
        self.enc1 = self._conv_block(3, 32)    # 224 -> 112
        self.enc2 = self._conv_block(32, 64)   # 112 -> 56
        self.enc3 = self._conv_block(64, 128)  # 56 -> 28
        self.enc4 = self._conv_block(128, 256) # 28 -> 14
        self.enc5 = self._conv_block(256, 512) # 14 -> 7 (Bottleneck)
        
        # enc5의 출력이 들어옴 (512)
        self.dec5 = self._up_block(512, 256) 
        # dec5(256) + enc4(256) = 512가 입력으로 들어옴
        self.dec4 = self._up_block(512, 128) 
        # dec4(128) + enc3(128) = 256이 입력으로 들어옴
        self.dec3 = self._up_block(256, 64)
        # dec3(64) + enc2(64) = 128이 입력으로 들어옴
        self.dec2 = self._up_block(128, 32)
        # dec2(32) + enc1(32) = 64가 입력으로 들어옴
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # Encoder (중간 결과 저장)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Decoder (중간 결과를 Concatenate 하여 디테일 복원)
        d5 = self.dec5(e5)
        d4 = self.dec4(torch.cat([d5, e4], dim=1)) # 256 + 256 = 512
        d3 = self.dec3(torch.cat([d4, e3], dim=1)) # 128 + 128 = 256
        d2 = self.dec2(torch.cat([d3, e2], dim=1)) # 64 + 64 = 128
        out = self.dec1(torch.cat([d2, e1], dim=1)) # 32 + 32 = 64
        
        return out