import torch
import torch.nn as nn

class DeepNormalLinkAE(nn.Module):
    def __init__(self):
        super(DeepNormalLinkAE, self).__init__()
        
        # --- Encoder: 정보를 압축하는 과정 ---
        self.encoder = nn.Sequential(
            self._conv_block(3, 32),    # 224 -> 112
            self._conv_block(32, 64),   # 112 -> 56
            self._conv_block(64, 128),  # 56 -> 28
            self._conv_block(128, 256), # 28 -> 14
            self._conv_block(256, 512)  # 14 -> 7 (Bottleneck: 가장 압축된 상태)
        )
        
        # --- Decoder: 병목(Bottleneck) 정보만으로 정상 흉부를 상상해서 그리는 과정 ---
        self.decoder = nn.Sequential(
            self._up_block(512, 256),   # 7 -> 14
            self._up_block(256, 128),   # 14 -> 28
            self._up_block(128, 64),    # 28 -> 56
            self._up_block(64, 32),     # 56 -> 112
            # 마지막 레이어는 up_block 대신 직접 정의하여 3채널(RGB) 및 Sigmoid 적용
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()                # 112 -> 224
        )

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # 스킵 커넥션 없이 순수하게 압축(Encoder) 후 복원(Decoder)
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out