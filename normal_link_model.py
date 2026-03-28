import torch
import torch.nn as nn

class DeepNormalLinkAE(nn.Module):
    def __init__(self, latent_dim=512):
        super(DeepNormalLinkAE, self).__init__()
        
        # --- Encoder: 정보를 압축하는 과정 ---
        self.encoder = nn.Sequential(
            self._conv_block(3, 32),    # 224 -> 112
            self._conv_block(32, 64),   # 112 -> 56
            self._conv_block(64, 128),  # 56 -> 28
            self._conv_block(128, 256), # 28 -> 14
            self._conv_block(256, 512)  # 14 -> 7 (Bottleneck: 가장 압축된 상태)
        )

        # --- 2. True Bottleneck (일렬종대 압축) ---
        # 7*7*512 = 25088 개의 공간 정보를 완전히 파괴하고 512개의 핵심 벡터로만 압축합니다.
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(512 * 7 * 7, latent_dim)
        
        # 다시 25088 개로 복구
        self.fc_dec = nn.Linear(latent_dim, 512 * 7 * 7)

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
        batch_size = x.size(0)
        
        # 1. Convolutional Encoding
        x = self.encoder(x)
        
        # 2. Strict Bottleneck (공간 정보 파괴 및 극도 압축)
        x = self.flatten(x)
        latent = self.fc_enc(x)             # [Batch, 512] 벡터로 압축됨
        x = self.fc_dec(latent)             # [Batch, 25088] 벡터로 복원됨
        x = x.view(batch_size, 512, 7, 7)   # 다시 2D 맵으로 재조립
        
        # 3. Convolutional Decoding
        out = self.decoder(x)
        return out