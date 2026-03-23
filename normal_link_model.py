import torch
import torch.nn as nn

class NormalLinkAE(nn.Module):
    def __init__(self):
        super(NormalLinkAE, self).__init__()
        
        # 1. Encoder: 배치 정규화(BatchNorm)와 LeakyReLU 추가
        self.encoder = nn.Sequential(
            # Layer 1: 224 -> 112
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16), # [추가] 학습 안정화 및 속도 향상
            nn.LeakyReLU(0.2),  # [변경] 음수 영역의 정보를 살려 죽은 뉴런 방지
            
            # Layer 2: 112 -> 56
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Layer 3: 56 -> 28
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Layer 4 (Bottleneck): 28 -> 14
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1) # [추가] 과적합 방지 및 강건한 특징 학습 유도
        )
        
        # 2. Decoder: 대칭 구조 유지
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))