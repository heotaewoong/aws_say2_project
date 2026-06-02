import torch
import torch.nn.functional as F
from pytorch_msssim import ssim # pip install pytorch-msssim 필수

class HybridLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha # MSE와 SSIM의 반영 비율

    def forward(self, pred, target):
        # 1. MSE Loss (밝기 차이)
        mse_loss = F.mse_loss(pred, target)
        
        # 2. SSIM Loss (구조적 차이)
        # ssim 함수는 1(완전 일치) ~ 0(완전 다름)을 반환하므로 1에서 뺍니다.
        # 데이터 범위(data_range)는 Normalize를 했으므로 유동적이지만 보통 1.0~2.0 사이입니다.
        ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
        ssim_loss = 1 - ssim_val
        
        # 3. 최종 결합
        return (1 - self.alpha) * mse_loss + self.alpha * ssim_loss