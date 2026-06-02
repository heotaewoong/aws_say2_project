import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from pytorch_msssim import ssim # [추가] SSIM 라이브러리

# 1. 환경 설정 및 모델 로드
def load_model(model_path, device):
    from normal_link_model import SkipNormalLinkAE # [수정] Skip 버전으로 변경
    model = SkipNormalLinkAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate(image_path, model_path):
    # SageMaker 환경 고려 (GPU 우선, 없으면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        return tensor * std + mean

    # 이미지 로드
    img_orig_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_orig_pil).unsqueeze(0).to(device)

    # 3. 모델 추론 (복원 및 에러 계산)
    with torch.no_grad():
        reconstructed = model(img_tensor)
        
        # [A] MSE 계산 (기존 방식)
        mse_loss = nn.MSELoss(reduction='none')(reconstructed, img_tensor)
        mse_score = torch.mean(mse_loss).item()
        mse_map = torch.mean(mse_loss, dim=1).squeeze().cpu().numpy()

        # [B] SSIM 계산 (추가 방식)
        # size_average=False를 주어 픽셀별 SSIM 지도를 얻습니다.
        ssim_val, ssim_map = ssim(reconstructed, img_tensor, data_range=1.0, size_average=False, full=True)
        ssim_score = 1 - ssim_val.item()
        ssim_map_np = (1 - ssim_map).mean(dim=1).squeeze().cpu().numpy()

    # 4. 시각화 (원본 | 복원 | MSE 맵 | SSIM 맵)
    orig_np = denormalize(img_tensor.squeeze()).permute(1, 2, 0).cpu().numpy()
    recon_np = denormalize(reconstructed.squeeze()).permute(1, 2, 0).cpu().numpy()
    
    orig_np = np.clip(orig_np, 0, 1)
    recon_np = np.clip(recon_np, 0, 1)

    plt.figure(figsize=(20, 5))
    
    # 1. 원본
    plt.subplot(1, 4, 1)
    plt.title(f"Original\n({os.path.basename(image_path)})")
    plt.imshow(orig_np)
    plt.axis('off')

    # 2. 복원
    plt.subplot(1, 4, 2)
    plt.title("Reconstructed\n(Skip-Connection)")
    plt.imshow(recon_np)
    plt.axis('off')

    # 3. MSE 에러 맵 (밝기 차이)
    plt.subplot(1, 4, 3)
    plt.title(f"MSE Anomaly Map\nScore: {mse_score:.6f}")
    plt.imshow(mse_map, cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # 4. SSIM 에러 맵 (구조/질감 차이)
    plt.subplot(1, 4, 4)
    plt.title(f"SSIM Texture Map\nScore: {ssim_score:.6f}")
    plt.imshow(ssim_map_np, cmap='magma') # 질감 차이는 magma 컬러맵이 잘 보입니다
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.tight_layout()
    output_name = f"eval_res_final_{os.path.basename(image_path)}"
    plt.savefig(output_name)
    plt.show()
    
    # 최종 판단 지표 (두 점수의 가중 합산 가능)
    final_combined_score = (0.5 * mse_score) + (0.5 * ssim_score)
    print(f"✅ 결과 저장: {output_name}")
    print(f"📊 MSE: {mse_score:.4f} | SSIM: {ssim_score:.4f} | Combined: {final_combined_score:.4f}")

if __name__ == "__main__":
    # SageMaker에서 다운로드한 파일명으로 경로 설정
    TEST_IMG = "./data/pneumonia.jpeg" 
    MODEL_WEIGHTS = "model.pth"
    
    if os.path.exists(TEST_IMG) and os.path.exists(MODEL_WEIGHTS):
        evaluate(TEST_IMG, MODEL_WEIGHTS)
    else:
        print("❌ 파일 경로를 확인해주세요.")