import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# 1. 환경 설정 및 모델 로드
def load_model(model_path, device):
    from normal_link_model import DeepNormalLinkAE
    model = DeepNormalLinkAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate(image_path, model_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(model_path, device)

    # 2. 이미지 전처리 (학습 때와 동일해야 함)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 시각화를 위한 역정규화(De-normalization) 함수
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean

    # 이미지 로드
    img_orig_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_orig_pil).unsqueeze(0).to(device)

    # 3. 모델 추론 (복원)
    with torch.no_grad():
        reconstructed = model(img_tensor)
        
        # MSE 계산 (이상 점수)
        mse_loss = nn.MSELoss(reduction='none')(reconstructed, img_tensor)
        anomaly_score = torch.mean(mse_loss).item()
        
        # 에러 맵 생성 (채널 평균)
        error_map = torch.mean(mse_loss, dim=1).squeeze().cpu().numpy()

    # 4. 시각화 (원본 | 복원 | 에러 맵)
    orig_np = denormalize(img_tensor.squeeze().cpu()).permute(1, 2, 0).numpy()
    recon_np = denormalize(reconstructed.squeeze().cpu()).permute(1, 2, 0).numpy()
    
    # 픽셀값 범위 제한 (0~1)
    orig_np = np.clip(orig_np, 0, 1)
    recon_np = np.clip(recon_np, 0, 1)

    plt.figure(figsize=(15, 5))
    
    # 원본 이미지
    plt.subplot(1, 3, 1)
    plt.title(f"Original Image\n(ID: {os.path.basename(image_path)})")
    plt.imshow(orig_np)
    plt.axis('off')

    # 복원 이미지
    plt.subplot(1, 3, 2)
    plt.title(f"Reconstructed\n(Normal Standard)")
    plt.imshow(recon_np)
    plt.axis('off')

    # 에러 맵 (어디가 다른가?)
    plt.subplot(1, 3, 3)
    plt.title(f"Anomaly Heatmap\nScore: {anomaly_score:.6f}")
    plt.imshow(error_map, cmap='jet') # 차이가 큰 곳이 붉게 표시됨
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    output_name = f"eval_res_{os.path.basename(image_path)}"
    plt.savefig(output_name)
    plt.show()
    print(f"✅ 결과가 저장되었습니다: {output_name} | Score: {anomaly_score}")

if __name__ == "__main__":
    # 테스트할 이미지 경로와 학습된 모델 경로를 입력하세요.
    TEST_IMG = "./data/pneumonia_data/test/PNEUMONIA/person1682_virus_2899.jpeg" 
    MODEL_WEIGHTS = "./mini_project/normal_link_v2_ep50.pth"
    
    if os.path.exists(TEST_IMG) and os.path.exists(MODEL_WEIGHTS):
        evaluate(TEST_IMG, MODEL_WEIGHTS)
    else:
        print("❌ 이미지나 모델 파일을 찾을 수 없습니다. 경로를 확인해주세요.")