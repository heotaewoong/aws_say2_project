import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from unet_lung_model import UNet  # 기태님의 모델 클래스 파일

def test_single_image(model_path, img_path, device):
    # 1. 모델 로드
    model = UNet(n_channels=3, n_classes=3).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

<<<<<<< Updated upstream
# =====================================================================
# 1. 평가용 엔진 클래스
# =====================================================================
class UNetEvaluator:
    def __init__(self, model_path, device, target_size=(512, 512)):
        self.device = device
        self.target_size = target_size
        
        # 모델 로드
        self.model = UNet(n_channels=3, n_classes=3).to(device)
        
        # 가중치 로드 (Best Model의 경우 dict 형태이므로 model_state_dict 접근)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Best 가중치 로드 완료 (Epoch: {checkpoint['epoch']}, Dice: {checkpoint['best_dice']:.4f})")
        else:
            self.model.load_state_dict(checkpoint)
            print("✅ 가중치 로드 완료")
            
        self.model.eval()

        # 전처리 파이프라인 (학습 때와 동일한 Normalize 적용)
        self.transform = A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
=======
    # 2. 이미지 및 마스크 로드 & 전처리
    # 원본 이미지
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
>>>>>>> Stashed changes
    
    # 모델 입력용 전처리 (256x256, Normalization)
    input_img = cv2.resize(orig_img, (256, 256))
    input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    input_tensor = input_tensor.to(device)

    # 3. 추론 (Inference)
    with torch.no_grad():
        output = model(input_tensor)
        # 확률이 가장 높은 클래스 선택 (0: 배경, 1: 폐, 2: 심장)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 4. 시각화 (원본, 정답, 예측)
    plt.figure(figsize=(18, 5))

    # (1) 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(input_img)
    plt.title("Original X-ray")
    plt.axis('off')

    # (3) Prediction (모델 예측)
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap='nipy_spectral')
    plt.title("Model Prediction")
    plt.axis('off')

    # (4) Overlay (예측값을 원본에 투영)
    overlay = input_img.copy()
    overlay[pred_mask == 1] = [255, 0, 0] # Lung -> Red
    overlay[pred_mask == 2] = [0, 0, 255] # Heart -> Blue
    combined = cv2.addWeighted(input_img, 0.7, overlay, 0.3, 0)
    
    plt.subplot(1, 3, 3)
    plt.imshow(combined)
    plt.title("Overlay (R: Lung, B: Heart)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- 실행부 ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_FILE = "aws_say2_project/unet_chexmask_best.pth" # 기태님의 베스트 모델 경로

# 데이터셋에서 샘플 한 장 경로 지정 (예시)
SAMPLE_IMG = r"/Users/skku_aws2_15/med/data/Lung Segmentation Data/Test/COVID-19/images/covid_19.png"

<<<<<<< Updated upstream
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original X-ray")
        plt.imshow(image_np)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Segmentation Overlay (Lung:Cyan, Heart:Magenta)")
        plt.imshow(result)
        plt.axis('off')

        if save_path:
            plt.savefig(save_path)
            print(f"🖼️ 시각화 결과 저장 완료: {save_path}")
        plt.show()

# =====================================================================
# 2. 실행부
# =====================================================================
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 🎯 설정
    BEST_MODEL_PATH = "aws_say2_project/unet_chexmask_best.pth"
    # 기태님이 말씀하신 CheXpert 이미지 경로
    TEST_IMAGE_PATH = "data/files/p13/p13001519/s51844642/6512b578-3174975a-e596785d-97bd0065-6e6bbb6c.jpg"
    
    evaluator = UNetEvaluator(BEST_MODEL_PATH, device)
    
    if os.path.exists(TEST_IMAGE_PATH):
        img_np, mask = evaluator.predict(TEST_IMAGE_PATH)
        evaluator.visualize_result(img_np, mask, save_path="eval_result_chexpert.png")
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {TEST_IMAGE_PATH}")
=======
test_single_image(MODEL_FILE, SAMPLE_IMG, DEVICE)
>>>>>>> Stashed changes
