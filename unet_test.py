import os
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from unet_lung_model import UNet # 기태님의 모델

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
            print(f"✅ Best 가중치 로드 완료 (Epoch: {checkpoint['epoch']}, Dice: {checkpoint['dice_score']:.4f})")
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
    
    def post_process_mask(self, mask, n_classes=3):
        clean_mask = np.zeros_like(mask)
        for cls in range(1, n_classes):
            cls_mask = (mask == cls).astype(np.uint8)
            # 덩어리(Label) 찾기
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cls_mask)
            if num_labels > 1:
                # 면적(stats[:, 4])이 가장 큰 순서대로 정렬
                largest_indices = np.argsort(stats[1:, 4]) + 1 
                
                # 폐는 상위 2개, 심장은 상위 1개만 복원
                keep_count = 2 if cls == 1 else 1
                for i in range(min(keep_count, len(largest_indices))):
                    idx = largest_indices[-(i+1)]
                    clean_mask[labels == idx] = cls
        return clean_mask

    def predict(self, image_path):
        # 이미지 로드 및 전처리
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        transformed = self.transform(image=image_np)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        pred_mask = self.post_process_mask(pred_mask)
            
        return image_np, pred_mask
    
    

    def visualize_result(self, image_np, pred_mask, save_path=None):
        """ 시각적 평가: 원본 이미지 위에 폐(Cyan)와 심장(Magenta) 오버레이 """
        h, w = image_np.shape[:2]
        # 마스크를 원본 이미지 크기로 복원
        mask_resized = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        overlay = image_np.copy()
        # 폐(1) 영역: Cyan색 (B:255, G:255, R:0)
        overlay[mask_resized == 1] = [0, 255, 255] 
        # 심장(2) 영역: Magenta색 (B:255, G:0, R:255)
        overlay[mask_resized == 2] = [255, 0, 255]

        # 투명도 조절하여 합성
        alpha = 0.4
        result = cv2.addWeighted(overlay, alpha, image_np, 1 - alpha, 0)

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
    BEST_MODEL_PATH = "aws_say2_project/unet_lung_heart_best.pth"
    # 기태님이 말씀하신 CheXpert 이미지 경로
    TEST_IMAGE_PATH = "data/files/p13/p13001519/s51844642/6512b578-3174975a-e596785d-97bd0065-6e6bbb6c.jpg"
    
    evaluator = UNetEvaluator(BEST_MODEL_PATH, device)
    
    if os.path.exists(TEST_IMAGE_PATH):
        img_np, mask = evaluator.predict(TEST_IMAGE_PATH)
        evaluator.visualize_result(img_np, mask, save_path="eval_result_chexpert.png")
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {TEST_IMAGE_PATH}")