import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# 🚀 기존 모듈 임포트
from mimic_dataset import prepare_mimic_df, MimicDataset
from soo_net import SooNetEngine
from unet_lung_model import UNet

# ⚙️ 설정
SAVE_DIR = "analysis_results_2"
os.makedirs(SAVE_DIR, exist_ok=True)
REPORT_TXT_PATH = os.path.join(SAVE_DIR, "report_summary.txt") # 💡 텍스트 리포트 경로 추가

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
LABEL_ORDER = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "No Finding", "Pleural Effusion", 
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

# =====================================================================
# 1. 시각화 유틸리티 (Mask & Prediction 출력)
# =====================================================================
def save_visual_report(idx, original_img, mask, cam_img_path):
    """마스크, 원본, Grad-CAM을 하나의 리포트 이미지로 병합 저장"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original CXR")
    plt.imshow(original_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("U-Net Lung Mask")
    plt.imshow(mask, cmap='jet', alpha=0.8)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Grad-CAM Analysis")
    cam_img = plt.imread(cam_img_path)
    plt.imshow(cam_img)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/report_{idx}.png")
    plt.close()

# =====================================================================
# 2. 메인 정밀 평가 루프
# =====================================================================
def repair_fragmented_mask(mask_np, threshold=0.5, closing_kernel_size=15):
    """
    조각난 마스크 파편들을 이어붙이고 내부 구멍을 메웁니다.
    """
    # 1. 바이너리 마스크 생성
    binary_mask = (mask_np > threshold).astype(np.uint8)
    
    # 2. 모폴로지 '닫기(Closing)' 연산: 팽창(Dilation) 후 침식(Erosion)
    # 조각난 파편들 사이의 간격이 메워집니다.
    kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. 컨벡스 헐(Convex Hull) 적용 (선택 사항: 더 견고한 외곽선이 필요할 때)
    # 파편들을 모두 포함하는 가장 작은 볼록 다각형을 만듭니다.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask)
    final_mask = np.zeros_like(closed_mask)
    
    # 면적이 너무 작은 쓰레기 노이즈는 버리고 상위 덩어리들만 유지
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 1000: # 최소 면적 기준
            component_mask = (labels == i).astype(np.uint8)
            # 덩어리의 외곽 점들을 찾아 컨벡스 헐 생성
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                hull = cv2.convexHull(contours[0])
                cv2.drawContours(final_mask, [hull], -1, 1, thickness=cv2.FILLED)
                
    return final_mask.astype(float)

def run_visual_evaluation(num_samples=10):
    # 💡 1. 모델 로드
    engine = SooNetEngine(model_path="model/chexnet_1ch_448_chexpert_best.pth")
    unet = UNet(n_channels=3, n_classes=1).to(DEVICE)
    unet.load_state_dict(torch.load("aws_say2_project/unet_lung_mask_ep10.pth", map_location=DEVICE))
    unet.eval()

    # 💡 2. 데이터 준비
    test_df = prepare_mimic_df("data/mimic-iv-cxr/mimic_cxr_aug_validate.csv", 
                                "data/mimic-cxr-2.0.0-chexpert.csv", 
                                "data/mimic-iv-cxr/official_data_iccv_final")
    dataset = MimicDataset(test_df)

    print(f"🔍 총 {num_samples}개의 샘플 분석을 시작합니다. 결과는 '{REPORT_TXT_PATH}'에 저장됩니다.")

    # 텍스트 파일 오픈 (새로 쓰기 모드)
    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("=== SooNet(胸Net) Anatomy-Aware 정밀 판독 리포트 ===\n\n")

        for i in range(num_samples):
            pil_img, labels = dataset[i]
            img_path = test_df.iloc[i]['path']
            
            # (1) U-Net 마스크 생성
            input_for_unet = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                mask_output = torch.sigmoid(unet(input_for_unet))
                mask_np = mask_output.squeeze().cpu().numpy()
                clean_mask = repair_fragmented_mask(mask_np, closing_kernel_size=25)

            # (2) SooNet 추론 및 Grad-CAM
            predictions = engine.predict(img_path)
            prob_values = [predictions[label][0] for label in LABEL_ORDER]
            top_idx = np.argmax(prob_values)
            
            cam_path = f"{SAVE_DIR}/temp_cam_{i}.png"
            engine.get_cam_visualize(img_path, top_idx, cam_path)

            # (3) 리포트 텍스트 생성 및 저장
            header = f"\n📂 [Sample {i+1}] {os.path.basename(img_path)}\n"
            table_header = f"{'Disease Finding':<25} | {'Prob':<8} | {'GT':<8} | {'Status'}\n"
            line = "-" * 65 + "\n"
            
            # 콘솔 출력 및 파일 쓰기
            print(header + table_header + line, end="")
            f.write(header + table_header + line)

            for idx, label in enumerate(LABEL_ORDER):
                prob, _ = predictions[label]
                gt_val = labels[idx].item()
                gt_str = "POS" if gt_val == 1.0 else "NEG"
                
                # 🎯 정오답 판정 로직
                status = ""
                if prob >= 0.35: # 기태님이 설정하신 Threshold
                    status = "✅ CORRECT" if gt_val == 1.0 else "❌ FALSE POS"
                elif gt_val == 1.0:
                    status = "⚠️ MISS (FN)"

                if prob >= 0.4 or gt_val == 1.0:
                    row = f"{label:<25} | {prob:>7.2%} | {gt_str:<8} | {status}\n"
                    print(row, end="")
                    f.write(row)

            # (4) 시각적 리포트 저장
            save_visual_report(i, pil_img, clean_mask, cam_path)
            # 임시 CAM 파일 삭제 (선택 사항)
            if os.path.exists(cam_path): os.remove(cam_path)

    print(f"\n✨ 모든 분석이 완료되었습니다! 텍스트 리포트: {REPORT_TXT_PATH}")

if __name__ == "__main__":
    run_visual_evaluation(num_samples=5)