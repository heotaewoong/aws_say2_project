import os
import ast
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, roc_auc_score

LABEL_ORDER = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "No Finding", "Pleural Effusion", 
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

# =====================================================================
# 1. 의료 영상 전용 전처리 (학습할 때 썼던 CLAHE + Padding 그대로 적용)
# =====================================================================
class ChestXrayPreprocess:
    def __init__(self, target_size=(448, 448), clip_limit=2.0):
        self.target_size = target_size
        self.clip_limit = clip_limit

    def __call__(self, img):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        img_np = np.array(img.convert('L'))
        img_clahe = clahe.apply(img_np)
        img_pil = Image.fromarray(img_clahe).convert('RGB')
        img_padded = ImageOps.pad(img_pil, self.target_size, method=Image.BILINEAR, color=(0, 0, 0))
        return img_padded

# =====================================================================
# 2. MIMIC 데이터 파싱 (AP, PA 영상 및 14개 라벨 매칭)
# =====================================================================
def get_mimic_test_data(aug_csv_path, chexpert_csv_path, img_root):
    print("🔍 정답지(Labels) 및 이미지 경로 파싱 중...")
    labels_df = pd.read_csv(chexpert_csv_path)
    
    # 💡 불확실한 라벨(-1)을 비정상(1)으로 간주하고 결측치는 0으로 채움
    labels_df[LABEL_ORDER] = labels_df[LABEL_ORDER].fillna(0).replace(-1, 1)
    
    aug_df = pd.read_csv(aug_csv_path)
    test_data = []
    
    for _, row in aug_df.iterrows():
        # 파트너님 요청대로 AP, PA 뷰만 추출
        for view_col in ['AP', 'PA']:
            raw_string = str(row[view_col])
            if raw_string == 'nan' or raw_string == '[]':
                continue
                
            try:
                img_list = ast.literal_eval(raw_string)
                for img_path in img_list:
                    # img_path 예시: 'files/p10/p10003502/...jpg'
                    # img_root와 결합하여 파트너님의 실제 로컬 경로로 만듭니다.
                    img_full_path = os.path.join(img_root, img_path)
                    
                    # study_id 추출 ('.../s50414267/...' -> 50414267)
                    study_id = int(img_path.split('/')[-2][1:])
                    label_row = labels_df[labels_df['study_id'] == study_id]
                    
                    if not label_row.empty:
                        # 14개의 질환 여부를 [0.0, 1.0, ...] 형태의 배열로 추출
                        labels = label_row[LABEL_ORDER].values[0].astype(np.float32)
                        test_data.append({'path': img_full_path, 'labels': labels})
            except Exception:
                continue
                
    print(f"✅ 파싱 완료: 총 {len(test_data)}장의 평가용 이미지 세팅!")
    return test_data

class MimicDenseNetDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = self.data_list[idx]['path']
        labels = self.data_list[idx]['labels']
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # 혹시라도 깨진 이미지가 있으면 다음 이미지를 가져와 멈춤 현상 방지
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(labels, dtype=torch.float32)

# =====================================================================
# 3. 모델 평가 함수 (AUROC, Precision, Recall, F1)
# =====================================================================
def evaluate_densenet(weight_path, test_dataloader, num_classes=14):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 평가 시작! 디바이스: {device}")

    # 1. 모델 아키텍처 로드 (학습 시 사용한 DenseNet121 기준)
    print("🧠 DenseNet121 가중치 로드 중...")
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    # 마지막 출력단을 14개 질환에 맞게 수정
    model.classifier = nn.Linear(num_ftrs, num_classes)
    
    # 2. 파트너님의 학습된 가중치 덮어쓰기
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    print("📊 테스트 데이터 추론 중 (시간이 조금 걸릴 수 있습니다)...")
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            
            outputs = model(images)
            # Sigmoid를 거쳐 각 질환별 발병 확률(0.0 ~ 1.0) 계산
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # 확률이 50%를 넘으면 병이 있다고 판단(1)
            preds = (probs > 0.5).astype(int)
            targets = labels.numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets)

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    

    print("\n===================================================================")
    print("🏆 1. 각 질환별 AUROC Score (현업 1순위 지표)")
    print("===================================================================")
    auroc_list = []
    for i, class_name in enumerate(LABEL_ORDER):
        try:
            # 실제 정답에 양성과 음성이 모두 존재해야만 AUROC 계산이 가능합니다.
            auroc = roc_auc_score(all_targets[:, i], all_probs[:, i])
            auroc_list.append(auroc)
            print(f" - {class_name:27s}: {auroc:.4f}")
        except ValueError:
            print(f" - {class_name:27s}: 계산 불가 (Test Set에 정답이 0 또는 1뿐임)")
            
    if auroc_list:
        print(f"\n🔥 전체 평균 AUROC (Macro): {np.mean(auroc_list):.4f}")

    print("\n===================================================================")
    print("🏆 2. Classification Report (Precision, Recall, F1-Score)")
    print("===================================================================")
    print(classification_report(all_targets, all_preds, target_names=LABEL_ORDER, zero_division=0))

    from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
    
    print("\n===================================================================")
    print("🏆 3. Optimal Threshold 적용 후 최종 성적표 (AUROC, Precision, Recall, F1)")
    print("===================================================================")
    
    print(f"{'Disease':<27s} | {'AUROC':<6s} | {'Best_Th':<7s} | {'Precision':<9s} | {'Recall':<6s} | {'F1-Score':<8s}")
    print("-" * 75)
    
    optimal_thresholds = {}
    
    for i, class_name in enumerate(LABEL_ORDER):
        try:
            # 1. AUROC 계산 (임계값 무관)
            auroc = roc_auc_score(all_targets[:, i], all_probs[:, i])
            
            # 2. 최적의 임계값(Threshold) 찾기
            precisions, recalls, thresholds = precision_recall_curve(all_targets[:, i], all_probs[:, i])
            
            numerator = 2 * recalls * precisions
            denominator = recalls + precisions
            f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator!=0))
            
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            optimal_thresholds[class_name] = best_threshold
            
            # 3. 최적 임계값을 적용한 새로운 예측값 생성
            # (주의: thresholds 배열 길이는 precisions/recalls보다 1 짧을 수 있어, 새로 계산하는 것이 가장 정확합니다)
            opt_preds = (all_probs[:, i] >= best_threshold).astype(int)
            
            # 4. 새로운 예측값으로 Precision, Recall, F1 계산
            final_precision = precision_score(all_targets[:, i], opt_preds, zero_division=0)
            final_recall = recall_score(all_targets[:, i], opt_preds, zero_division=0)
            final_f1 = f1_score(all_targets[:, i], opt_preds, zero_division=0)
            
            print(f"{class_name:<27s} | {auroc:.4f} | {best_threshold:.4f}  | {final_precision:.4f}    | {final_recall:.4f} | {final_f1:.4f}")
            
        except ValueError:
            print(f"{class_name:<27s} |  N/A   |  N/A     |   N/A      |  N/A   |  N/A")

    print("-" * 75)

if __name__ == "__main__":
    # 💡 1. 경로 설정 (파트너님의 로컬 파일 경로를 기입하세요!)
    IMG_ROOT = "data/mimic-iv-cxr/official_data_iccv_final"
    VAL_CSV = "data/mimic-iv-cxr/mimic_cxr_aug_validate.csv"
    CHEXPERT_CSV = "data/mimic-cxr-2.0.0-chexpert.csv"
    
    # 압축 해제한 DenseNet 가중치 경로 (예: model.pth 또는 model.pt)
    WEIGHT_PATH = "model\chexnet_chexpert_frontal_best.pth" 
    
    # 💡 2. 데이터로더 준비
    test_transform = transforms.Compose([
        ChestXrayPreprocess(target_size=(448, 448)),
        transforms.ToTensor() 
    ])
    
    test_data_list = get_mimic_test_data(VAL_CSV, CHEXPERT_CSV, IMG_ROOT)
    
    # Mac 환경 프리징 방지를 위해 num_workers는 0 또는 2로 설정
    test_dataset = MimicDenseNetDataset(test_data_list, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 💡 3. 본격적인 평가 실행
    if len(test_dataset) > 0:
        evaluate_densenet(WEIGHT_PATH, test_loader, num_classes=14)
    else:
        print("🚨 에러: 테스트 데이터가 0장입니다. 경로를 다시 확인해 주세요!")