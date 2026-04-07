import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image, ImageOps
import pandas as pd
import os
import ast
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score

from mimic_dataset import prepare_mimic_df, MimicDataset
from chexpert_dataset import prepare_chexpert_df, ChexpertDataset
from soo_net import SooNetEngine

# 1. 데이터셋 스위치
USE_MIMIC = True  # True: MIMIC-CXR 사용 / False: CheXpert 사용

# 2. 모델 & 학습 하이퍼파라미터
TARGET_SIZE = (448, 448)    # 입력 이미지 해상도 (TXV 고해상도 방식)
BATCH_SIZE = 16             # 배치 사이즈 (메모리 부족 시 8로 하향)
NUM_WORKERS = 2             # 데이터 로더 워커 수 (Mac MPS 환경 고려 2 이하 권장)
LEARNING_RATE = 1e-4        # 초기 학습률
EPOCHS = 15                 # 총 학습 에포크 수
LR_PATIENCE = 2             # Validation 성능이 개선되지 않을 때 대기하는 에포크 수
NUM_CLASSES = 14            # 분류할 질환 개수

# 3. 데이터셋 및 가중치 저장 경로 자동 세팅
if USE_MIMIC:
    IMG_ROOT = "data" 
    TRAIN_CSV = "data/mimic_cxr_aug_train.csv"
    VAL_CSV = "data/mimic_cxr_aug_validate.csv"
    CHEXPERT_CSV = "data/mimic-cxr-2.0.0-chexpert.csv"
    SAVE_MODEL_NAME = "chexnet_1ch_448_mimic_best.pth"
else:
    IMG_ROOT = "data/chexpert" 
    TRAIN_CSV = "data/chexpert/train.csv"
    VAL_CSV = "data/chexpert/valid.csv"
    CHEXPERT_CSV = None  # CheXpert는 별도 정답지 불필요
    SAVE_MODEL_NAME = "chexnet_1ch_448_chexpert_best.pth"

LABEL_ORDER = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "No Finding", "Pleural Effusion", 
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

# =====================================================================
# 🚀 1. 전처리 클래스
# =====================================================================
class ChestXrayPreprocess:
    def __init__(self, target_size=(224, 224), clip_limit=2.0):
        self.target_size = target_size
        self.clip_limit = clip_limit

    def __call__(self, img):
        # Pickle 에러 방지를 위해 __call__ 내부에서 객체 생성
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        img_np = np.array(img.convert('L'))
        img_clahe = clahe.apply(img_np)
        img_pil = Image.fromarray(img_clahe).convert('RGB')
        img_padded = ImageOps.pad(img_pil, self.target_size, method=Image.BILINEAR, color=(0, 0, 0))
        return img_padded

class TXV_Transform:
    def __init__(self, target_size=(448, 448), is_train=False):
        self.target_size = target_size
        self.is_train = is_train

    def __call__(self, img):
        # 1. 크기 조정
        img = img.resize(self.target_size, Image.BILINEAR)
        
        # 2. (선택) 학습 시에만 가벼운 데이터 증강 적용
        if self.is_train:
            # 1채널 이미지이므로 밝기/대비 조절은 조심스럽게 접근하거나 회전 정도만 줍니다.
            img = transforms.RandomRotation(5)(img)
            
        # 3. 텐서 변환 (0.0 ~ 1.0)
        img_tensor = transforms.ToTensor()(img)
        
        # 4. TXV 공식 스케일링 (-1024.0 ~ +1024.0)
        img_tensor = (img_tensor * 2048.0) - 1024.0
        
        return img_tensor

# =====================================================================
# 🚀 3. 메인 학습 루프 (Train Loop)
# =====================================================================
def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 시작! Device: {device}")
    
    # --------------------------------------------------
    # 💡 1. 학습할 데이터셋 선택 (주석을 풀어서 사용하세요)
    # --------------------------------------------------
    
    if USE_MIMIC:
        train_df = prepare_mimic_df(TRAIN_CSV, CHEXPERT_CSV, IMG_ROOT)
        val_df = prepare_mimic_df(VAL_CSV, CHEXPERT_CSV, IMG_ROOT)
        TrainDatasetClass = MimicDataset
    else:
        train_df = prepare_chexpert_df(TRAIN_CSV, IMG_ROOT)
        val_df = prepare_chexpert_df(VAL_CSV, IMG_ROOT)
        TrainDatasetClass = ChexpertDataset

    # --------------------------------------------------
    # 💡 2. 트랜스폼 및 데이터로더 설정
    # --------------------------------------------------
    train_transform = TXV_Transform(target_size=TARGET_SIZE, is_train=True)
    val_transform = TXV_Transform(target_size=TARGET_SIZE, is_train=False)
    
    train_ds = TrainDatasetClass(train_df, train_transform)
    val_ds = TrainDatasetClass(val_df, val_transform)
    
    # 해상도가 커졌으므로 메모리를 고려해 batch_size를 16으로 하향 조정
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --------------------------------------------------
    # 💡 3. 모델 초기화 (1채널 고해상도 DenseNet-121)
    # --------------------------------------------------
    print("🧠 SooNetEngine을 통해 모델 아키텍처를 로드합니다...")
    engine = SooNetEngine(model_path=None)
    model = engine.model
    model.train()

    # 손실 함수 및 옵티마이저
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=LR_PATIENCE)

    # --------------------------------------------------
    # 💡 4. 학습 시작
    # --------------------------------------------------
    best_auroc = 0.0
    for epoch in range(EPOCHS): # 필요에 따라 에포크 수 조정
        model.train()
        epoch_loss = 0
        
        for i, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Batch [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # === 검증(Validation) 및 AUROC 계산 ===
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                outputs = torch.sigmoid(model(imgs)) 
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(lbls.numpy())
        
        val_preds = np.vstack(all_preds)
        val_labels = np.vstack(all_labels)
        
        print(f"\n📊 --- Epoch {epoch+1} Validation Report ---")
        print(f"{'Disease':<27s} | {'AUROC':<6s} | {'Best_Th':<7s} | {'Prec':<6s} | {'Recall':<6s} | {'F1':<6s}")
        print("-" * 75)

        val_auroc_list = []
        val_f1_list = []

        for c, class_name in enumerate(LABEL_ORDER): 
            if len(np.unique(val_labels[:, c])) > 1:
                # 1. AUROC
                auroc = roc_auc_score(val_labels[:, c], val_preds[:, c])
                val_auroc_list.append(auroc)
                
                # 2. Optimal Threshold 찾기
                precisions, recalls, thresholds = precision_recall_curve(val_labels[:, c], val_preds[:, c])
                
                # 💡 [시니어의 안전장치] precision_recall_curve는 threshold보다 배열 길이가 1개 깁니다.
                # 배열 길이를 맞추기 위해 마지막 극단값 제외 (IndexError 방지)
                precisions = precisions[:-1]
                recalls = recalls[:-1]
                
                numerator = 2 * recalls * precisions
                denominator = recalls + precisions
                f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator!=0))
                
                best_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_idx]
                
                # 3. 최적 임계값 적용 후 P, R, F1 계산
                opt_preds = (val_preds[:, c] >= best_threshold).astype(int)
                val_prec = precision_score(val_labels[:, c], opt_preds, zero_division=0)
                val_rec = recall_score(val_labels[:, c], opt_preds, zero_division=0)
                val_f1 = f1_score(val_labels[:, c], opt_preds, zero_division=0)
                
                val_f1_list.append(val_f1)
                
                print(f"{class_name:<27s} | {auroc:.4f} | {best_threshold:.4f}  | {val_prec:.4f} | {val_rec:.4f} | {val_f1:.4f}")
            else:
                print(f"{class_name:<27s} |  N/A   |  N/A     |  N/A   |  N/A   |  N/A")
        
        print("-" * 75)
        
        avg_auroc = np.mean(val_auroc_list) if len(val_auroc_list) > 0 else 0.0
        avg_f1 = np.mean(val_f1_list) if len(val_f1_list) > 0 else 0.0
            
        print(f"✅ Epoch [{epoch+1}/{EPOCHS}] Avg Loss: {epoch_loss/len(train_loader):.4f} | Macro AUROC: {avg_auroc:.4f} | Macro F1: {avg_f1:.4f}\n")
        
        # 모델 저장과 스케줄러는 여전히 임계값에 흔들리지 않는 절대 지표인 AUROC를 기준으로 합니다.
        scheduler.step(avg_auroc)

        if avg_auroc > best_auroc:
            best_auroc = avg_auroc
            torch.save(model.state_dict(), SAVE_MODEL_NAME)
            print(f"💾 Best Model Saved: {SAVE_MODEL_NAME} (AUROC: {avg_auroc:.4f})")

if __name__ == "__main__":
    train()