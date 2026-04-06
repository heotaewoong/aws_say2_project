import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
import ast
import numpy as np
from sklearn.metrics import roc_auc_score

LABEL_ORDER = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "No Finding", "Pleural Effusion", 
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

# =====================================================================
# 🚀 1. 고해상도(448) & 1채널(Grayscale) 전용 TXV 전처리
# =====================================================================
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
# 📊 2-A. MIMIC-IV-CXR 전용 데이터셋
# =====================================================================
def prepare_mimic_df(aug_csv_path, chexpert_csv_path, img_root):
    labels_df = pd.read_csv(chexpert_csv_path)
    labels_df[LABEL_ORDER] = labels_df[LABEL_ORDER].fillna(0).replace(-1, 1)
    aug_df = pd.read_csv(aug_csv_path)
    
    flat_data = []
    print(f"🔍 MIMIC 데이터 파싱 중: {aug_csv_path}")
    missing_count = 0 

    for _, row in aug_df.iterrows():
        for view_col in ['AP', 'PA']:
            raw_string = str(row[view_col])
            if raw_string == 'nan' or not any(folder in raw_string for folder in ('p10', 'p11', 'p12', 'p13')):
                continue
                
            try:
                img_list = ast.literal_eval(raw_string)
                for img_path in img_list:
                    if not any(folder in img_path for folder in ('p10', 'p11', 'p12', 'p13')):
                        continue
                    
                    img_full_path = os.path.join(img_root, img_path)
                    if not os.path.exists(img_full_path):
                        missing_count += 1
                        continue

                    study_id = int(img_path.split('/')[-2][1:])
                    label_row = labels_df[labels_df['study_id'] == study_id]
                    if not label_row.empty:
                        flat_data.append({
                            'path': img_full_path,
                            'labels': label_row[LABEL_ORDER].values[0]
                        })
            except Exception:
                continue
                
    final_df = pd.DataFrame(flat_data)
    print(f"✅ MIMIC 파싱 완료: 총 {len(final_df)}장 (누락 {missing_count}장)")
    return final_df

class MimicDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_full_path = row['path']
        
        try:
            # 🚀 [핵심] 1채널 흑백(Grayscale) 모드로 이미지를 엽니다!
            image = Image.open(img_full_path).convert('L')
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
            
        label = torch.FloatTensor(row['labels'].astype(float))
        
        if self.transform:
            image = self.transform(image)
        return image, label

# =====================================================================
# 📊 2-B. CheXpert 전용 데이터셋
# =====================================================================
def prepare_chexpert_df(csv_path, img_root):
    df = pd.read_csv(csv_path)
    
    # CheXpert의 라벨들을 가져옵니다 (결측치는 0, 불확실(-1)은 1로 간주)
    df[LABEL_ORDER] = df[LABEL_ORDER].fillna(0).replace(-1, 1)
    
    # 프론탈 뷰(Frontal)만 사용할 경우 필터링 (선택 사항)
    if 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal']
    
    flat_data = []
    print(f"🔍 CheXpert 데이터 파싱 중: {csv_path}")
    missing_count = 0

    for _, row in df.iterrows():
        # CheXpert의 Path 컬럼 예시: 'CheXpert-v1.0-small/train/patient00001/...'
        img_full_path = os.path.join(img_root, row['Path'])
        
        if not os.path.exists(img_full_path):
            missing_count += 1
            continue
            
        flat_data.append({
            'path': img_full_path,
            'labels': row[LABEL_ORDER].values
        })
        
    final_df = pd.DataFrame(flat_data)
    print(f"✅ CheXpert 파싱 완료: 총 {len(final_df)}장 (누락 {missing_count}장)")
    return final_df

class ChexpertDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_full_path = row['path']
        
        try:
            # 🚀 [핵심] 1채널 흑백(Grayscale) 모드로 불러옵니다.
            image = Image.open(img_full_path).convert('L')
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
            
        label = torch.FloatTensor(row['labels'].astype(float))
        
        if self.transform:
            image = self.transform(image)
        return image, label

# =====================================================================
# 🚀 3. 메인 학습 루프 (Train Loop)
# =====================================================================
def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 시작! Device: {device}")
    
    # --------------------------------------------------
    # 💡 1. 학습할 데이터셋 선택 (주석을 풀어서 사용하세요)
    # --------------------------------------------------
    USE_MIMIC = True  # True면 MIMIC 사용, False면 CheXpert 사용
    
    if USE_MIMIC:
        IMG_ROOT = "data" 
        TRAIN_CSV = "data/mimic_cxr_aug_train.csv"
        VAL_CSV = "data/mimic_cxr_aug_validate.csv"
        CHEXPERT_CSV = "data/mimic-cxr-2.0.0-chexpert.csv"
        
        train_df = prepare_mimic_df(TRAIN_CSV, CHEXPERT_CSV, IMG_ROOT)
        val_df = prepare_mimic_df(VAL_CSV, CHEXPERT_CSV, IMG_ROOT)
        TrainDatasetClass = MimicDataset
    else:
        # CheXpert용 경로 설정
        IMG_ROOT = "data/chexpert" 
        TRAIN_CSV = "data/chexpert/train.csv"
        VAL_CSV = "data/chexpert/valid.csv"
        
        train_df = prepare_chexpert_df(TRAIN_CSV, IMG_ROOT)
        val_df = prepare_chexpert_df(VAL_CSV, IMG_ROOT)
        TrainDatasetClass = ChexpertDataset

    # --------------------------------------------------
    # 💡 2. 트랜스폼 및 데이터로더 설정
    # --------------------------------------------------
    train_transform = TXV_Transform(target_size=(448, 448), is_train=True)
    val_transform = TXV_Transform(target_size=(448, 448), is_train=False)
    
    train_ds = TrainDatasetClass(train_df, train_transform)
    val_ds = TrainDatasetClass(val_df, val_transform)
    
    # 해상도가 커졌으므로 메모리를 고려해 batch_size를 16으로 하향 조정
    batch_size = 16 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # --------------------------------------------------
    # 💡 3. 모델 초기화 (1채널 고해상도 DenseNet-121)
    # --------------------------------------------------
    model = models.densenet121(weights=None) # 입력 레이어를 바꿀 것이므로 초기 가중치 사용 안 함
    
    # [입구 공사] 1채널 흑백 이미지를 받도록 첫 번째 Conv 교체
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # [출구 공사] 14개 질환 예측
    model.classifier = nn.Linear(model.classifier.in_features, 14)
    model = model.to(device)

    # 손실 함수 및 옵티마이저
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    # --------------------------------------------------
    # 💡 4. 학습 시작
    # --------------------------------------------------
    best_auroc = 0.0
    for epoch in range(15): # 필요에 따라 에포크 수 조정
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
        
        val_auroc_list = []
        for c in range(14): 
            if len(np.unique(val_labels[:, c])) > 1:
                score = roc_auc_score(val_labels[:, c], val_preds[:, c])
                val_auroc_list.append(score)
        
        auroc = np.mean(val_auroc_list) if len(val_auroc_list) > 0 else 0.0
            
        print(f"✅ Epoch [{epoch+1}] Avg Loss: {epoch_loss/len(train_loader):.4f} | Val AUROC: {auroc:.4f}")
        scheduler.step(auroc)

        if auroc > best_auroc:
            best_auroc = auroc
            # 저장 이름으로 MIMIC인지 CheXpert인지 구분
            save_name = "chexnet_1ch_448_mimic_best.pth" if USE_MIMIC else "chexnet_1ch_448_chexpert_best.pth"
            torch.save(model.state_dict(), save_name)
            print(f"💾 Best Model Saved: {save_name} (AUROC: {auroc:.4f})")

if __name__ == "__main__":
    train()