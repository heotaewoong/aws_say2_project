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
    "Atelectasis", 
    "Cardiomegaly", 
    "Consolidation", 
    "Edema", 
    "Enlarged Cardiomediastinum", 
    "Fracture", 
    "Lung Lesion", 
    "Lung Opacity", 
    "No Finding", 
    "Pleural Effusion", 
    "Pleural Other", 
    "Pneumonia", 
    "Pneumothorax", 
    "Support Devices"
]

# 💡 img_root 매개변수를 추가로 받습니다.
def prepare_mimic_df(aug_csv_path, chexpert_csv_path, img_root):
    labels_df = pd.read_csv(chexpert_csv_path)
    labels_df[LABEL_ORDER] = labels_df[LABEL_ORDER].fillna(0).replace(-1, 1)
    
    aug_df = pd.read_csv(aug_csv_path)
    
    flat_data = []
    print(f"🔍 '{aug_csv_path}' 데이터 파싱 중...")
    
    missing_count = 0 # 누락된 파일이 몇 개인지 세어봅시다.

    for _, row in aug_df.iterrows():
        for view_col in ['AP', 'PA']:
            raw_string = str(row[view_col])
            if raw_string == 'nan':
                continue
                
            try:
                img_list = ast.literal_eval(raw_string)
                for img_path in img_list:
                    if 'p10' not in img_path:
                        continue
                    
                    # 🚀 [핵심 추가] 실제 파일 존재 여부 검사
                    # 만약 로컬 폴더에 'files' 폴더 없이 바로 'p10'이 있다면 아래 주석을 해제하세요.
                    # img_path = img_path.replace('files/', '') 
                    
                    img_full_path = os.path.join(img_root, img_path)
                    
                    # 로컬 하드디스크에 파일이 없으면 리스트에 넣지 않고 스킵합니다!
                    if not os.path.exists(img_full_path):
                        missing_count += 1
                        continue
                    # -----------------------------------------

                    study_id = int(img_path.split('/')[-2][1:])
                    label_row = labels_df[labels_df['study_id'] == study_id]
                    if not label_row.empty:
                        flat_data.append({
                            'path': img_path,
                            'labels': label_row[LABEL_ORDER].values[0]
                        })
            except:
                continue
                
    final_df = pd.DataFrame(flat_data)
    print(f"✅ 파싱 완료: 총 {len(final_df)}장의 실제 이미지 확보 (존재하지 않는 파일 {missing_count}장 제외됨)")
    return final_df

class MimicDataset(Dataset):
    def __init__(self, df, img_root, transform=None):
        self.df = df
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_full_path = os.path.join(self.img_root, row['path'])
        
        # 🚀 [핵심 수정] try-except 블록을 통해 손상된 이미지 예외 처리
        try:
            image = Image.open(img_full_path).convert('RGB')
        except Exception as e:
            # 이미지가 손상되어 열리지 않으면 콘솔에 경고만 띄우고
            print(f"⚠️ [경고] 손상된 이미지 건너뜀 (다음 이미지 대체): {img_full_path}")
            # 리스트의 다음 인덱스 이미지를 재귀적으로 불러옵니다.
            return self.__getitem__((idx + 1) % len(self))
            
        label = torch.FloatTensor(row['labels'])
        
        if self.transform:
            image = self.transform(image)
        return image, label

def train():
    # --- [경로 설정] 기태님의 로컬 환경에 맞춰 수정하세요 ---
    IMG_ROOT = "data" # 이미지 최상위 폴더 (files/p10/... 가 시작되는 곳)
    TRAIN_CSV = "data/mimic_cxr_aug_train.csv"
    VAL_CSV = "data/mimic_cxr_aug_validate.csv"
    CHEXPERT_CSV = "data/mimic-cxr-2.0.0-chexpert.csv"
    # --------------------------------------------------

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 데이터 준비 (라벨 순서가 보장된 DataFrame 생성)
    train_df = prepare_mimic_df(TRAIN_CSV, CHEXPERT_CSV, IMG_ROOT)
    val_df = prepare_mimic_df(VAL_CSV, CHEXPERT_CSV, IMG_ROOT)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = MimicDataset(train_df, IMG_ROOT, transform)
    val_ds = MimicDataset(val_df, IMG_ROOT, transform)
    
    # 로컬 사양에 맞춰 batch_size 조절 (메모리 부족 시 16으로 하향)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    # 1. 모델 초기화 (DenseNet-121)
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, 14) # 14개 질환
    model = model.to(device)

    # 2. 손실 함수 및 옵티마이저 (Multi-label 특화)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    print(f"🚀 학습 시작! Device: {device}")

    best_auroc = 0.0
    for epoch in range(10):
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

        # 검증 및 AUROC 계산
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                outputs = torch.sigmoid(model(imgs)) # 확률값으로 변환
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(lbls.numpy())
        
        val_preds = np.vstack(all_preds)
        val_labels = np.vstack(all_labels)
        
        # 🚀 [시니어의 최적화] 경고(Warning)를 방지하는 안전한 AUROC 계산법
        val_auroc_list = []
        for c in range(14): # 14개 질환 각각에 대해 평가
            # 해당 질환의 정답지(Label)에 0과 1이 모두 존재하는지 확인 (고유값이 2개 이상인가?)
            if len(np.unique(val_labels[:, c])) > 1:
                score = roc_auc_score(val_labels[:, c], val_preds[:, c])
                val_auroc_list.append(score)
        
        # 유효한 AUROC 점수들이 모였다면 평균을 내고, 아니면 0.0 처리
        if len(val_auroc_list) > 0:
            auroc = np.mean(val_auroc_list)
        else:
            auroc = 0.0
            
        print(f"✅ Epoch [{epoch+1}] Avg Loss: {epoch_loss/len(train_loader):.4f} | Val AUROC: {auroc:.4f}")
        scheduler.step(auroc)

        # 성능이 좋아지면 모델 저장
        if auroc > best_auroc:
            best_auroc = auroc
            torch.save(model.state_dict(), "chexnet_mimic_best.pth")
            print(f"💾 Best Model Saved! AUROC: {auroc:.4f}")

if __name__ == "__main__":
    train()