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

def prepare_mimic_df(aug_csv_path, chexpert_csv_path):
    # 1. 라벨 정보 로드 및 전처리
    labels_df = pd.read_csv(chexpert_csv_path)
    # 불확실성(-1.0) 처리 (U-Ones: 1.0으로 변환), 결측치는 0.0
    labels_df[LABEL_ORDER] = labels_df[LABEL_ORDER].fillna(0).replace(-1, 1)
    
    # 2. 분류된 이미지 리스트 CSV 로드
    aug_df = pd.read_csv(aug_csv_path)
    
    flat_data = []
    print(f"🔍 '{aug_csv_path}' 데이터 파싱 중...")
    
    for _, row in aug_df.iterrows():
        # AP, PA 컬럼에서 이미지 경로 추출
        for view_col in ['AP', 'PA']:
            try:
                img_list = ast.literal_eval(row[view_col])
                for img_path in img_list:
                    # 경로에서 study_id 추출 (예: .../s50084553/...)
                    study_id = int(img_path.split('/')[-2][1:])
                    
                    # 해당 study_id의 라벨 행 찾기
                    label_row = labels_df[labels_df['study_id'] == study_id]
                    if not label_row.empty:
                        flat_data.append({
                            'path': img_path,
                            'labels': label_row[LABEL_ORDER].values[0] # 지정된 순서대로 추출
                        })
            except:
                continue
                
    final_df = pd.DataFrame(flat_data)
    print(f"✅ 파싱 완료: 총 {len(final_df)}장의 이미지 확보")
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
        
        # 이미지 로드 (MIMIC-CXR-JPG 경로 구조 대응)
        image = Image.open(img_full_path).convert('RGB')
        label = torch.FloatTensor(row['labels'])
        
        if self.transform:
            image = self.transform(image)
        return image, label

def train():
    # --- [경로 설정] 기태님의 로컬 환경에 맞춰 수정하세요 ---
    IMG_ROOT = "data\mimic-iv-cxr\official_data_iccv_final" # 이미지 최상위 폴더 (files/p10/... 가 시작되는 곳)
    TRAIN_CSV = "data\mimic-iv-cxr\mimic_cxr_aug_train.csv"
    VAL_CSV = "data\mimic-iv-cxr\mimic_cxr_aug_validate.csv"
    CHEXPERT_CSV = "data/mimic-cxr-2.0.0-chexpert.csv"
    # --------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 준비 (라벨 순서가 보장된 DataFrame 생성)
    train_df = prepare_mimic_df(TRAIN_CSV, CHEXPERT_CSV)
    val_df = prepare_mimic_df(VAL_CSV, CHEXPERT_CSV)
    
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

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
        
        try:
            auroc = roc_auc_score(val_labels, val_preds, average='macro')
        except:
            auroc = 0.0
            
        print(f"✅ Epoch [{epoch+1}] Avg Loss: {epoch_loss/len(train_loader):.4f} | Val AUROC: {auroc:.4f}")
        scheduler.step(epoch_loss)

        # 성능이 좋아지면 모델 저장
        if auroc > best_auroc:
            best_auroc = auroc
            torch.save(model.state_dict(), "chexnet_mimic_best.pth")
            print(f"💾 Best Model Saved! AUROC: {auroc:.4f}")

if __name__ == "__main__":
    train()