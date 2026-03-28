import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageOps
import cv2
import numpy as np
import pandas as pd
import os
import ast
from loss import HybridLoss

LABEL_ORDER = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "No Finding", "Pleural Effusion", 
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

class ChestXrayPreprocess:
    """흉부 X-ray에 최적화된 전처리 파이프라인 (CLAHE + 비율 유지 패딩)"""
    # 💡 clip_limit=1.5로 낮춰서 노이즈를 줄이고 이미지를 조금 더 부드럽게 만들었습니다.
    def __init__(self, target_size=(224, 224), clip_limit=1.5):
        self.target_size = target_size
        self.clip_limit = clip_limit
        # ❌ 여기 있던 self.clahe 선언을 제거했습니다!

    def __call__(self, img):
        # 🚀 [핵심 해결책] __call__ 내부에서 매번 CLAHE 객체를 생성합니다. 
        # (객체 생성 비용이 매우 작아서 속도에는 전혀 지장이 없습니다)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        
        # 1. Grayscale 변환
        img_np = np.array(img.convert('L'))
        
        # 2. CLAHE 적용 (대비 극대화)
        img_clahe = clahe.apply(img_np)
        
        # 3. RGB 변환
        img_pil = Image.fromarray(img_clahe).convert('RGB')
        
        # 4. 비율 유지 패딩
        img_padded = ImageOps.pad(img_pil, self.target_size, method=Image.BILINEAR, color=(0, 0, 0))
        
        return img_padded

# 💡 img_root 매개변수를 추가로 받도록 수정
def get_mimic_ae_dfs(aug_csv_path, chexpert_csv_path, img_root):
    """MIMIC-CXR 데이터를 정상(No Finding)과 비정상(나머지)으로 분리하여 DataFrame으로 반환합니다."""
    labels_df = pd.read_csv(chexpert_csv_path)
    labels_df[LABEL_ORDER] = labels_df[LABEL_ORDER].fillna(0).replace(-1, 1)
    
    aug_df = pd.read_csv(aug_csv_path)
    
    normal_data = []
    abnormal_data = []
    print(f"🔍 '{aug_csv_path}' 파싱하여 정상/비정상 분류 중...")
    
    missing_count = 0 # 누락된 파일 카운트

    # 🚀 우리가 가진 폴더 목록 정의
    target_folders = ('p10', 'p11', 'p12', 'p13')
    
    for _, row in aug_df.iterrows():
        for view_col in ['AP', 'PA']:
            raw_string = str(row[view_col])
            
            # 빠른 필터링: 결측치이거나 target_folders가 문자열에 아예 없으면 건너뜀
            if raw_string == 'nan' or not any(folder in raw_string for folder in target_folders):
                continue
                
            try:
                img_list = ast.literal_eval(raw_string)
                for img_path in img_list:
                    # 리스트 내에서 다시 한 번 필터링
                    if not any(folder in img_path for folder in target_folders):
                        continue
                    
                    # 🚀 실제 파일 존재 여부 검사
                    img_full_path = os.path.join(img_root, img_path)
                    if not os.path.exists(img_full_path):
                        missing_count += 1
                        continue

                    study_id = int(img_path.split('/')[-2][1:])
                    label_row = labels_df[labels_df['study_id'] == study_id]
                    
                    if not label_row.empty:
                        is_normal = (label_row['No Finding'].values[0] == 1.0)
                        
                        if is_normal:
                            normal_data.append({'path': img_path})
                        else:
                            abnormal_data.append({'path': img_path})
            except:
                continue
                
    normal_df = pd.DataFrame(normal_data)
    abnormal_df = pd.DataFrame(abnormal_data)
    print(f"✅ 분류 완료: 정상 {len(normal_df)}장 / 비정상 {len(abnormal_df)}장 (없는 파일 {missing_count}장 제외)")
    return normal_df, abnormal_df

class MimicAutoencoderDataset(Dataset):
    def __init__(self, df, img_root, transform=None):
        self.df = df
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_full_path = os.path.join(self.img_root, self.df.iloc[idx]['path'])
        
        # 🚀 [핵심 수정] 손상된 이미지 예외 처리 로직 추가
        try:
            image = Image.open(img_full_path).convert("RGB")
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)
        return image

# 2. 학습 환경 설정
def train():
    # 하이퍼파라미터
    batch_size = 32
    lr = 1e-3
    epochs = 50
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # ---------------------------------------------------------
    # 1. Transform 분리 (Train용과 Val용을 따로 만듭니다)
    # ---------------------------------------------------------
    train_transform = transforms.Compose([
        ChestXrayPreprocess(target_size=(224,224)),
        transforms.RandomRotation(5), # 학습할 때만 살짝 흔들어줍니다.
        transforms.ToTensor() 
    ])

    val_transform = transforms.Compose([
        ChestXrayPreprocess(target_size=(224,224)),
        transforms.ToTensor() # 🚀 검증할 때는 정자세(원본 그대로)를 유지해야 합니다!
    ])

    # 경로 설정 (기태님 환경에 맞게 수정)
    IMG_ROOT = "data"
    TRAIN_CSV = "data/mimic_cxr_aug_train.csv"
    VAL_CSV = "data/mimic_cxr_aug_validate.csv"
    CHEXPERT_CSV = "data/mimic-cxr-2.0.0-chexpert.csv"

    # 1. DataFrame 파싱 (Train은 정상만, Val은 정상/비정상 모두 활용)
    # 💡 IMG_ROOT 매개변수를 넣어줍니다.
    train_normal_df, _ = get_mimic_ae_dfs(TRAIN_CSV, CHEXPERT_CSV, IMG_ROOT)
    val_normal_df, val_abnormal_df = get_mimic_ae_dfs(VAL_CSV, CHEXPERT_CSV, IMG_ROOT)

    # ---------------------------------------------------------
    # 2. Dataset에 알맞은 Transform 적용
    # ---------------------------------------------------------
    train_dataset = MimicAutoencoderDataset(train_normal_df, IMG_ROOT, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 🚀 Val 데이터셋에는 회전이 없는 val_transform을 넣습니다.
    val_normal_ds = MimicAutoencoderDataset(val_normal_df, IMG_ROOT, transform=val_transform)
    val_normal_loader = DataLoader(val_normal_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    val_abnormal_ds = MimicAutoencoderDataset(val_abnormal_df, IMG_ROOT, transform=val_transform)
    val_abnormal_loader = DataLoader(val_abnormal_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # 모델 초기화
    from normal_link_model import DeepNormalLinkAE
    model = DeepNormalLinkAE().to(device)
    
    # 손실 함수 및 최적화 도구
    criterion = HybridLoss(alpha=0.8) 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # 스케줄러는 그대로 유지하되, 이후 step()에 검증(Val) Loss를 넣을 예정입니다.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    print(f"🚀 학습 시작! 기기: {device} | Train 데이터 수: {len(train_dataset)}")

    # 3. 학습 및 검증 루프
    for epoch in range(epochs):
        # === [Train 단계] ===
        model.train()
        train_loss = 0.0
        
        # dataloader -> train_dataloader 로 변경
        for images in train_dataloader:
            images = images.to(device)
            
            # 🚀 [시니어의 비법] 가우시안 노이즈 주입 (Denoising)
            # 이미지에 10% 정도의 노이즈를 섞습니다.
            noise_factor = 0.1 
            noisy_images = images + noise_factor * torch.randn_like(images)
            noisy_images = torch.clamp(noisy_images, 0., 1.) # 픽셀 범위를 0~1로 유지
            
            # Forward: 입력은 '노이즈 낀 이미지', 정답은 '원본 깨끗한 이미지'
            outputs = model(noisy_images)
            loss = criterion(outputs, images) 
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        avg_train_loss = train_loss / len(train_dataset)

        # === [Validation 단계 추가] ===
        model.eval()
        val_loss_normal = 0.0
        val_loss_abnormal = 0.0

        with torch.no_grad():
            # 1. 정상 데이터 검증 (잘 복원해야 하므로 오차가 낮아야 함)
            for images in val_normal_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss_normal += loss.item() * images.size(0)

            # 2. 비정상(질환) 데이터 검증 (제대로 복원하지 못해 오차가 높아야 함)
            for images in val_abnormal_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss_abnormal += loss.item() * images.size(0)

        # 평균 Validation Loss 계산
        avg_val_normal = val_loss_normal / len(val_normal_ds) if len(val_normal_ds) > 0 else 0
        avg_val_abnormal = val_loss_abnormal / len(val_abnormal_ds) if len(val_abnormal_ds) > 0 else 0

        # 스케줄러 업데이트: 모델의 목표는 '정상 데이터'를 잘 복원하는 것이므로 Val Normal Loss를 기준으로 삼습니다.
        scheduler.step(avg_val_normal)
        
        # 학습 현황 출력 (오차 갭을 눈으로 확인할 수 있습니다)
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.6f}")
        print(f" └─ Val Normal Loss: {avg_val_normal:.6f} | Val Abnormal Loss: {avg_val_abnormal:.6f}")

        # 10에폭마다 모델 저장
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"normal_link_v2_ep{epoch+1}.pth")

    print("✅ 모든 학습이 완료되었습니다!")

if __name__ == "__main__":
    train()