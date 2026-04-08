import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
import pandas as pd
import os
import ast
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score

# 🚀 모듈화된 파일들 임포트
from mimic_dataset import prepare_mimic_df, MimicDataset
from soo_net import SooNetEngine
from unet_lung_model import UNet # 파트너님의 U-Net 모델 임포트

LABEL_ORDER = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "No Finding", "Pleural Effusion", 
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

def prepare_chexpert_df(csv_path, img_root):
    df = pd.read_csv(csv_path)
    df[LABEL_ORDER] = df[LABEL_ORDER].fillna(0).replace(-1, 0)
    
    if 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal']
    
    flat_data = []
    print(f"🔍 CheXpert 데이터 동적 경로 파싱 중... (FastFile 환경에 최적화됨)")
    missing_count = 0
    
    # 💡 파트너님의 실제 S3 배치 폴더 이름들
    candidate_folders = [
        "CheXpert-v1.0 batch 1 (validate & csv)",
        "CheXpert-v1.0 batch 2 (train 1)",
        "CheXpert-v1.0 batch 3 (train 2)",
        "CheXpert-v1.0 batch 4 (train 3)"
    ]

    for _, row in df.iterrows():
        raw_path = str(row['Path']) # 예: 'CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg'
        
        # 1. 'patient'라는 단어가 시작되는 위치를 찾아 핵심 경로만 잘라냅니다.
        path_parts = raw_path.split('/')
        patient_idx = -1
        for i, part in enumerate(path_parts):
            if part.startswith('patient'):
                patient_idx = i
                break
                
        if patient_idx == -1:
            missing_count += 1
            continue
            
        # core_path = 'patient00001/study1/view1_frontal.jpg'
        core_path = "/".join(path_parts[patient_idx:]) 
        
        # 2. 4개의 배치 폴더 중 이 파일이 진짜로 존재하는 곳을 찾습니다.
        found_real_path = None
        for folder in candidate_folders:
            candidate_full_path = os.path.join(img_root, folder, core_path)
            
            # FastFile 모드에서도 os.path.exists가 S3를 찔러서 파일 유무를 확인해줍니다!
            if os.path.exists(candidate_full_path):
                found_real_path = candidate_full_path
                break
                
        if found_real_path:
            flat_data.append({
                'path': found_real_path,
                'labels': row[LABEL_ORDER].values
            })
        else:
            missing_count += 1
            # 너무 많이 출력되면 로그가 지저분해지므로 생략 (디버깅 시 주석 해제)
            # print(f"⚠️ 파일 없음: {core_path}")

    final_df = pd.DataFrame(flat_data)
    print(f"✅ CheXpert 동적 파싱 완료: 총 {len(final_df)}장 준비 완료! (누락/매칭실패 {missing_count}장)")
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
            # 1채널 흑백(Grayscale) 모드로 이미지를 엽니다
            image = Image.open(img_full_path).convert('L')
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
            
        label = torch.FloatTensor(row['labels'].astype(float))
        
        if self.transform:
            image = self.transform(image)
        return image, label
# =====================================================================
# 🚀 1. 전처리 클래스 (U-Net을 위해 3채널 RGB 텐서 반환)
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

class Base_Transform:
    def __init__(self, target_size=(448, 448), is_train=False):
        self.target_size = target_size
        self.is_train = is_train
        self.clahe_preprocess = ChestXrayPreprocess(target_size=target_size, clip_limit=2.0)

    def __call__(self, img):
        img = self.clahe_preprocess(img) # RGB 이미지 반환
        if self.is_train:
            img = transforms.RandomRotation(5)(img)
        # U-Net 입력용으로 3채널, 0.0~1.0 스케일의 텐서로 변환
        return transforms.ToTensor()(img)

# =====================================================================
# 🚀 2. U-Net 크롭 & 비율 유지 리사이즈 & TXV 스케일링 함수
# =====================================================================
def process_unet_crops(images, masks, padding=10, target_size=(448, 448)):
    batch_size = images.size(0)
    processed_batch = []

    for i in range(batch_size):
        img = images[i]
        mask = masks[i].squeeze() > 0.5
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)

        # 마스크가 비어있을 경우 (예외 처리)
        if not torch.any(rows) or not torch.any(cols):
            gray_img = img.mean(dim=0, keepdim=True)
            processed_batch.append(gray_img)
            continue

        y_indices = torch.where(rows)[0]
        x_indices = torch.where(cols)[0]

        y_min, y_max = y_indices[0].item(), y_indices[-1].item()
        x_min, x_max = x_indices[0].item(), x_indices[-1].item()

        H, W = mask.shape
        y_min, y_max = max(0, y_min - padding), min(H, y_max + padding)
        x_min, x_max = max(0, x_min - padding), min(W, x_max + padding)

        # 1. Bounding Box 크롭
        cropped_img = img[:, y_min:y_max, x_min:x_max]

        # 💡 2. 비율 보존(Aspect Ratio) 정방형 패딩 로직
        c, h, w = cropped_img.shape
        diff = abs(h - w)
        if h > w:
            pad_left = diff // 2
            pad_right = diff - pad_left
            cropped_img = TF.pad(cropped_img, (pad_left, pad_right, 0, 0), fill=0)
        elif w > h:
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            cropped_img = TF.pad(cropped_img, (0, 0, pad_top, pad_bottom), fill=0)

        # 3. 448x448 로 안전하게 리사이즈 (찌그러짐 없음)
        resized_img = TF.resize(cropped_img, target_size, antialias=True)

        # 4. DenseNet을 위해 1채널(Grayscale)로 변경
        gray_img = resized_img.mean(dim=0, keepdim=True)
        processed_batch.append(gray_img)

    # 5. 스택 후 TXV 공식 스케일링 (-1024 ~ 1024) 적용
    final_batch = torch.stack(processed_batch)
    txv_scaled_batch = (final_batch * 2048.0) - 1024.0
    
    return txv_scaled_batch

# =====================================================================
# 🚀 3. 메인 학습 루프
# =====================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"🚀 클라우드 학습 시작! Device: {device}")
    
    # --- 데이터 로더 세팅 ---
    if args.use_mimic:
        IMG_ROOT = args.data_dir 
        TRAIN_CSV = os.path.join(args.data_dir, "mimic_cxr_aug_train.csv")
        VAL_CSV = os.path.join(args.data_dir, "mimic_cxr_aug_validate.csv")
        CHEXPERT_CSV = os.path.join(args.data_dir, "mimic-cxr-2.0.0-chexpert.csv")
        SAVE_MODEL_NAME = os.path.join(args.model_dir, "chexnet_1ch_448_mimic_best.pth")
        
        train_df = prepare_mimic_df(TRAIN_CSV, CHEXPERT_CSV, IMG_ROOT)
        val_df = prepare_mimic_df(VAL_CSV, CHEXPERT_CSV, IMG_ROOT)
        TrainDatasetClass = MimicDataset
    else:
        IMG_ROOT = args.data_dir 
        TRAIN_CSV = os.path.join(args.data_dir, "train.csv")
        VAL_CSV = os.path.join(args.data_dir, "valid.csv")
        CHEXPERT_CSV = None  
        SAVE_MODEL_NAME = os.path.join(args.model_dir, "chexnet_1ch_448_chexpert_best.pth")
        
        train_df = prepare_chexpert_df(TRAIN_CSV, IMG_ROOT)
        val_df = prepare_chexpert_df(VAL_CSV, IMG_ROOT)
        TrainDatasetClass = ChexpertDataset

    train_transform = Base_Transform(target_size=(448, 448), is_train=True)
    val_transform = Base_Transform(target_size=(448, 448), is_train=False)
    
    train_loader = DataLoader(TrainDatasetClass(train_df, train_transform), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(TrainDatasetClass(val_df, val_transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- 💡 U-Net 초기화 및 가중치 로드 ---
    print("✂️ U-Net 모델 로드 중...")
    unet = UNet(n_channels=3, n_classes=1).to(device)
    # SageMaker source_dir에 올려둔 U-Net 가중치를 로드합니다.
    unet.load_state_dict(torch.load(args.unet_weight_path, map_location=device))
    unet.eval() # U-Net은 마스크만 따면 되므로 학습(Freeze)하지 않습니다.

    # --- DenseNet 초기화 ---
    print("🧠 SooNetEngine 로드 중...")
    engine = SooNetEngine(model_path=None)
    model = engine.model
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=args.lr_patience)

    best_auroc = 0.0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for i, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            # 1. U-Net으로 마스크 예측 (그라디언트 계산 안 함)
            with torch.no_grad():
                masks = unet(imgs)
            
            # 2. 크롭 -> 정방형 패딩 -> 448 리사이즈 -> 1채널 흑백 -> TXV 스케일링
            txv_imgs = process_unet_crops(imgs, masks, target_size=(448, 448))
            
            # 3. DenseNet 추론
            outputs = model(txv_imgs)
            loss = criterion(outputs, lbls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Batch [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # === 검증(Validation) ===
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                
                # Validation 시에도 동일하게 U-Net 파이프라인 적용!
                masks = unet(imgs)
                txv_imgs = process_unet_crops(imgs, masks, target_size=(448, 448))
                
                outputs = torch.sigmoid(model(txv_imgs)) 
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(lbls.numpy())
        
        val_preds = np.vstack(all_preds)
        val_labels = np.vstack(all_labels)
        
        print(f"\n📊 --- Epoch {epoch+1} Validation Report ---")
        print(f"{'Disease':<27s} | {'AUROC':<6s} | {'Best_Th':<7s} | {'Prec':<6s} | {'Recall':<6s} | {'F1':<6s}")
        print("-" * 75)

        val_auroc_list, val_f1_list = [], []

        for c, class_name in enumerate(LABEL_ORDER): 
            if len(np.unique(val_labels[:, c])) > 1:
                auroc = roc_auc_score(val_labels[:, c], val_preds[:, c])
                val_auroc_list.append(auroc)
                
                precisions, recalls, thresholds = precision_recall_curve(val_labels[:, c], val_preds[:, c])
                precisions, recalls = precisions[:-1], recalls[:-1]
                
                numerator = 2 * recalls * precisions
                denominator = recalls + precisions
                f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator!=0))
                
                best_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_idx]
                
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
            
        print(f"✅ Epoch [{epoch+1}/{args.epochs}] Avg Loss: {epoch_loss/len(train_loader):.4f} | Macro AUROC: {avg_auroc:.4f} | Macro F1: {avg_f1:.4f}\n")
        
        scheduler.step(avg_auroc)

        if avg_auroc > best_auroc:
            best_auroc = avg_auroc
            torch.save(model.state_dict(), SAVE_MODEL_NAME)
            print(f"💾 Best Model Saved to Cloud: {SAVE_MODEL_NAME} (AUROC: {avg_auroc:.4f})")

# =====================================================================
# 🚀 4. SageMaker 환경 변수
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data'))
    
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--lr-patience', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--use-mimic', type=lambda x: (str(x).lower() == 'true'), default=True)
    
    # 💡 [추가] U-Net 가중치 파일 이름 (SageMaker source_dir 폴더 안에 같이 올려주세요!)
    parser.add_argument('--unet-weight-path', type=str, default='unet_lung_mask_ep10.pth')

    args = parser.parse_args()
    train(args)