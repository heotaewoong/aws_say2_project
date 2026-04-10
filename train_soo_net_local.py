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
import re
import matplotlib
matplotlib.use('Agg') # GUI 없는 환경(서버/터미널)에서 에러 방지
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score, auc, roc_curve

# 🚀 모듈화된 파일들 임포트
from soo_net import SooNetEngine
from unet_lung_model import UNet

LABEL_ORDER = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "No Finding", "Pleural Effusion", 
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

# =====================================================================
# 🚀 1. Dataset 및 파싱 로직 (.jpg 원본 로드용으로 복구)
# =====================================================================
def prepare_mimic_df(aug_csv_path, chexpert_csv_path, img_root):
    print(f"🔍 MIMIC 데이터 병합 및 원본(.jpg) 경로 파싱 시작...")
    
    aug_df = pd.read_csv(aug_csv_path)
    label_df = pd.read_csv(chexpert_csv_path)
    label_df[LABEL_ORDER] = label_df[LABEL_ORDER].fillna(0).replace(-1, 1)
    
    flat_data = []
    missing_count = 0
    debug_printed = False
    
    for _, row in aug_df.iterrows():
        for view in ['AP', 'PA']:
            if view not in row: continue
            
            raw_paths = row[view]
            if pd.isna(raw_paths): continue
                
            try:
                path_list = ast.literal_eval(raw_paths)
                for rel_path in path_list:
                    # 💡 .jpg 원본을 그대로 유지합니다.
                    clean_rel_path = rel_path.replace('files/', '').lstrip('\\/')
                    
                    candidate_1 = os.path.normpath(os.path.join(img_root, rel_path))
                    candidate_2 = os.path.normpath(os.path.join(img_root, clean_rel_path))
                    
                    full_path = None
                    if os.path.exists(candidate_1):
                        full_path = candidate_1
                    elif os.path.exists(candidate_2):
                        full_path = candidate_2
                        
                    if full_path:
                        match = re.search(r's(\d{8})', rel_path)
                        if match:
                            study_id = int(match.group(1))
                            label_row = label_df[label_df['study_id'] == study_id]
                            
                            if not label_row.empty:
                                flat_data.append({
                                    'path': full_path,
                                    'labels': label_row[LABEL_ORDER].values[0].astype(float)
                                })
                            else:
                                missing_count += 1
                        else:
                            missing_count += 1
                    else:
                        if not debug_printed:
                            print(f"\n🚨 [경로 추적 탐지기 작동]")
                            print(f"❌ 원본 CSV 경로: {rel_path}")
                            print(f"❌ 탐색 시도 1: {candidate_1}")
                            print(f"❌ 탐색 시도 2: {candidate_2}")
                            print(f"👉 U-Net을 돌리려면 반드시 .jpg 원본 경로를 지정해야 합니다!\n")
                            debug_printed = True
                        missing_count += 1
            except (ValueError, SyntaxError, IndexError):
                continue

    final_df = pd.DataFrame(flat_data)
    print(f"✅ MIMIC 병합 완료: 총 {len(final_df)}장 준비되었습니다. (누락/매칭실패 {missing_count}장)")
    return final_df

class MimicDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        
        try:
            image = Image.open(img_path).convert('L')
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
            
        label = torch.FloatTensor(row['labels'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# =====================================================================
# 🚀 2. CLAHE 전처리 및 U-Net 크롭 파이프라인
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
        img = self.clahe_preprocess(img) # RGB 반환 (U-Net 입력용)
        if self.is_train:
            img = transforms.RandomRotation(5)(img)
        return transforms.ToTensor()(img)

def process_unet_crops(images, masks, padding=10, target_size=(448, 448)):
    batch_size = images.size(0)
    processed_batch = []

    for i in range(batch_size):
        img = images[i]
        mask = masks[i].squeeze() > 0.5
        rows, cols = torch.any(mask, dim=1), torch.any(mask, dim=0)

        if not torch.any(rows) or not torch.any(cols):
            gray_img = img.mean(dim=0, keepdim=True)
            processed_batch.append(gray_img)
            continue

        y_indices, x_indices = torch.where(rows)[0], torch.where(cols)[0]
        y_min, y_max = y_indices[0].item(), y_indices[-1].item()
        x_min, x_max = x_indices[0].item(), x_indices[-1].item()

        H, W = mask.shape
        y_min, y_max = max(0, y_min - padding), min(H, y_max + padding)
        x_min, x_max = max(0, x_min - padding), min(W, x_max + padding)

        cropped_img = img[:, y_min:y_max, x_min:x_max]
        
        # 비율 보존 패딩
        c, h, w = cropped_img.shape
        diff = abs(h - w)
        if h > w:
            cropped_img = TF.pad(cropped_img, (diff // 2, diff - diff // 2, 0, 0), fill=0)
        elif w > h:
            cropped_img = TF.pad(cropped_img, (0, 0, diff // 2, diff - diff // 2), fill=0)

        resized_img = TF.resize(cropped_img, target_size, antialias=True)
        gray_img = resized_img.mean(dim=0, keepdim=True)
        processed_batch.append(gray_img)

    final_batch = torch.stack(processed_batch)
    txv_scaled_batch = (final_batch * 2048.0) - 1024.0 # TXV 스케일링
    
    return txv_scaled_batch

# =====================================================================
# 🚀 3. 시각화 지표 생성 함수 (ROC & PR Curve)
# =====================================================================
def plot_and_save_curves(val_labels, val_preds, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. ROC Curve ---
    plt.figure(figsize=(12, 10))
    for c, class_name in enumerate(LABEL_ORDER): 
        if len(np.unique(val_labels[:, c])) > 1:
            fpr, tpr, _ = roc_curve(val_labels[:, c], val_preds[:, c])
            plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {auc(fpr, tpr):.3f})")
            
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - Epoch {epoch}', fontsize=16)
    plt.legend(loc="lower right", fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'roc_curve_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 2. PR Curve ---
    plt.figure(figsize=(12, 10))
    for c, class_name in enumerate(LABEL_ORDER): 
        if len(np.unique(val_labels[:, c])) > 1:
            prec, rec, _ = precision_recall_curve(val_labels[:, c], val_preds[:, c])
            plt.plot(rec, prec, lw=2, label=f"{class_name}")
            
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - Epoch {epoch}', fontsize=16)
    plt.legend(loc="lower left", fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'pr_curve_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# =====================================================================
# 🚀 4. 메인 학습 루프
# =====================================================================
def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🚀 U-Net Crop 통합 학습 시작! Device: {device}")
    
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # --- 데이터 로더 세팅 ---
    IMG_ROOT = args.img_dir 
    TRAIN_CSV = os.path.join(args.csv_dir, "mimic_cxr_aug_train.csv")
    VAL_CSV = os.path.join(args.csv_dir, "mimic_cxr_aug_validate.csv")
    CHEXPERT_CSV = os.path.join(args.csv_dir, "mimic-cxr-2.0.0-chexpert.csv")
    SAVE_MODEL_NAME = os.path.join(args.model_dir, "chexnet_unet_crop_best.pth")
    
    train_df = prepare_mimic_df(TRAIN_CSV, CHEXPERT_CSV, IMG_ROOT)
    val_df = prepare_mimic_df(VAL_CSV, CHEXPERT_CSV, IMG_ROOT)
    
    train_transform = Base_Transform(target_size=(448, 448), is_train=True)
    val_transform = Base_Transform(target_size=(448, 448), is_train=False)
    
    train_loader = DataLoader(MimicDataset(train_df, transform=train_transform), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(MimicDataset(val_df, transform=val_transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- 💡 U-Net 로드 ---
    print("✂️ U-Net 모델 로드 중...")
    unet = UNet(n_channels=3, n_classes=1).to(device)
    try:
        unet.load_state_dict(torch.load(args.unet_weight_path, map_location=device))
        print("✅ U-Net 가중치 로드 성공")
    except Exception as e:
        print(f"⚠️ U-Net 로드 실패: {e}")
    unet.eval() 

    # --- 분류 모델 로드 ---
    print("🧠 SooNetEngine 로드 중...")
    engine = SooNetEngine(model_path=None)
    model = engine.model
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=args.lr_patience)

    best_auroc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        
        for i, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            # 💡 1. U-Net으로 마스크 추출
            with torch.no_grad():
                masks = unet(imgs)
            
            # 💡 2. 마스크 기반 크롭 및 TXV 변환
            txv_imgs = process_unet_crops(imgs, masks, target_size=(448, 448))
            
            # 💡 3. DenseNet 추론
            outputs = model(txv_imgs)
            loss = criterion(outputs, lbls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if (i+1) % 50 == 0:
                print(f"Batch [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # === 검증(Validation) ===
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                
                # 검증 시에도 U-Net 크롭 수행
                masks = unet(imgs)
                txv_imgs = process_unet_crops(imgs, masks, target_size=(448, 448))
                
                outputs = torch.sigmoid(model(txv_imgs)) 
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(lbls.numpy())
        
        val_preds = np.vstack(all_preds)
        val_labels = np.vstack(all_labels)
        
        print(f"\n📊 --- Epoch {epoch} Validation Report ---")
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
            
        print(f"✅ Epoch [{epoch}/{args.epochs}] Avg Loss: {epoch_loss/len(train_loader):.4f} | Macro AUROC: {avg_auroc:.4f} | Macro F1: {avg_f1:.4f}\n")
        
        plot_and_save_curves(val_labels, val_preds, epoch, args.plot_dir)
        
        scheduler.step(avg_auroc)

        if avg_auroc > best_auroc:
            best_auroc = avg_auroc
            torch.save(model.state_dict(), SAVE_MODEL_NAME)
            print(f"💾 Best Model Saved: {SAVE_MODEL_NAME} (AUROC: {avg_auroc:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='./model')
    parser.add_argument('--plot-dir', type=str, default='./plots')
    
    # 🚨 주의: U-Net을 돌리기 위해 반드시 .pt가 아닌 .jpg가 위치한 원본 폴더 경로를 입력하세요!
    parser.add_argument('--img-dir', type=str, default=r'E:\Programming\data\mimic-cxr\files') 
    parser.add_argument('--csv-dir', type=str, default=r'E:\Programming\data')
    parser.add_argument('--unet-weight-path', type=str, default='unet_lung_mask_ep10.pth')

    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=16) # U-Net 연산이 추가되어 VRAM을 더 먹으므로 배치를 8로 낮춤
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--lr-patience', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=2)

    args = parser.parse_args()
    train(args)