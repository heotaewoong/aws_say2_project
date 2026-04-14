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
    
    # 최고 성능 모델 저장 경로
    SAVE_MODEL_NAME = os.path.join(args.model_dir, "chexnet_unet_crop_best.pth")
    
    # 💡 [여기!] 타임캡슐(체크포인트) 백업 파일 경로
    CHECKPOINT_PATH = os.path.join(args.model_dir, "last_checkpoint.pth")

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

    start_epoch = 1
    best_auroc = 0.0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n🔄 중단된 학습 기록을 발견했습니다! 복구를 시도합니다...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_auroc = checkpoint['best_auroc']

            # 💡 [수정됨] 완벽한 이어달리기 에포크 계산
            saved_epoch = checkpoint['epoch']
            is_end = checkpoint.get('is_end_of_epoch', False)
            
            # 끝까지 다 돌고 저장된 거면 다음 에포크부터, 절반에서 튕긴 거면 현재 에포크 처음부터!
            start_epoch = saved_epoch + 1 if is_end else saved_epoch
            start_epoch = max(1, start_epoch) # 혹시 모를 0 에포크 방지

            print(f"✅ 복구 완료! Epoch {start_epoch}부터 이어 달립니다! (기존 Best: {best_auroc:.4f})\n")
        except Exception as e:
            print(f"⚠️ 체크포인트 파일이 손상되었습니다. 처음부터 시작합니다: {e}\n")

    # =====================================================================
    # 🚀 메인 학습 & Sub-epoch 검증 루프
    # =====================================================================
    total_batches = len(train_loader)
    half_batch = total_batches // 2  # 💡 정확히 1/2 지점 계산

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0
        
        for i, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            with torch.no_grad():
                masks = unet(imgs)
            
            txv_imgs = process_unet_crops(imgs, masks, target_size=(448, 448))
            
            outputs = model(txv_imgs)
            loss = criterion(outputs, lbls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if (i+1) % 50 == 0:
                print(f"Batch [{i+1}/{total_batches}] Loss: {loss.item():.4f}")

            # =================================================================
            # 💡 [핵심] 1/2 지점 또는 에포크 끝 지점 도달 시 검증 및 저장 발동!
            # =================================================================
            is_halfway = (i + 1) == half_batch
            is_end_of_epoch = (i + 1) == total_batches

            if is_halfway or is_end_of_epoch:
                # 상태 표시용 이름 (예: Epoch 1.5 또는 Epoch 1.0)
                step_name = f"{epoch}.5 (Halfway)" if is_halfway else f"{epoch}.0 (End)"
                print(f"\n⏳ [Epoch {step_name}] 검증 및 체크포인트 저장을 시작합니다...")

                model.eval()
                all_preds, all_labels = [], []
                
                with torch.no_grad():
                    # ⚠️ 주의: 학습 배치 변수(imgs, lbls)와 겹치지 않게 val_ 접두사 사용
                    for val_imgs, val_lbls in val_loader:
                        val_imgs = val_imgs.to(device)
                        val_masks = unet(val_imgs)
                        val_txv = process_unet_crops(val_imgs, val_masks, target_size=(448, 448))
                        
                        val_outputs = torch.sigmoid(model(val_txv)) 
                        all_preds.append(val_outputs.cpu().numpy())
                        all_labels.append(val_lbls.numpy())
                
                val_preds = np.vstack(all_preds)
                val_labels = np.vstack(all_labels)
                
                print(f"\n📊 --- Epoch {step_name} Validation Report ---")
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
                    
                print(f"✅ [Epoch {step_name}] Macro AUROC: {avg_auroc:.4f} | Macro F1: {avg_f1:.4f}\n")
                
                scheduler.step(avg_auroc)

                # 💡 타임캡슐 저장 (절반 지점이든 끝이든 무조건 백업)
                checkpoint = {
                    'epoch': epoch, 
                    'is_end_of_epoch': is_end_of_epoch, # 💡 끝났는지 여부도 함께 메모표로 남깁니다.
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_auroc': best_auroc
                }
                torch.save(checkpoint, CHECKPOINT_PATH)
                print(f"💾 진행 상황 백업 완료 -> {CHECKPOINT_PATH}")

                # 🏆 신기록 경신 시 Best Model 별도 저장
                if avg_auroc > best_auroc:
                    best_auroc = avg_auroc
                    torch.save(model.state_dict(), SAVE_MODEL_NAME)
                    print(f"🏆 신기록 경신! Best Model Saved: {SAVE_MODEL_NAME} (AUROC: {avg_auroc:.4f})")
                
                # 💡 [아주 중요!] 검증이 끝나면 멈춰 있던 학습 모드를 다시 켭니다.
                model.train() 

    # 전체 학습 종료 후 딱 한 번 그래프 출력
    print("\n🎉 모든 학습이 완료되었습니다! 최종 시각화 그래프를 생성합니다...")
    plot_and_save_curves(val_labels, val_preds, "Final", args.plot_dir)
    print(f"✅ 최종 시각화 지표가 '{args.plot_dir}' 폴더에 저장되었습니다!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=r'E:\Programming\aws_say2_project\model')
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