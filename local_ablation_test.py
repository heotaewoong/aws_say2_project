import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
import pandas as pd
import os
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score

from mimic_dataset import prepare_mimic_df, MimicDataset
from unet_lung_model import UNet

LABEL_ORDER = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "No Finding", "Pleural Effusion", 
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

# =====================================================================
# 1. 실험 전용 전처리 클래스 및 함수
# =====================================================================
class AblationTransform:
    def __init__(self, target_size=(224, 224), use_clahe=False):
        self.target_size = target_size
        self.use_clahe = use_clahe

    def __call__(self, img):
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_np = np.array(img.convert('L'))
            img_clahe = clahe.apply(img_np)
            img = Image.fromarray(img_clahe).convert('RGB')
        else:
            img = img.convert('RGB')

        img = img.resize(self.target_size, Image.BILINEAR)

        return transforms.ToTensor()(img) # 0.0 ~ 1.0 텐서로 변환

def apply_normalization(tensor_batch, norm_type, in_channels):
    """선택된 정규화(Normalization) 기법을 일괄 적용합니다."""
    if norm_type == 'txv':
        # TXV 공식 스케일링 (-1024.0 ~ +1024.0)
        return (tensor_batch * 2048.0) - 1024.0
    elif norm_type == 'imagenet':
        # ImageNet 표준 정규화
        if in_channels == 3:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            # 1채널 흑백일 경우 ImageNet 평균/표준편차의 평균값 사용
            normalize = transforms.Normalize(mean=[0.449], std=[0.226])
        
        # 배치(Batch) 내 모든 이미지에 정규화 적용
        return torch.stack([normalize(img) for img in tensor_batch])
    else:
        # 정규화 미적용 (0.0 ~ 1.0 유지)
        return tensor_batch

def process_ablation_inputs(images, masks, target_size, in_channels, use_preproc, norm_type):
    """U-Net 크롭 여부 및 채널, 정규화를 한 번에 처리하는 마스터 함수"""
    batch_size = images.size(0)
    processed_batch = []

    for i in range(batch_size):
        img = images[i]
        
        if use_preproc and masks is not None:
            mask = masks[i].squeeze() > 0.5
            rows, cols = torch.any(mask, dim=1), torch.any(mask, dim=0)

            if not torch.any(rows) or not torch.any(cols):
                resized = TF.resize(img, target_size, antialias=True)
            else:
                y_indices, x_indices = torch.where(rows)[0], torch.where(cols)[0]
                y_min, y_max = max(0, y_indices[0].item() - 10), min(mask.shape[0], y_indices[-1].item() + 10)
                x_min, x_max = max(0, x_indices[0].item() - 10), min(mask.shape[1], x_indices[-1].item() + 10)

                cropped = img[:, y_min:y_max, x_min:x_max]
                c, h, w = cropped.shape
                diff = abs(h - w)
                if h > w:
                    cropped = TF.pad(cropped, (diff // 2, diff - diff // 2, 0, 0), fill=0)
                elif w > h:
                    cropped = TF.pad(cropped, (0, 0, diff // 2, diff - diff // 2), fill=0)
                    
                resized = TF.resize(cropped, target_size, antialias=True)
        else:
            resized = TF.resize(img, target_size, antialias=True)

        # 1채널/3채널 스위치
        if in_channels == 1:
            resized = resized.mean(dim=0, keepdim=True)
            
        processed_batch.append(resized)

    tensor_batch = torch.stack(processed_batch)
    
    # 💡 마지막으로 선택된 정규화(Normalization) 적용
    return apply_normalization(tensor_batch, norm_type, in_channels)

# =====================================================================
# 2. 초고속 미니 검증 루프
# =====================================================================
def run_ablation(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*70)
    print(f"🚀 [Ablation] Size: {args.img_size} | 채널: {args.in_channels} | 전처리: {args.use_preproc} | Norm: {args.norm_type.upper()}")
    print("="*70)

    # 1. 데이터 축소 로드 (로컬 테스트용)
    IMG_ROOT = "/Users/skku_aws2_15/med/data" 
    TRAIN_CSV = "/Users/skku_aws2_15/med/data/mimic_cxr_aug_train.csv"
    VAL_CSV = "/Users/skku_aws2_15/med/data/mimic_cxr_aug_validate.csv"
    CHEXPERT_CSV = "/Users/skku_aws2_15/med/data/mimic-cxr-2.0.0-chexpert.csv"

    print("📊 데이터 로드 중... (Train 30,000장 샘플링)")
    
    # 🚀 [수정 1] 앞에서부터 자르지 않고(head), 전체 데이터 중 랜덤하게 3만 장을 뽑습니다.
    # random_state=42를 주어 4번의 실험 모두 '완벽하게 똑같은 3만 장'으로 공정하게 시험을 치르도록 고정합니다.
    train_df = prepare_mimic_df(TRAIN_CSV, CHEXPERT_CSV, IMG_ROOT).sample(n=30000, random_state=42)
    
    # Validation은 원래 개수(약 900장)가 평가하기 딱 좋으므로 전부 다 씁니다.
    val_df = prepare_mimic_df(VAL_CSV, CHEXPERT_CSV, IMG_ROOT) 
    
    transform = AblationTransform(target_size=(args.img_size, args.img_size), use_clahe=args.use_preproc)
    
    # 🚀 [수정 2] batch_size=16 뒤에 빠져있던 쉼표(,)를 추가했습니다! (SyntaxError 방지)
    train_loader = DataLoader(MimicDataset(train_df, transform), batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(MimicDataset(val_df, transform), batch_size=16, shuffle=False, num_workers=0)

    # 2. 실험용 모델 조립
    model = models.densenet121(weights='IMAGENET1K_V1')
    if args.in_channels == 1:
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(model.classifier.in_features, 14)
    model = model.to(device)

    unet = None
    if args.use_preproc:
        print("✂️ U-Net 로드 중...")
        unet = UNet(n_channels=3, n_classes=1).to(device)
        try:
            unet.load_state_dict(torch.load('unet_lung_mask_ep10.pth', map_location=device))
        except:
            print("⚠️ U-Net 가중치를 찾을 수 없어 초기화된 모델로 진행합니다.")
        unet.eval()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 3. 미니 학습 루프 (3 Epochs)
    for epoch in range(5):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            masks = None
            if args.use_preproc and unet is not None:
                with torch.no_grad():
                    masks = unet(imgs)
                    
            inputs = process_ablation_inputs(imgs, masks, (args.img_size, args.img_size), args.in_channels, args.use_preproc, args.norm_type)
            
            outputs = model(inputs)
            loss = criterion(outputs, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # === 💡 4. Validation 및 4대 지표 종합 평가 ===
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                masks = unet(imgs) if (args.use_preproc and unet is not None) else None
                inputs = process_ablation_inputs(imgs, masks, (args.img_size, args.img_size), args.in_channels, args.use_preproc, args.norm_type)
                        
                outputs = torch.sigmoid(model(inputs))
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(lbls.numpy())
                
        val_preds = np.vstack(all_preds)
        val_labels = np.vstack(all_labels)
        
        auroc_list, prec_list, rec_list, f1_list = [], [], [], []
        
        for c in range(14):
            if len(np.unique(val_labels[:, c])) > 1:
                # AUROC
                auroc = roc_auc_score(val_labels[:, c], val_preds[:, c])
                auroc_list.append(auroc)
                
                # 최적 임계값 탐색
                precisions, recalls, thresholds = precision_recall_curve(val_labels[:, c], val_preds[:, c])
                precisions, recalls = precisions[:-1], recalls[:-1]
                
                numerator = 2 * recalls * precisions
                denominator = recalls + precisions
                f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator!=0))
                best_threshold = thresholds[np.argmax(f1_scores)] if len(f1_scores) > 0 else 0.5
                
                # P, R, F1 계산
                opt_preds = (val_preds[:, c] >= best_threshold).astype(int)
                prec_list.append(precision_score(val_labels[:, c], opt_preds, zero_division=0))
                rec_list.append(recall_score(val_labels[:, c], opt_preds, zero_division=0))
                f1_list.append(f1_score(val_labels[:, c], opt_preds, zero_division=0))

        m_auroc = np.mean(auroc_list) if auroc_list else 0.0
        m_prec = np.mean(prec_list) if prec_list else 0.0
        m_rec = np.mean(rec_list) if rec_list else 0.0
        m_f1 = np.mean(f1_list) if f1_list else 0.0
        
        print(f"✅ Epoch [{epoch+1}/5] ➡️ AUROC: {m_auroc:.4f} | Prec: {m_prec:.4f} | Rec: {m_rec:.4f} | F1: {m_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=224) 
    parser.add_argument('--in-channels', type=int, default=3)
    parser.add_argument('--use-preproc', action='store_true')
    # 🚀 [추가] 정규화 스위치 (txv, imagenet, none)
    parser.add_argument('--norm-type', type=str, default='imagenet', choices=['imagenet', 'txv', 'none'])
    
    args = parser.parse_args()
    run_ablation(args)