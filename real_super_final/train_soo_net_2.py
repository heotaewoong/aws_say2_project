import argparse
import os
import ast
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torchvision.ops as ops
import torchvision.transforms.functional as TF

from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score

from soo_net_2 import SooNetEngine
from unet_lung_model import UNet
from visualizer import MedicalVisualizer
from chexpert_dataset import ChexpertDataset

# Target Disease Labels
LABEL_ORDER = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "No Finding", "Pleural Effusion", 
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

# =====================================================================
# 1. Loss Function (Focal Loss for Long-tail Distribution)
# =====================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# =====================================================================
# 2. Data Parser
# =====================================================================
def prepare_unified_df(csv_path, chexpert_root, mimic_root):
    df = pd.read_csv(csv_path)

    # Identify path column dynamically
    if 'image_path' in df.columns:
        path_col = 'image_path'
    elif 'Path' in df.columns:
        path_col = 'Path'
    else:
        raise KeyError(f"❌ {csv_path}에서 경로 컬럼('image_path' 또는 'Path')을 찾을 수 없습니다!")
    
    flat_data = []
    chexpert_batches = ["CheXpert-v1.0 batch 1 (validate & csv)", "CheXpert-v1.0 batch 2 (train 1)", 
                        "CheXpert-v1.0 batch 3 (train 2)", "CheXpert-v1.0 batch 4 (train 3)"]

    for _, row in df.iterrows():
        raw_path = str(row[path_col])
        
        # CheXpert Data Parsing
        if 'patient' in raw_path:
            core_path = raw_path.split('patient')[-1]
            core_path = 'patient' + core_path
            
            for folder in chexpert_batches:
                full_path = os.path.join(chexpert_root, folder, core_path)
                if os.path.exists(full_path):
                    flat_data.append({'path': full_path, 'labels': row[LABEL_ORDER].values})
                    break
                    
        # MIMIC Data Parsing
        elif 'files/p' in raw_path:
            core_path = raw_path.split('files/')[-1]
            full_path = os.path.join(mimic_root, 'files', core_path)
            
            if os.path.exists(full_path):
                flat_data.append({'path': full_path, 'labels': row[LABEL_ORDER].values})
                
    final_df = pd.DataFrame(flat_data)
    print(f"✅ 하이브리드 파싱 완료: 총 {len(final_df)}장 (CheXpert + MIMIC 통합)")
    return final_df
    
# =====================================================================
# 3. Image Preprocessing & Transforms
# =====================================================================
class ChestXrayPreprocess:
    def __init__(self, target_size=(512, 512), clip_limit=2.0):
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
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
        self.clahe_preprocess = ChestXrayPreprocess(target_size=target_size, clip_limit=2.0)

    def __call__(self, img):
        img = self.clahe_preprocess(img) # RGB 이미지 반환
        # U-Net 입력용으로 3채널, 0.0~1.0 스케일의 텐서로 변환
        return transforms.ToTensor()(img)

# =====================================================================
# 4. GPU-Accelerated ROI Masking 
# =====================================================================
def process_unet_crops_with_masks(images, masks, padding=10, target_size=(512, 512)):
    B, C_img, H, W = images.shape
    device = images.device

    combined_mask = masks.sum(dim=1) > 0.5
    is_empty = ~torch.any(combined_mask.view(B, -1), dim=1)
    
    rows = torch.any(combined_mask, dim=2)
    cols = torch.any(combined_mask, dim=1)

    y_min_indices = torch.argmax(rows.int(), dim=1)
    y_max_indices = H - 1 - torch.argmax(torch.flip(rows, dims=[1]).int(), dim=1)
    x_min_indices = torch.argmax(cols.int(), dim=1)
    x_max_indices = W - 1 - torch.argmax(torch.flip(cols, dims=[1]).int(), dim=1)

    y_mins = torch.clamp(y_min_indices - padding, min=0).float()
    y_maxs = torch.clamp(y_max_indices + padding, max=H).float()
    x_mins = torch.clamp(x_min_indices - padding, min=0).float()
    x_maxs = torch.clamp(x_max_indices + padding, max=W).float()

    crop_heights = y_maxs - y_mins
    crop_widths = x_maxs - x_mins

    center_y = (y_mins + y_maxs) / 2
    center_x = (x_mins + x_maxs) / 2
    
    max_dims = torch.max(crop_heights, crop_widths)
    
    roi_x1 = center_x - max_dims / 2
    roi_y1 = center_y - max_dims / 2
    roi_x2 = center_x + max_dims / 2
    roi_y2 = center_y + max_dims / 2

    batch_indices = torch.arange(B, device=device).view(-1, 1).float()
    boxes_for_roi = torch.stack([roi_x1, roi_y1, roi_x2, roi_y2], dim=1)
    boxes_for_roi[is_empty] = 0.0 # 비어있는 마스크의 박스는 (0,0,0,0)으로 만들어 에러 방지
    boxes_for_roi = torch.cat([batch_indices, boxes_for_roi], dim=1)

    resized_imgs_rgb = ops.roi_align(images, boxes_for_roi, output_size=target_size, aligned=True)
    final_masks = ops.roi_align(masks, boxes_for_roi, output_size=target_size, aligned=True)

    final_imgs = resized_imgs_rgb.mean(dim=1, keepdim=True)
    final_imgs[is_empty] = 0.0
    final_masks[is_empty] = 0.0

    # TXV Scaling (-1024 ~ 1024) for DenseNet Pre-trained range
    txv_scaled_imgs = (final_imgs * 2048.0) - 1024.0
    return txv_scaled_imgs, final_masks

# =====================================================================
# 5. Dataset with Precomputed Masks
# =====================================================================
class PrecomputedMaskDataset(Dataset):
    def __init__(self, df, mask_dir, transform=None, augment=False):
        self.df = df
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_full_path = row['path']
        
        try:
            image = Image.open(img_full_path).convert('L')
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))
            

        unique_part = "_".join(img_full_path.split(os.sep)[-4:]).replace('.jpg', '').replace('.png', '')
        mask_path = os.path.join(self.mask_dir, f"{unique_part}.pt")
        
        try:
            mask_tensor = torch.load(mask_path)
        except FileNotFoundError:
            mask_tensor = torch.zeros((2, 512, 512))

        label = torch.FloatTensor(row['labels'].astype(float))

        # Color Augmentation
        if self.augment:
            image = transforms.ColorJitter(brightness=0.3, contrast=0.3)(image)

        # 기본 변환 (CLAHE, Pad, ToTensor) 적용
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = TF.to_tensor(image)

        # Geometric Augmentation
        if self.augment:
            if torch.rand(1) < 0.5:
                image_tensor = TF.hflip(image_tensor)
                mask_tensor = TF.hflip(mask_tensor)

            affine_params = transforms.RandomAffine.get_params(
                degrees=(0, 0), 
                translate=(0.1, 0.1), 
                scale_ranges=(0.9, 1.1), 
                shears=None, 
                img_size=list(image_tensor.shape[1:])
            )
            image_tensor = TF.affine(image_tensor, *affine_params, interpolation=TF.InterpolationMode.BILINEAR)
            mask_tensor = TF.affine(mask_tensor, *affine_params, interpolation=TF.InterpolationMode.NEAREST)

        return image_tensor, mask_tensor, label

# =====================================================================
# 6. Main Training Loop
# =====================================================================
def train(args):
    # Device setup prioritizing CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training Environment Initiated on Device: {device}")

    base_transform = Base_Transform(target_size=(512, 512))
    history = {'train_loss': [], 'val_loss': [], 'val_auroc': []}
    visualizer = MedicalVisualizer(labels=LABEL_ORDER, output_dir=args.plot_dir)
    
    # Path configuration
    CSV_PATH = os.path.join(args.train_csv_dir, "chexpert_balanced_u_ones.csv")
    SAVE_MODEL_NAME = os.path.join(args.model_dir, "anatomy_soonet_unified_best.pth")
    
    # Load the unified dataframe from a single CSV
    unified_df = prepare_unified_df(CSV_PATH, args.chexpert_dir, args.mimic_dir)

    # Shuffle and split the dataframe into 80:20 for train and validation
    unified_df = unified_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_index = int(0.8 * len(unified_df))
    train_df = unified_df.iloc[:split_index]
    val_df = unified_df.iloc[split_index:]
    print(f"데이터셋 분할 완료: Train {len(train_df)}장, Validation {len(val_df)}장")

    # UNet Initialization for pre-computation
    print("✂️ U-Net 모델 로드 중...")
    unet = UNet(n_channels=3, n_classes=3).to(device)
    checkpoint = torch.load(args.unet_weight_path, map_location=device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ U-Net 가중치 로드 완료! (Best Dice: {checkpoint['dice_score']:.4f})")
    unet.eval() 

    # Mask Pre-computation
    mask_cache_dir = os.path.join(args.model_dir, 'mask_cache')
    train_mask_dir = os.path.join(mask_cache_dir, 'train')
    val_mask_dir = os.path.join(mask_cache_dir, 'val')
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)

    def precompute_masks(df, transform, mask_save_dir, dataset_name):
        if len(os.listdir(mask_save_dir)) >= len(df):
            print(f"Masks for {dataset_name} already exist. Skipping pre-computation.")
            return

        precompute_batch_size = 16
        print(f"Generating Pre-computed Masks for {dataset_name} ({len(df)} samples) with batch size {precompute_batch_size}...")
        dataset = ChexpertDataset(df, transform)
        loader = DataLoader(dataset, batch_size=precompute_batch_size, shuffle=False, num_workers=args.num_workers)
        
        with torch.no_grad():
            for i, (imgs, _) in enumerate(loader):
                imgs = imgs.to(device)
                raw_masks = torch.sigmoid(unet(imgs))
                raw_masks = raw_masks[:, 1:, :, :]

                for j in range(imgs.size(0)):
                    df_idx = i * precompute_batch_size + j
                    if df_idx < len(df):
                        img_path = df.iloc[df_idx]['path']
                        unique_part = "_".join(img_path.split(os.sep)[-4:]).replace('.jpg', '').replace('.png', '')
                        mask_filename = f"{unique_part}.pt"
                        mask_save_path = os.path.join(mask_save_dir, mask_filename)
                        torch.save(raw_masks[j].cpu(), mask_save_path)
                
                if (i+1) % 100 == 0:
                    print(f"   - {dataset_name} 마스크 생성 중... [배치 {i+1}/{len(loader)}]")
        print(f"✅ {dataset_name} 마스크 사전 생성 완료!")

    precompute_masks(train_df, base_transform, train_mask_dir, "Train")
    precompute_masks(val_df, base_transform, val_mask_dir, "Validation")

    # DataLoader setup with pre-computed masks
    train_dataset = PrecomputedMaskDataset(train_df, train_mask_dir, base_transform, augment=True)
    val_dataset = PrecomputedMaskDataset(val_df, val_mask_dir, base_transform, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"Train Loader: {len(train_loader)} batches | Val Loader: {len(val_loader)} batches")

    # AnatomySooNet Initialization
    print("Initializing Anatomy-SooNet Engine...")
    engine = SooNetEngine(model_path=None, num_classes=len(LABEL_ORDER)) 
    model = engine.model 
    model.to(device)

    # PyTorch 2.0 Compiler Optimization
    if hasattr(torch, 'compile'):
        print("Applying PyTorch 2.0 Optimization (torch.compile)...")
        model = torch.compile(model)

    # Optimizer, Loss, and Scheduler Configuration
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)

    best_auroc = 0.0
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for i, (imgs, masks, lbls) in enumerate(train_loader):
            imgs, masks, lbls = imgs.to(device), masks.to(device), lbls.to(device)
            
            txv_imgs, final_masks = process_unet_crops_with_masks(imgs, masks, target_size=(512, 512))
            
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(txv_imgs, final_masks)
                loss = criterion(outputs, lbls)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Batch [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        history['train_loss'].append(epoch_loss / len(train_loader))

        # Validation Loop
        model.eval()
        all_preds, all_labels = [], []
        val_epoch_loss = 0
        
        with torch.no_grad():
            for imgs, masks, lbls in val_loader:
                imgs, masks, lbls = imgs.to(device), masks.to(device), lbls.to(device)
                txv_imgs, final_masks = process_unet_crops_with_masks(imgs, masks, target_size=(512, 512))
                
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    outputs = model(txv_imgs, final_masks) 
                    v_loss = criterion(outputs, lbls)
                    val_epoch_loss += v_loss.item()
                
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(lbls.cpu().numpy())
        
        history['val_loss'].append(val_epoch_loss / len(val_loader))
        
        val_preds = np.vstack(all_preds)
        val_labels = np.vstack(all_labels)
        
        # Metrics Evaluation
        print(f"\n[Epoch {epoch+1} Validation Report]")
        print(f"{'Disease':<27s} | {'AUROC':<6s} | {'Best_Th':<7s} | {'Prec':<6s} | {'Recall':<6s} | {'F1':<6s}")
        print("-" * 75)

        val_auroc_list, val_f1_list, best_thresholds = [], [], []

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
                best_thresholds.append(best_threshold)
                
                print(f"{class_name:<27s} | {auroc:.4f} | {best_threshold:.4f}  | {val_prec:.4f} | {val_rec:.4f} | {val_f1:.4f}")
            else:
                best_thresholds.append(0.5)
                print(f"{class_name:<27s} |  N/A   |  N/A     |  N/A   |  N/A   |  N/A")
        
        print("-" * 75)
        
        avg_auroc = np.mean(val_auroc_list) if len(val_auroc_list) > 0 else 0.0
        avg_f1 = np.mean(val_f1_list) if len(val_f1_list) > 0 else 0.0
        history['val_auroc'].append(avg_auroc)
            
        print(f"Epoch Summary -> Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f} | Macro AUROC: {avg_auroc:.4f} | Macro F1: {avg_f1:.4f}\n")
        
        # Scheduler Update (Bug Fixed: Removed parameter for CosineAnnealingWarmRestarts)
        scheduler.step()

        # Save Best Model & Generate Visualizations
        if avg_auroc > best_auroc:
            best_auroc = avg_auroc
            torch.save(model.state_dict(), SAVE_MODEL_NAME)
            print(f"New Best Model Saved: {SAVE_MODEL_NAME} (AUROC: {avg_auroc:.4f})")
            
            try:
                visualizer.generate_all_reports(val_labels, val_preds, best_thresholds, history)
            except Exception as e:
                print(f"Warning: Failed to generate visualization reports. Training continues. ({e})")

# =====================================================================
# 7. Execution Entry Point
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train-csv-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_CSV'))
    parser.add_argument('--valid-csv-dir', type=str, default=os.environ.get('SM_CHANNEL_VALID_CSV'))
    parser.add_argument('--chexpert-dir', type=str, default=os.environ.get('SM_CHANNEL_CHEXPERT'))
    parser.add_argument('--mimic-dir', type=str, default=os.environ.get('SM_CHANNEL_MIMIC'))

    parser.add_argument('--plot-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './plots'))
    
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--lr-patience', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=32)

    parser.add_argument('--unet-weight-path', type=str, default='unet_lung_heart_best.pth')

    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    train(args)