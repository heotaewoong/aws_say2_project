"""
SageMaker 학습 스크립트 — DenseNet-121 (4차)
MIMIC-CXR 448x448 전용 학습
- DenseNet-121 (ImageNet pretrained, torchvision)
- Adam (lr=1e-4)
- ReduceLROnPlateau
- epochs=15, lr=1e-4, early stopping(patience=5)
- 8:2 train/val 자동 분리
- 체크포인트 저장 (스팟 인스턴스 중단 대비)
"""
import os, json, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image, ImageOps
import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import boto3
from io import BytesIO

LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices"
]


def load_image_s3(s3_client, bucket, key):
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return Image.open(BytesIO(obj['Body'].read())).convert('RGB')
    except Exception:
        return None


def apply_clahe(pil_img):
    """CLAHE 대비 향상 — 폐 구조를 더 선명하게"""
    img_np = np.array(pil_img.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    return Image.fromarray(enhanced).convert('RGB')


class MIMICCXRDataset(Dataset):
    """MIMIC-CXR 로컬 이미지 데이터셋 (SageMaker가 S3에서 미리 다운로드)
    balanced=True  : 이미 균형 잡힌 CSV — 내부 샘플링 없이 그대로 사용
    balanced=False : 기존 방식 — No Finding 5000개 캡 + 내부 shuffle
    """
    def __init__(self, csv_path, img_root, transform=None, max_samples=None, balanced=False):
        df = pd.read_csv(csv_path)
        df[LABELS] = df[LABELS].fillna(0)
        df = df.drop_duplicates(subset=['subject_id', 'study_id'])

        if balanced:
            # 이미 균형잡힌 CSV → 그대로 사용
            pass
        else:
            normal_df   = df[df['No Finding'] == 1]
            abnormal_df = df[df['No Finding'] != 1]
            if len(normal_df) > 5000:
                normal_df = normal_df.sample(5000, random_state=42)
            df = pd.concat([normal_df, abnormal_df]).sample(frac=1, random_state=42)

        if max_samples:
            df = df.sample(min(max_samples, len(df)), random_state=42)

        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform
        uncertain = (self.df[LABELS] == -1).sum().sum()
        normal_cnt = (self.df['No Finding'] == 1).sum()
        print(f"MIMIC-CXR: {len(self.df):,}건 로드 | 정상:{normal_cnt:,} 질환:{len(self.df)-normal_cnt:,}")
        print(f"  불확실 라벨(-1): {uncertain:,}개 → 마스킹 처리")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid  = str(int(row['subject_id']))
        stid = str(int(row['study_id']))
        img_dir = os.path.join(self.img_root, f"p{sid[:2]}", f"p{sid}", f"s{stid}")

        jpgs = []
        if os.path.exists(img_dir):
            jpgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

        if not jpgs:
            return self.__getitem__((idx + 1) % len(self))

        img_path = os.path.join(img_dir, jpgs[0])
        try:
            img = Image.open(img_path).convert('RGB')
            img = apply_clahe(img)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            img = self.transform(img)

        raw = row[LABELS].values.astype(float)
        labels = torch.FloatTensor(np.where(raw == -1, 0, raw))
        mask   = torch.FloatTensor((raw != -1).astype(float))
        return img, labels, mask


def find_best_thresholds(preds, gts):
    thresholds = []
    for c in range(len(LABELS)):
        gt_c, pr_c = gts[:, c], preds[:, c]
        best_t, best_f1 = 0.4, 0.0
        for t in np.arange(0.20, 0.81, 0.05):
            f1 = f1_score(gt_c, (pr_c >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(round(t, 2))
        thresholds.append(best_t)
    return thresholds


def compute_metrics(preds, gts, thresholds=None):
    if thresholds is None:
        thresholds = [0.4] * len(LABELS)
    metrics = {}
    aurocs, f1s, precs, recs = [], [], [], []
    for c in range(len(LABELS)):
        gt_c, pr_c = gts[:, c], preds[:, c]
        pred_bin = (pr_c >= thresholds[c]).astype(int)
        auroc = roc_auc_score(gt_c, pr_c) if len(np.unique(gt_c)) > 1 else float('nan')
        f1   = f1_score(gt_c, pred_bin, zero_division=0)
        prec = precision_score(gt_c, pred_bin, zero_division=0)
        rec  = recall_score(gt_c, pred_bin, zero_division=0)
        metrics[LABELS[c]] = {"auroc": auroc, "f1": f1, "precision": prec,
                               "recall": rec, "threshold": thresholds[c]}
        if not np.isnan(auroc): aurocs.append(auroc)
        f1s.append(f1); precs.append(prec); recs.append(rec)
    metrics["__mean__"] = {
        "auroc":     np.mean(aurocs) if aurocs else 0.0,
        "f1":        np.mean(f1s),
        "precision": np.mean(precs),
        "recall":    np.mean(recs),
    }
    return metrics


def print_metrics(epoch, total_epochs, train_loss, val_loss, metrics):
    m = metrics["__mean__"]
    print(f"\n{'='*68}")
    print(f"Epoch [{epoch}/{total_epochs}]  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
    print(f"  Mean  AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}  "
          f"Precision={m['precision']:.4f}  Recall={m['recall']:.4f}")
    print(f"{'─'*68}")
    print(f"  {'질환':<30} {'AUROC':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Thr':>5}")
    print(f"{'─'*68}")
    for label in LABELS:
        v = metrics[label]
        auroc_str = f"{v['auroc']:.4f}" if not np.isnan(v['auroc']) else "  N/A "
        print(f"  {label:<30} {auroc_str:>6} {v['f1']:>6.4f} {v['precision']:>6.4f}"
              f" {v['recall']:>6.4f} {v['threshold']:>5.2f}")
    print(f"{'='*68}\n")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    s3 = boto3.client('s3', region_name=args.region)

    # ── 전처리 ──
    train_tf = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ── 데이터셋 로드 ──
    print("\n데이터 준비 중...")
    use_split_csv = bool(args.train_csv and args.valid_csv)

    if use_split_csv:
        # train/valid CSV가 분리된 경우 (balanced CSV 사용)
        print(f"Train CSV: {args.train_csv}")
        print(f"Valid CSV: {args.valid_csv}")
        train_ds_raw = MIMICCXRDataset(
            csv_path=args.train_csv,
            img_root=args.mimic_images_dir,
            transform=train_tf,
            balanced=True,
        )
        val_ds_raw = MIMICCXRDataset(
            csv_path=args.valid_csv,
            img_root=args.mimic_images_dir,
            transform=val_tf,
            balanced=True,
        )
        train_loader = DataLoader(train_ds_raw, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds_raw, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)
        label_vals_for_pw = train_ds_raw.df[LABELS].values
        print(f"학습: {len(train_ds_raw):,}건 / 검증: {len(val_ds_raw):,}건")
    else:
        # 기존 방식: CSV 1개 → 8:2 내부 분리
        print(f"CSV: {args.mimic_csv} (8:2 내부 분리)")
        full_ds = MIMICCXRDataset(
            csv_path=args.mimic_csv,
            img_root=args.mimic_images_dir,
            transform=None,
            max_samples=args.mimic_max_samples,
            balanced=False,
        )
        val_size   = int(len(full_ds) * 0.2)
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(42))

        class TfDataset(Dataset):
            def __init__(self, subset, tf): self.subset = subset; self.tf = tf
            def __len__(self): return len(self.subset)
            def __getitem__(self, idx):
                img, label, mask = self.subset[idx]
                return self.tf(img), label, mask

        full_ds.transform = None

    if not use_split_csv:
        # ── WeightedRandomSampler (기존 방식) ──
        train_indices = train_ds.indices
        train_labels = full_ds.df.iloc[train_indices][LABELS].values

        pos_freq = (train_labels == 1).mean(axis=0)
        pos_freq = np.clip(pos_freq, 1e-4, 1.0)

        sample_weights = []
        for row in train_labels:
            pos_mask = row == 1
            if pos_mask.any():
                w = (1.0 / pos_freq[pos_mask]).max()
            else:
                w = 1.0
            sample_weights.append(w)

        sample_weights = torch.FloatTensor(sample_weights)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        print(f"WeightedRandomSampler 적용: 희귀 질환 오버샘플링")

        train_loader = DataLoader(TfDataset(train_ds, train_tf), batch_size=args.batch_size,
                                  sampler=sampler, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(TfDataset(val_ds, val_tf), batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)
        print(f"학습: {train_size:,}건 / 검증: {val_size:,}건")
        label_vals_for_pw = full_ds.df[LABELS].values

    # ── 모델 (DenseNet-121, ImageNet pretrained) ──
    print("\n모델 초기화 (DenseNet-121, ImageNet pretrained)...")
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 14)
    if torch.cuda.device_count() > 1:
        print(f"멀티 GPU 사용: {torch.cuda.device_count()}개")
        model = nn.DataParallel(model)
    model = model.to(device)

    # ── 손실함수 (pos_weight + 마스킹) ──
    pos_counts = (label_vals_for_pw == 1).sum(axis=0)
    neg_counts = (label_vals_for_pw == 0).sum(axis=0)
    pw = np.clip(neg_counts / (pos_counts + 1e-6), 1, 5)
    pos_weight = torch.FloatTensor(pw).to(device)
    print(f"\npos_weight 범위: {pw.min():.1f} ~ {pw.max():.1f}")

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def masked_loss(outputs, labels, masks):
        loss = bce(outputs, labels)
        loss = (loss * masks).sum() / (masks.sum() + 1e-6)
        return loss

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)

    # ── 체크포인트 복원 ──
    checkpoint_dir  = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")

    start_epoch = 1
    best_auroc  = 0.0
    no_improve  = 0
    history     = []

    if os.path.exists(checkpoint_path):
        print(f"체크포인트 복원: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_auroc  = ckpt['best_auroc']
        no_improve  = ckpt.get('no_improve', 0)
        history     = ckpt.get('history', [])
        print(f"  → epoch {ckpt['epoch']}부터 재개, best_auroc={best_auroc:.4f}")

    model_dir = args.model_dir

    # ── 학습 루프 ──
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for i, (imgs, labels, masks) in enumerate(train_loader):
            imgs, labels, masks = imgs.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = masked_loss(model(imgs), labels, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (i + 1) % 200 == 0:
                print(f"  Epoch {epoch} | Batch [{i+1}/{len(train_loader)}] | loss={loss.item():.4f}")
        train_loss /= len(train_loader)

        # 검증
        model.eval()
        val_loss = 0.0
        preds, gts = [], []
        with torch.no_grad():
            for imgs, labels, masks in val_loader:
                imgs, labels, masks = imgs.to(device), labels.to(device), masks.to(device)
                out = model(imgs)
                val_loss += masked_loss(out, labels, masks).item()
                preds.append(torch.sigmoid(out).cpu().numpy())
                gts.append(labels.cpu().numpy())
        val_loss /= len(val_loader)
        preds = np.vstack(preds)
        gts   = np.vstack(gts)

        best_thresholds = find_best_thresholds(preds, gts)
        metrics    = compute_metrics(preds, gts, thresholds=best_thresholds)
        mean_auroc = metrics["__mean__"]["auroc"]
        print_metrics(epoch, args.epochs, train_loss, val_loss, metrics)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            **{k: round(v, 4) for k, v in metrics["__mean__"].items()}
        })

        scheduler.step(mean_auroc)

        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            no_improve = 0
            state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state, os.path.join(model_dir, "chexnet_best.pth"))
            thr_dict = {LABELS[i]: best_thresholds[i] for i in range(len(LABELS))}
            with open(os.path.join(model_dir, "best_thresholds.json"), "w") as f:
                json.dump(thr_dict, f, indent=2)
            print(f"  ★ Best 모델 저장 (mAUROC: {mean_auroc:.4f})")
            print(f"  ★ Threshold 저장: { {k: v for k, v in thr_dict.items()} }")
        else:
            no_improve += 1
            print(f"  Early stopping 카운트: {no_improve}/{args.early_stop_patience}")
            if no_improve >= args.early_stop_patience:
                print(f"\n★ Early stopping! (epoch {epoch}, best mAUROC: {best_auroc:.4f})")
                break

        state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save({
            'epoch': epoch, 'model': state,
            'optimizer': optimizer.state_dict(),
            'best_auroc': best_auroc, 'no_improve': no_improve, 'history': history,
        }, checkpoint_path)
        print(f"  체크포인트 저장: epoch {epoch}")

    with open(os.path.join(model_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n학습 완료! Best mAUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",              type=int,   default=15)
    parser.add_argument("--batch-size",          type=int,   default=16)
    parser.add_argument("--lr",                  type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int,   default=5)
    parser.add_argument("--region",              type=str,   default="ap-northeast-2")
    parser.add_argument("--mimic-bucket",        type=str,   default="say2-2team-bucket")
    parser.add_argument("--mimic-csv",           type=str,   default="/opt/ml/input/data/mimic_csv/mimic-cxr-2.0.0-chexpert.csv")
    parser.add_argument("--train-csv",           type=str,   default="")   # 분리 CSV 사용 시
    parser.add_argument("--valid-csv",           type=str,   default="")   # 분리 CSV 사용 시
    parser.add_argument("--mimic-images-dir",    type=str,   default="/opt/ml/input/data/mimic_images")
    parser.add_argument("--mimic-max-samples",   type=int,   default=None)
    parser.add_argument("--model-dir",           type=str,   default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = parser.parse_args()
    train(args)
