"""
SageMaker 학습 스크립트 — CheXNet (DenseNet-121)
MIMIC-CXR 448x448 전용 학습
- epochs=15, lr=1e-4, early stopping(patience=5)
- 8:2 train/val 자동 분리
- 체크포인트 저장 (스팟 인스턴스 중단 대비)
- 매 epoch 상세 지표 출력 (AUROC, F1, Precision, Recall)
"""
import os, json, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
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
    img_np = np.array(pil_img.convert('L'))  # 그레이스케일
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    # 다시 RGB로 변환 (3채널 유지)
    return Image.fromarray(enhanced).convert('RGB')


class MIMICCXRDataset(Dataset):
    """MIMIC-CXR 로컬 이미지 데이터셋 (SageMaker가 S3에서 미리 다운로드)"""
    def __init__(self, csv_path, img_root, transform=None, max_samples=None):
        df = pd.read_csv(csv_path)
        # -1(불확실) 은 NaN으로 유지 — 손실 계산 시 마스킹으로 제외
        df[LABELS] = df[LABELS].fillna(0)  # NaN → 0, -1은 그대로 유지
        df = df.drop_duplicates(subset=['subject_id', 'study_id'])

        # ── 방법 3: 정상 언더샘플링 + WeightedRandomSampler (동시 적용) ──
        # 방법 1: 정상(No Finding==1) 케이스를 5,000장으로 언더샘플링
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
        print(f"MIMIC-CXR: {len(self.df):,}건 로드 | 질환:{len(abnormal_df):,} 정상:{len(normal_df):,}")
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
            img = apply_clahe(img)  # CLAHE 대비 향상
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            img = self.transform(img)

        raw = row[LABELS].values.astype(float)
        labels = torch.FloatTensor(np.where(raw == -1, 0, raw))   # -1 → 0 (값)
        mask   = torch.FloatTensor((raw != -1).astype(float))      # -1이면 0 (마스크)
        return img, labels, mask


def find_best_thresholds(preds, gts):
    """val 셋에서 클래스별 F1 최대화 threshold 탐색 (0.2 ~ 0.8)"""
    thresholds = []
    for c in range(len(LABELS)):
        gt_c, pr_c = gts[:, c], preds[:, c]
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.2, 0.81, 0.05):
            f1 = f1_score(gt_c, (pr_c >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds.append(best_t)
    return thresholds


def compute_metrics(preds, gts, thresholds=None):
    if thresholds is None:
        thresholds = [0.5] * len(LABELS)
    metrics = {}
    aurocs, f1s, precs, recs = [], [], [], []
    for c in range(len(LABELS)):
        gt_c, pr_c = gts[:, c], preds[:, c]
        pred_bin = (pr_c >= thresholds[c]).astype(int)
        auroc = roc_auc_score(gt_c, pr_c) if len(np.unique(gt_c)) > 1 else float('nan')
        f1   = f1_score(gt_c, pred_bin, zero_division=0)
        prec = precision_score(gt_c, pred_bin, zero_division=0)
        rec  = recall_score(gt_c, pred_bin, zero_division=0)
        metrics[LABELS[c]] = {"auroc": auroc, "f1": f1, "precision": prec, "recall": rec,
                               "threshold": thresholds[c]}
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
    print(f"\n{'='*60}")
    print(f"Epoch [{epoch}/{total_epochs}] train={train_loss:.4f} val={val_loss:.4f}")
    print(f"mAUROC={m['auroc']:.4f}  F1={m['f1']:.4f}  Prec={m['precision']:.4f}  Rec={m['recall']:.4f}")
    print(f"{'─'*60}")
    print(f"  {'질환':<30} {'AUROC':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Thr':>5}")
    print(f"{'─'*60}")
    for label in LABELS:
        v = metrics[label]
        auroc_str = f"{v['auroc']:.4f}" if not np.isnan(v['auroc']) else "  N/A"
        print(f"  {label:<30} {auroc_str:>6} {v['f1']:>6.4f} {v['precision']:>6.4f} {v['recall']:>6.4f} {v['threshold']:>5.2f}")
    print(f"{'='*60}\n")


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
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ── MIMIC-CXR 데이터셋 로드 후 8:2 분리 ──
    print("\n데이터 준비 중...")
    full_ds = MIMICCXRDataset(
        csv_path=args.mimic_csv,
        img_root=args.mimic_images_dir,
        transform=None,
        max_samples=args.mimic_max_samples
    )
    val_size   = int(len(full_ds) * 0.2)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    # transform 래퍼 (mask도 전달)
    class TfDataset(Dataset):
        def __init__(self, subset, tf): self.subset = subset; self.tf = tf
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            img, label, mask = self.subset[idx]
            return self.tf(img), label, mask

    full_ds.transform = None  # PIL 반환 보장

    # ── WeightedRandomSampler: 희귀 질환 오버샘플링 ──
    # train_ds 인덱스 기준으로 각 샘플의 가중치 계산
    train_indices = train_ds.indices
    train_labels = full_ds.df.iloc[train_indices][LABELS].values  # (N, 14)

    # 각 클래스별 양성 비율 계산 → 희귀할수록 가중치 높게
    pos_freq = (train_labels == 1).mean(axis=0)  # (14,)
    pos_freq = np.clip(pos_freq, 1e-4, 1.0)

    # 각 샘플의 가중치 = 포함된 양성 라벨 중 가장 희귀한 것의 역수
    sample_weights = []
    for row in train_labels:
        pos_mask = row == 1
        if pos_mask.any():
            # 희귀 질환 포함 시 높은 가중치
            w = (1.0 / pos_freq[pos_mask]).max()
        else:
            w = 1.0  # No Finding 등 음성만 있는 샘플은 기본 가중치
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

    # ── 모델 (멀티 GPU 지원) ──
    print("\n모델 초기화 (DenseNet-121, ImageNet pretrained)...")
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, 14)
    if torch.cuda.device_count() > 1:
        print(f"멀티 GPU 사용: {torch.cuda.device_count()}개")
        model = nn.DataParallel(model)
    model = model.to(device)

    # ── 손실함수 (Focal Loss + pos_weight + 마스킹) ──
    # pos_weight cap: 5 → 2 (recall 편향 완화, precision 개선)
    label_vals = full_ds.df[LABELS].values
    pos_counts = (label_vals == 1).sum(axis=0)
    neg_counts = (label_vals == 0).sum(axis=0)
    pw = np.clip(neg_counts / (pos_counts + 1e-6), 1, 2)
    pos_weight = torch.FloatTensor(pw).to(device)
    print(f"\npos_weight 범위: {pw.min():.1f} ~ {pw.max():.1f}")

    # Focal Loss: 쉬운 샘플 loss 축소 → 어려운 경계 샘플에 집중 → FP 감소 → Precision↑
    import torch.nn.functional as F

    def masked_loss(outputs, labels, masks, gamma=2.0):
        """-1(불확실) 라벨은 손실 계산에서 제외, Focal Loss 적용"""
        bce = F.binary_cross_entropy_with_logits(
            outputs, labels, pos_weight=pos_weight, reduction='none')
        prob = torch.sigmoid(outputs)
        pt = torch.where(labels == 1, prob, 1 - prob)   # 맞힌 확률
        focal = ((1 - pt) ** gamma) * bce               # 확실한 샘플 loss 축소
        loss = (focal * masks).sum() / (masks.sum() + 1e-6)
        return loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.5, verbose=True)

    # ── 체크포인트 복원 (스팟 중단 대비) ──
    checkpoint_dir  = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")

    start_epoch      = 1
    best_auroc       = 0.0
    no_improve       = 0
    history          = []
    best_thresholds  = [0.5] * len(LABELS)

    if os.path.exists(checkpoint_path):
        print(f"체크포인트 복원: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch     = ckpt['epoch'] + 1
        best_auroc      = ckpt['best_auroc']
        no_improve      = ckpt.get('no_improve', 0)
        history         = ckpt.get('history', [])
        best_thresholds = ckpt.get('best_thresholds', [0.5] * len(LABELS))
        print(f"  → epoch {ckpt['epoch']}부터 재개, best_auroc={best_auroc:.4f}")

    model_dir = args.model_dir

    # ── 학습 루프 ──
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for i, (imgs, labels, masks) in enumerate(train_loader):
            imgs, labels, masks = imgs.to(device), labels.to(device), masks.to(device)
            loss = masked_loss(model(imgs), labels, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
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

        # Best 모델 저장
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            no_improve = 0
            best_thresholds = best_thresholds  # 현재 epoch의 최적 threshold 유지
            state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state, os.path.join(model_dir, "chexnet_best.pth"))
            # threshold도 함께 저장 (추론 시 재사용)
            with open(os.path.join(model_dir, "best_thresholds.json"), "w") as f:
                json.dump({LABELS[i]: best_thresholds[i] for i in range(len(LABELS))}, f, indent=2)
            print(f"  ★ Best 모델 저장 (mAUROC: {mean_auroc:.4f})")
        else:
            no_improve += 1
            print(f"  Early stopping 카운트: {no_improve}/{args.early_stop_patience}")
            if no_improve >= args.early_stop_patience:
                print(f"\n★ Early stopping! (epoch {epoch}, best mAUROC: {best_auroc:.4f})")
                break

        # 체크포인트 저장 (매 epoch)
        state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save({
            'epoch': epoch, 'model': state,
            'optimizer': optimizer.state_dict(),
            'best_auroc': best_auroc, 'no_improve': no_improve,
            'history': history, 'best_thresholds': best_thresholds,
        }, checkpoint_path)
        print(f"  체크포인트 저장: epoch {epoch}")

    # 히스토리 저장
    with open(os.path.join(model_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n학습 완료! Best mAUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",              type=int,   default=15)
    parser.add_argument("--batch-size",          type=int,   default=32)
    parser.add_argument("--lr",                  type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int,   default=5)
    parser.add_argument("--region",              type=str,   default="ap-northeast-2")
    parser.add_argument("--mimic-bucket",        type=str,   default="say2-2team-bucket")
    parser.add_argument("--mimic-csv",           type=str,   default="/opt/ml/input/data/mimic_csv/mimic-cxr-2.0.0-chexpert.csv")
    parser.add_argument("--mimic-images-dir",    type=str,   default="/opt/ml/input/data/mimic_images")
    parser.add_argument("--mimic-max-samples",   type=int,   default=None)
    parser.add_argument("--model-dir",           type=str,   default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = parser.parse_args()
    train(args)
