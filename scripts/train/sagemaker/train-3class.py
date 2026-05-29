"""
SageMaker 학습 스크립트 — DenseNet-121 3-class
Edema (0) / Lung Opacity (1) / Pneumonia (2)
- DenseNet-121 (ImageNet pretrained, torchvision)
- CrossEntropyLoss (3-class 단일 라벨 분류)
- Adam (lr=1e-4) + ReduceLROnPlateau
- 이미 분리된 train/valid CSV 사용 (single_3label_*)
"""
import os, json, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, accuracy_score, confusion_matrix,
                              classification_report)

CLASSES = ["Edema", "Lung Opacity", "Pneumonia"]  # class_id: 0, 1, 2
NUM_CLASSES = len(CLASSES)


def apply_clahe(pil_img):
    img_np = np.array(pil_img.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    return Image.fromarray(enhanced).convert('RGB')


class ThreeClassMIMICDataset(Dataset):
    """single_3label_*.csv 전용 데이터셋
    CSV 컬럼: subject_id, study_id, Edema, Lung Opacity, Pneumonia
    각 행에서 1.0인 컬럼 → class_id (0, 1, 2)
    """
    def __init__(self, csv_path, img_root, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform

        # 각 행에서 1.0인 컬럼의 인덱스를 class_id로 사용
        self.df['class_id'] = self.df[CLASSES].values.argmax(axis=1)

        print(f"데이터셋 로드: {len(self.df):,}건")
        for i, cls in enumerate(CLASSES):
            cnt = (self.df['class_id'] == i).sum()
            print(f"  {cls}: {cnt:,}건")

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

        label = torch.tensor(int(row['class_id']), dtype=torch.long)
        return img, label


def compute_metrics(preds_prob, preds_bin, gts):
    """3-class 분류 지표 계산"""
    # AUROC: OvR(One-vs-Rest) macro 평균
    try:
        auroc = roc_auc_score(gts, preds_prob, multi_class='ovr', average='macro')
    except Exception:
        auroc = float('nan')

    acc  = accuracy_score(gts, preds_bin)
    f1   = f1_score(gts, preds_bin, average='macro', zero_division=0)
    prec = precision_score(gts, preds_bin, average='macro', zero_division=0)
    rec  = recall_score(gts, preds_bin, average='macro', zero_division=0)
    cm   = confusion_matrix(gts, preds_bin, labels=[0, 1, 2])

    # 클래스별 F1
    f1_per_class = f1_score(gts, preds_bin, average=None, zero_division=0)

    return {
        "auroc": auroc, "acc": acc, "f1": f1,
        "precision": prec, "recall": rec, "cm": cm,
        "f1_per_class": f1_per_class,
    }


def print_metrics(epoch, total_epochs, train_loss, val_loss, m):
    print(f"\n{'='*65}")
    print(f"Epoch [{epoch}/{total_epochs}]  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
    print(f"  mAUROC={m['auroc']:.4f}  Acc={m['acc']:.4f}  "
          f"F1={m['f1']:.4f}  Prec={m['precision']:.4f}  Rec={m['recall']:.4f}")
    print(f"  클래스별 F1:")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls:<22}: {m['f1_per_class'][i]:.4f}")
    print(f"  Confusion Matrix (행=정답, 열=예측):")
    header = "         " + "  ".join(f"{c[:6]:>8}" for c in CLASSES)
    print(f"  {header}")
    for i, cls in enumerate(CLASSES):
        row_str = "  ".join(f"{m['cm'][i][j]:>8}" for j in range(NUM_CLASSES))
        print(f"  {cls[:6]:>8} | {row_str}")
    print(f"{'='*65}\n")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

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

    print("\n데이터 준비 중...")
    print("[Train]")
    train_ds = ThreeClassMIMICDataset(args.train_csv, args.mimic_images_dir, transform=train_tf)
    print("[Valid]")
    val_ds   = ThreeClassMIMICDataset(args.valid_csv, args.mimic_images_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── 모델 (DenseNet-121, 출력 3개) ──
    print("\n모델 초기화 (DenseNet-121 → 3-class 출력)...")
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)

    checkpoint_dir  = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_3class.pth")

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

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (i + 1) % 20 == 0:
                print(f"  Epoch {epoch} | Batch [{i+1}/{len(train_loader)}] | loss={loss.item():.4f}")
        train_loss /= len(train_loader)

        # 검증
        model.eval()
        val_loss = 0.0
        preds_prob, preds_bin, gts = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_loss += criterion(out, labels).item()
                prob = torch.softmax(out, dim=1).cpu().numpy()  # (N, 3) 각 클래스 확률
                pred = out.argmax(dim=1).cpu().numpy()
                preds_prob.extend(prob)
                preds_bin.extend(pred)
                gts.extend(labels.cpu().numpy())
        val_loss /= len(val_loader)

        preds_prob = np.array(preds_prob)  # (N, 3)
        preds_bin  = np.array(preds_bin)
        gts        = np.array(gts)

        m = compute_metrics(preds_prob, preds_bin, gts)
        print_metrics(epoch, args.epochs, train_loss, val_loss, m)

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "auroc":      round(float(m['auroc']), 4),
            "acc":        round(m['acc'], 4),
            "f1":         round(m['f1'], 4),
            "precision":  round(m['precision'], 4),
            "recall":     round(m['recall'], 4),
            "f1_ED":      round(float(m['f1_per_class'][0]), 4),
            "f1_LO":      round(float(m['f1_per_class'][1]), 4),
            "f1_PN":      round(float(m['f1_per_class'][2]), 4),
        })

        scheduler.step(m['auroc'])

        if m['auroc'] > best_auroc:
            best_auroc = m['auroc']
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(model_dir, "3class_best.pth"))
            print(f"  ★ Best 모델 저장 (mAUROC: {m['auroc']:.4f})")
        else:
            no_improve += 1
            print(f"  Early stopping 카운트: {no_improve}/{args.early_stop_patience}")
            if no_improve >= args.early_stop_patience:
                print(f"\n★ Early stopping! (epoch {epoch}, best mAUROC: {best_auroc:.4f})")
                break

        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_auroc': best_auroc, 'no_improve': no_improve, 'history': history,
        }, checkpoint_path)
        print(f"  체크포인트 저장: epoch {epoch}")

    with open(os.path.join(model_dir, "training_history_3class.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n학습 완료! Best mAUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",              type=int,   default=15)
    parser.add_argument("--batch-size",          type=int,   default=16)
    parser.add_argument("--lr",                  type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int,   default=5)
    parser.add_argument("--region",              type=str,   default="ap-northeast-2")
    parser.add_argument("--train-csv",           type=str,   default="/opt/ml/input/data/mimic_csv/pe_lo_co_train_3labels.csv")
    parser.add_argument("--valid-csv",           type=str,   default="/opt/ml/input/data/mimic_csv/pe_lo_co_valid_3labels.csv")
    parser.add_argument("--mimic-images-dir",    type=str,   default="/opt/ml/input/data/mimic_images")
    parser.add_argument("--model-dir",           type=str,   default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = parser.parse_args()
    train(args)
