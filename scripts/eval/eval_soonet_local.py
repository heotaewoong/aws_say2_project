"""
SooNet 로컬 성능 평가 — CheXpert validation set
실행:
    cd aws_say2_project_vision
    python eval_soonet_local.py

필요 조건:
    - /tmp/valid.csv  (이미 다운로드됨)
    - S3에서 valid 이미지 접근 가능 (AWS 키 .env에 설정됨)
    - model/chexnet_unet_crop_best.pth 존재
"""
import os, sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# .env 로드
_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env):
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from soo_net import SooNetEngine

LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

THRESHOLD = 0.3  # rag_pipeline.py와 동일한 임계값

def download_image(s3_path: str, local_path: str) -> bool:
    """S3에서 이미지 다운로드"""
    import subprocess
    result = subprocess.run(
        ["aws", "s3", "cp", s3_path, local_path],
        capture_output=True, text=True
    )
    return result.returncode == 0

def evaluate(max_samples: int = 50):
    """
    CheXpert valid set에서 max_samples개 이미지로 SooNet 성능 평가
    """
    # valid CSV 로드
    csv_path = "/tmp/valid.csv"
    if not os.path.exists(csv_path):
        print("❌ /tmp/valid.csv 없음. 먼저 실행:")
        print('   aws s3 cp "s3://say2-2team-bucket/csv/chexpert_valid_clean.csv" /tmp/valid.csv')
        sys.exit(1)

    df = pd.read_csv(csv_path)
    # -1 (불확실) → 1로 처리 (CheXpert 표준)
    df[LABELS] = df[LABELS].fillna(0).replace(-1, 1)

    print(f"✅ valid.csv 로드: {len(df)}개 이미지")
    print(f"   평가 샘플 수: {min(max_samples, len(df))}개")
    print(f"   임계값: {THRESHOLD}")

    # SooNet 초기화
    model_path = "model/chexnet_unet_crop_best.pth"
    if not os.path.exists(model_path):
        print(f"❌ 모델 없음: {model_path}")
        sys.exit(1)

    engine = SooNetEngine(model_path=model_path)
    print(f"✅ SooNet 로드 완료\n")

    all_preds = []
    all_labels = []
    success = 0
    fail = 0

    sample_df = df.head(max_samples)

    for idx, row in sample_df.iterrows():
        path = row['Path']  # CheXpert-v1.0/valid/patient.../view.jpg
        # patient.../view.jpg 부분만 추출
        parts = path.split('/')
        if 'valid' in parts:
            rel = '/'.join(parts[parts.index('valid')+1:])
        else:
            rel = '/'.join(parts[2:])

        local_path = f"/tmp/eval_img_{idx}.jpg"
        s3_path = f"s3://say2-2team-bucket/cheXpert_data/valid_only/{rel}"

        # 다운로드
        if not os.path.exists(local_path):
            ok = download_image(s3_path, local_path)
            if not ok:
                fail += 1
                continue

        # 추론
        try:
            preds_dict = engine.predict(local_path)
            probs = [preds_dict[label][0] for label in LABELS]
            gt = [row[label] for label in LABELS]
            all_preds.append(probs)
            all_labels.append(gt)
            success += 1

            if success % 10 == 0:
                print(f"  진행: {success}/{max_samples} 완료...")

            # 임시 파일 삭제
            os.remove(local_path)

        except Exception as e:
            fail += 1
            continue

    print(f"\n✅ 추론 완료: {success}개 성공, {fail}개 실패\n")

    if success < 5:
        print("❌ 샘플 수 부족 — 평가 불가")
        return

    all_preds = np.array(all_preds)   # (N, 14)
    all_labels = np.array(all_labels) # (N, 14)

    # ── 지표 계산 ──
    print("=" * 65)
    print(f"{'질환':<35} {'AUROC':>7} {'F1':>7} {'Sens':>7} {'Spec':>7}")
    print("-" * 65)

    auroc_list = []
    f1_list = []

    for i, label in enumerate(LABELS):
        y_true = all_labels[:, i]
        y_prob = all_preds[:, i]
        y_pred = (y_prob >= THRESHOLD).astype(int)

        # 양성 샘플이 없으면 스킵
        if len(np.unique(y_true)) < 2:
            print(f"  {label:<33} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7}  (양성 없음)")
            continue

        auroc = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)

        auroc_list.append(auroc)
        f1_list.append(f1)

        flag = " ⭐" if auroc >= 0.8 else ""
        print(f"  {label:<33} {auroc:>7.4f} {f1:>7.4f} {sens:>7.4f} {spec:>7.4f}{flag}")

    print("=" * 65)
    if auroc_list:
        print(f"  {'평균 (유효 질환)':<33} {np.mean(auroc_list):>7.4f} {np.mean(f1_list):>7.4f}")
    print("=" * 65)

    # 결과 저장
    result_path = "eval_soonet_result.csv"
    rows = []
    for i, label in enumerate(LABELS):
        y_true = all_labels[:, i]
        y_prob = all_preds[:, i]
        y_pred = (y_prob >= THRESHOLD).astype(int)
        if len(np.unique(y_true)) < 2:
            continue
        auroc = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rows.append({"Disease": label, "AUROC": round(auroc, 4), "F1": round(f1, 4)})
    pd.DataFrame(rows).to_csv(result_path, index=False)
    print(f"\n결과 저장: {result_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50, help="평가할 이미지 수 (기본 50)")
    args = parser.parse_args()
    evaluate(max_samples=args.samples)
