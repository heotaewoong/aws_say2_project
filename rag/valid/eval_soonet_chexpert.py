"""
SooNet 검증 — CheXpert validation set 전수 평가
=================================================
팀 보고용 "실제 병변 vs 예측 병변" 정량 비교

실행:
    cd aws_say2_project_vision
    python rag/valid/eval_soonet_chexpert.py [--samples 234]

결과:
    rag/valid/soonet_chexpert_results.csv    ← 병변별 AUROC/F1
    rag/valid/soonet_chexpert_summary.json   ← 종합 요약
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 → sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# .env 자동 로드
_env = os.path.join(ROOT, ".env")
if os.path.exists(_env):
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

os.environ.setdefault("AWS_DEFAULT_REGION", "ap-northeast-2")

from soo_net import SooNetEngine

# CheXpert 14개 라벨 (SooNet 출력 순서와 동일)
LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

THRESHOLD = 0.3  # rag_pipeline.py 와 동일

VALID_CSV_S3 = "s3://say2-2team-bucket/csv/chexpert_valid_clean.csv"
VALID_CSV_LOCAL = "/tmp/valid.csv"


def download_file(s3_path: str, local_path: str) -> bool:
    import subprocess
    result = subprocess.run(
        ["aws", "s3", "cp", s3_path, local_path, "--region", "ap-northeast-2"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def evaluate(max_samples: int = 234):
    # ── valid.csv 준비 ────────────────────────────────────────────
    if not os.path.exists(VALID_CSV_LOCAL):
        print(f"📥 valid.csv 다운로드: {VALID_CSV_S3}")
        if not download_file(VALID_CSV_S3, VALID_CSV_LOCAL):
            print("❌ valid.csv 다운로드 실패")
            sys.exit(1)

    df = pd.read_csv(VALID_CSV_LOCAL)
    # CheXpert 라벨 규칙: -1(불확실) → 1, NaN → 0
    df[LABELS] = df[LABELS].fillna(0).replace(-1, 1)

    print(f"✅ valid.csv 로드: {len(df)}장")
    print(f"   평가 대상: {min(max_samples, len(df))}장")
    print(f"   임계값: {THRESHOLD}")

    # ── SooNet 로드 ──────────────────────────────────────────────
    model_path = "model/chexnet_unet_crop_best.pth"
    if not os.path.exists(model_path):
        print(f"❌ 모델 없음: {model_path}")
        sys.exit(1)

    print(f"\n🧠 SooNet 로드: {model_path}")
    engine = SooNetEngine(model_path=model_path)
    print("✅ 로드 완료\n")

    # ── 추론 루프 ─────────────────────────────────────────────────
    all_preds = []
    all_labels = []
    success = 0
    fail = 0

    sample_df = df.head(max_samples)

    print(f"🚀 추론 시작 ({len(sample_df)}장)...")
    for idx, row in sample_df.iterrows():
        # Path 컬럼: "CheXpert-v1.0/valid/patient64541/study1/view1_frontal.jpg"
        # → "patient64541/study1/view1_frontal.jpg" 로 변환
        rel_path = row['Path'].replace("CheXpert-v1.0/valid/", "")
        s3_path = f"s3://say2-2team-bucket/cheXpert_data/valid_only/{rel_path}"
        local_img = f"/tmp/eval_img_{idx}.jpg"

        # S3 → 로컬
        if not download_file(s3_path, local_img):
            fail += 1
            continue

        try:
            preds_dict = engine.predict(local_img)
            # SooNet 출력 → 14개 확률 순서 유지
            probs = [preds_dict[label][0] for label in LABELS]
            gt = [row[label] for label in LABELS]
            all_preds.append(probs)
            all_labels.append(gt)
            success += 1

            if success % 25 == 0:
                print(f"  진행: {success}/{len(sample_df)}")

            os.remove(local_img)

        except Exception as e:
            fail += 1
            if os.path.exists(local_img):
                os.remove(local_img)
            continue

    print(f"\n✅ 추론 완료: {success}장 성공 / {fail}장 실패")

    if success < 10:
        print("❌ 샘플 수 부족 — 평가 불가")
        return

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── 지표 계산 ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'병변 (Label)':<32} {'AUROC':>8} {'F1':>8} {'Sens':>8} {'Spec':>8} {'N+':>6}")
    print("-" * 80)

    results = []
    auroc_list = []
    f1_list = []

    for i, label in enumerate(LABELS):
        y_true = all_labels[:, i]
        y_prob = all_preds[:, i]
        y_pred = (y_prob >= THRESHOLD).astype(int)

        n_pos = int(y_true.sum())

        if len(np.unique(y_true)) < 2:
            print(f"  {label:<30} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {n_pos:>6}  (양성 없음)")
            results.append({
                "disease": label,
                "n_positive": n_pos,
                "auroc": None, "f1": None, "sensitivity": None, "specificity": None,
            })
            continue

        auroc = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)

        auroc_list.append(auroc)
        f1_list.append(f1)

        flag = " ⭐" if auroc >= 0.8 else ""
        print(f"  {label:<30} {auroc:>8.4f} {f1:>8.4f} {sens:>8.4f} {spec:>8.4f} {n_pos:>6}{flag}")

        results.append({
            "disease": label,
            "n_positive": n_pos,
            "auroc": round(auroc, 4),
            "f1": round(f1, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })

    print("=" * 80)
    mean_auroc = np.mean(auroc_list) if auroc_list else 0
    mean_f1 = np.mean(f1_list) if f1_list else 0
    print(f"  {'평균 (유효 병변 only)':<30} {mean_auroc:>8.4f} {mean_f1:>8.4f}")
    print("=" * 80)

    # ── 결과 저장 ─────────────────────────────────────────────────
    out_csv = "rag/valid/soonet_chexpert_results.csv"
    out_json = "rag/valid/soonet_chexpert_summary.json"

    pd.DataFrame(results).to_csv(out_csv, index=False)

    summary = {
        "dataset": "CheXpert validation set",
        "n_total": success,
        "n_failed": fail,
        "threshold": THRESHOLD,
        "model": model_path,
        "mean_auroc": round(mean_auroc, 4),
        "mean_f1": round(mean_f1, 4),
        "valid_diseases": len(auroc_list),
        "per_disease": results,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n📄 결과 저장:")
    print(f"   {out_csv}")
    print(f"   {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=234,
                        help="평가할 이미지 수 (기본: 234 전체)")
    args = parser.parse_args()
    evaluate(max_samples=args.samples)
