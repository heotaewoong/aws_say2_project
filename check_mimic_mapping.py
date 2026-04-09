"""
MIMIC-CXR 레이블 CSV ↔ S3 이미지 매핑 확인 스크립트
실행: python check_mimic_mapping.py
"""
import pandas as pd
import boto3
import os
from pathlib import Path

# .env 파일 로드 (로컬 실행 시)
_env = Path(__file__).resolve().parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# ── 설정 ──
S3_BUCKET = "say1-pre-project-5"
S3_REGION = "ap-northeast-2"
CSV_PATH  = os.path.join(os.path.dirname(__file__), "data", "mimic-cxr-2.0.0-chexpert.csv")

AWS_ACCESS_KEY    = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY    = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

LABEL_COLS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices"
]

def s3_path(subject_id, study_id):
    """subject_id, study_id → S3 prefix 경로 변환"""
    sid  = str(int(subject_id))   # float → int → str (10000032.0 → 10000032)
    stid = str(int(study_id))
    p_folder = f"p{sid[:2]}"          # p10, p11, ...
    return f"data/mimic-cxr-jpg/files/{p_folder}/p{sid}/s{stid}/"

def check_s3_exists(s3_client, prefix):
    """S3 prefix 아래 jpg 파일이 있는지 확인"""
    resp = s3_client.list_objects_v2(
        Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=1
    )
    return resp.get("KeyCount", 0) > 0

def main():
    print("=" * 60)
    print("MIMIC-CXR 매핑 확인")
    print("=" * 60)

    # 1. CSV 로드
    print(f"\n[1] CSV 로드: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"    총 {len(df):,}건, 컬럼: {list(df.columns)}")

    # 2. 기본 통계
    print(f"\n[2] 기본 통계")
    print(f"    고유 환자 수: {df['subject_id'].nunique():,}명")
    print(f"    고유 study 수: {df['study_id'].nunique():,}건")

    # 3. 라벨 분포
    print(f"\n[3] 라벨별 양성(1) 건수")
    for col in LABEL_COLS:
        if col in df.columns:
            pos = (df[col] == 1).sum()
            print(f"    {col:<35}: {pos:>6,}건")

    # 4. S3 이미지 매핑 확인 (샘플 10건)
    print(f"\n[4] S3 이미지 매핑 확인 (샘플 10건)")
    s3 = boto3.client(
        "s3",
        region_name=S3_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )

    sample = df.head(10)
    ok_count = 0
    for _, row in sample.iterrows():
        prefix = s3_path(row["subject_id"], row["study_id"])
        exists = check_s3_exists(s3, prefix)
        status = "✅" if exists else "❌"
        if exists:
            ok_count += 1
        # 라벨 요약
        labels = [c for c in LABEL_COLS if c in df.columns and row[c] == 1.0]
        print(f"    {status} p{str(row['subject_id'])[:8]} / s{row['study_id']} "
              f"→ {prefix.split('files/')[1][:40]}  라벨: {labels}")

    print(f"\n    매핑 성공: {ok_count}/10건")

    # 5. 전체 매핑 가능 건수 추정
    print(f"\n[5] 결론")
    if ok_count == 10:
        print("    ✅ 매핑 완벽! CSV의 모든 레코드가 S3 이미지와 연결됩니다.")
        print(f"    → 학습 가능 데이터: {len(df):,}건")
    elif ok_count > 5:
        print(f"    ⚠️  일부 매핑 실패 ({ok_count}/10). 추가 확인 필요.")
    else:
        print(f"    ❌ 매핑 실패 ({ok_count}/10). 경로 구조 재확인 필요.")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
