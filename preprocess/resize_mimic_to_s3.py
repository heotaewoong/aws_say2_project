"""
MIMIC-CXR JPG 이미지를 448x448로 리사이즈해서 say2-2team-bucket에 저장
- 소스: say1-pre-project-5/data/mimic-cxr-jpg/files/
- 대상: say2-2team-bucket/data/mimic-cxr-448/
- mimic-cxr-2.0.0-chexpert.csv 기준 (subject_id, study_id)
- frontal 이미지만 처리 (PA/AP 포함 jpg)

실행: python preprocess/resize_mimic_to_s3.py
"""
import boto3
import pandas as pd
from PIL import Image
from io import BytesIO
import concurrent.futures
import time

# ── 설정 ──
SRC_BUCKET  = "say1-pre-project-5"
DST_BUCKET  = "say2-2team-bucket"
DST_PREFIX  = "data/mimic-cxr-448"
CSV_S3_KEY  = "mimic-cxr-2.0.0-chexpert.csv"
TARGET_SIZE = (448, 448)
MAX_WORKERS = 16
REGION      = "ap-northeast-2"

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION
)
s3 = session.client('s3')


def get_study_prefix(subject_id, study_id):
    sid = str(int(subject_id))
    stid = str(int(study_id))
    return f"data/mimic-cxr-jpg/files/p{sid[:2]}/p{sid}/s{stid}/"


def is_frontal(key):
    """PA/AP frontal 이미지 판별"""
    k = key.lower()
    return any(x in k for x in ['_pa', '_ap', 'frontal', 'pa.jpg', 'ap.jpg'])


def process_study(row):
    """study 1개의 frontal 이미지 처리"""
    results = []
    try:
        prefix = get_study_prefix(row['subject_id'], row['study_id'])
        resp = s3.list_objects_v2(Bucket=SRC_BUCKET, Prefix=prefix, MaxKeys=20)
        all_jpgs = [o['Key'] for o in resp.get('Contents', []) if o['Key'].endswith('.jpg')]

        # frontal 우선, 없으면 첫 번째 jpg
        frontal = [k for k in all_jpgs if is_frontal(k)]
        targets = frontal if frontal else (all_jpgs[:1] if all_jpgs else [])

        for src_key in targets:
            fname = src_key.split('/')[-1]
            sid = str(int(row['subject_id']))
            stid = str(int(row['study_id']))
            dst_key = f"{DST_PREFIX}/p{sid[:2]}/p{sid}/s{stid}/{fname}"

            # 이미 존재하면 스킵
            try:
                s3.head_object(Bucket=DST_BUCKET, Key=dst_key)
                results.append("skip")
                continue
            except Exception:
                pass

            # 다운로드 → 리사이즈 → 업로드
            obj = s3.get_object(Bucket=SRC_BUCKET, Key=src_key)
            img = Image.open(BytesIO(obj['Body'].read())).convert('RGB')
            img = img.resize(TARGET_SIZE, Image.LANCZOS)

            buf = BytesIO()
            img.save(buf, format='JPEG', quality=95)
            buf.seek(0)
            s3.put_object(Bucket=DST_BUCKET, Key=dst_key, Body=buf, ContentType='image/jpeg')
            results.append("ok")

    except Exception as e:
        results.append(f"err:{e}")

    return results


def main():
    print(f"CSV 로드 중: s3://{DST_BUCKET}/{CSV_S3_KEY}")
    obj = s3.get_object(Bucket=DST_BUCKET, Key=CSV_S3_KEY)
    df = pd.read_csv(BytesIO(obj['Body'].read()))
    df = df.drop_duplicates(subset=['subject_id', 'study_id']).reset_index(drop=True)
    total = len(df)
    print(f"처리할 study 수: {total:,}")
    print(f"소스: s3://{SRC_BUCKET}/")
    print(f"대상: s3://{DST_BUCKET}/{DST_PREFIX}/\n")

    ok = skip = err = 0
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_study, row): i
                   for i, row in df.iterrows()}

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            for r in future.result():
                if r == "ok":    ok += 1
                elif r == "skip": skip += 1
                else:             err += 1

            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (total - i - 1) / rate / 60
                print(f"[{i+1:>6}/{total}] ok={ok} skip={skip} err={err} "
                      f"| {rate:.1f} study/s | 남은시간 ~{remaining:.0f}분")

    elapsed = time.time() - start
    print(f"\n완료! ok={ok}, skip={skip}, err={err}, 총 {elapsed/60:.1f}분")
    print(f"저장 위치: s3://{DST_BUCKET}/{DST_PREFIX}/")


if __name__ == "__main__":
    main()
