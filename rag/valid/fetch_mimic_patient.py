"""
MIMIC 데이터에서 여러 환자의 X-ray + 소견서를 뽑아서
RAG 파이프라인 테스트용 데이터를 준비하는 스크립트

사용법:
    cd aws_say2_project_vision
    python rag/valid/fetch_mimic_patient.py

결과:
    /tmp/mimic_test/
    ├── patient_10000032/
    │   ├── xray.jpg
    │   ├── discharge.txt
    │   └── lab_results.json
    ├── patient_10000764/
    │   └── ...
    └── ...
"""
import os
import sys
import json
import csv
import io
import boto3

# rag/valid/ → 프로젝트 루트로 경로 설정
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT)

os.environ.setdefault('AWS_DEFAULT_REGION', 'ap-northeast-2')
# AWS 키는 환경변수로 설정하세요:
# export AWS_ACCESS_KEY_ID="..."
# export AWS_SECRET_ACCESS_KEY="..."

s3 = boto3.client('s3', region_name='ap-northeast-2')
OUT_DIR = '/tmp/mimic_test'

# ── 테스트할 환자 목록 ────────────────────────────────────────────
# MIMIC-CXR에 frontal X-ray가 있는 환자들
# subject_id, study_id, image_filename 순서
PATIENTS = [
    {
        'subject_id': '10000032',
        'study_id': 's50414267',
        'img_file': '174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.jpg',
        'prefix': 'p10',
    },
    {
        'subject_id': '10000764',
        'study_id': None,  # 자동 탐색
        'img_file': None,
        'prefix': 'p10',
    },
    {
        'subject_id': '10000898',
        'study_id': None,
        'img_file': None,
        'prefix': 'p10',
    },
]

# 기본 Lab 수치 (MIMIC labevents 2.5GB라 스트리밍 생략)
DEFAULT_LAB = {'WBC': 10.5, 'HGB': 11.2, 'SpO2': 94.0, 'CRP': 8.5}


def find_first_image(subject_id, prefix):
    """S3에서 해당 환자의 첫 번째 frontal 이미지 경로 찾기"""
    base = f'data/mimic-cxr-jpg/files/{prefix}/p{subject_id}/'
    try:
        # study 폴더 목록
        resp = s3.list_objects_v2(Bucket='say1-pre-project-5', Prefix=base, Delimiter='/')
        studies = [p['Prefix'] for p in resp.get('CommonPrefixes', [])]
        if not studies:
            return None

        # 첫 번째 study에서 이미지 찾기
        resp2 = s3.list_objects_v2(Bucket='say1-pre-project-5', Prefix=studies[0])
        for obj in resp2.get('Contents', []):
            key = obj['Key']
            if key.endswith('.jpg') and obj['Size'] > 10000:  # 10KB 이상 (빈 파일 제외)
                return key
    except Exception as e:
        print(f'  ⚠️ S3 탐색 실패: {e}')
    return None


def fetch_discharge(subject_id, max_read_mb=20):
    """discharge.csv에서 해당 환자 소견서 추출"""
    try:
        obj = s3.get_object(Bucket='say1-pre-project-7', Key='mimic-iv-note/2.2/note/discharge.csv')
        raw_bytes = obj['Body'].read(max_read_mb * 1024 * 1024)
        raw_text = raw_bytes.decode('utf-8', errors='replace')

        reader = csv.DictReader(io.StringIO(raw_text))
        for row in reader:
            if row.get('subject_id', '').strip() == subject_id:
                return row.get('text', '').strip()[:3000]
    except Exception as e:
        print(f'  ⚠️ 소견서 추출 실패: {e}')
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    print('='*60)
    print(f'MIMIC 환자 {len(PATIENTS)}명 데이터 준비')
    print('='*60)

    for i, p in enumerate(PATIENTS, 1):
        sid = p['subject_id']
        patient_dir = os.path.join(OUT_DIR, f'patient_{sid}')
        os.makedirs(patient_dir, exist_ok=True)

        print(f'\n--- [{i}/{len(PATIENTS)}] 환자 {sid} ---')

        # 1. X-ray 이미지
        if p['img_file'] and p['study_id']:
            img_key = f"data/mimic-cxr-jpg/files/{p['prefix']}/p{sid}/{p['study_id']}/{p['img_file']}"
        else:
            print(f'  [X-ray] S3에서 이미지 탐색 중...')
            img_key = find_first_image(sid, p['prefix'])

        xray_path = os.path.join(patient_dir, 'xray.jpg')
        if img_key:
            try:
                s3.download_file('say1-pre-project-5', img_key, xray_path)
                print(f'  ✅ X-ray: {os.path.getsize(xray_path)//1024}KB')
            except Exception as e:
                print(f'  ❌ X-ray 다운로드 실패: {e}')
                xray_path = None
        else:
            print(f'  ❌ X-ray 이미지 없음')
            xray_path = None

        # 2. 소견서
        print(f'  [소견서] discharge.csv 검색 중...')
        discharge = fetch_discharge(sid)
        discharge_path = os.path.join(patient_dir, 'discharge.txt')
        if discharge:
            with open(discharge_path, 'w') as f:
                f.write(discharge)
            print(f'  ✅ 소견서: {len(discharge)}자')
        else:
            with open(discharge_path, 'w') as f:
                f.write(f'Patient {sid}. Chest X-ray performed. Respiratory symptoms noted.')
            print(f'  ⚠️ 소견서 없음 → 기본 텍스트 사용')

        # 3. Lab
        lab_path = os.path.join(patient_dir, 'lab_results.json')
        with open(lab_path, 'w') as f:
            json.dump(DEFAULT_LAB, f, indent=2)

        results.append({
            'subject_id': sid,
            'xray_path': xray_path,
            'discharge_path': discharge_path,
            'lab_path': lab_path,
        })

    # 결과 요약
    print(f'\n{"="*60}')
    print(f'준비 완료! {len(results)}명 환자 데이터:')
    for r in results:
        status = '✅' if r['xray_path'] else '❌'
        print(f'  {status} patient_{r["subject_id"]}/')
    print(f'\n다음 명령으로 파이프라인 실행:')
    print(f'  python rag/valid/run_rag_test.py')


if __name__ == '__main__':
    main()
