"""
RAG 파이프라인 다중 환자 테스트 스크립트
- /tmp/mimic_test/patient_XXXXX/ 폴더들을 자동 탐색
- 각 환자별로 파이프라인 실행 → 리포트 저장 → PMID 검증

실행:
    cd aws_say2_project_vision
    python rag/valid/run_rag_test.py

결과:
    /tmp/mimic_test/patient_XXXXX/report.md  ← 각 환자별 리포트
    /tmp/mimic_test/validation_summary.txt   ← 전체 요약
"""
import sys, os, json, glob

# rag/valid/ → 프로젝트 루트로 경로 설정
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

os.environ.setdefault('AWS_DEFAULT_REGION', 'ap-northeast-2')

# .env 파일 자동 로드
_env_path = os.path.join(ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from rag_pipeline import RareLinkPipeline
from rag.ragas_eval import verify_pmids

OUT_DIR = '/tmp/mimic_test'


def run_all_patients():
    # 환자 폴더 자동 탐색
    patient_dirs = sorted(glob.glob(os.path.join(OUT_DIR, 'patient_*')))

    if not patient_dirs:
        # 단일 환자 모드 (이전 호환)
        xray = os.path.join(OUT_DIR, 'xray.jpg')
        if os.path.exists(xray):
            patient_dirs = [OUT_DIR]
        else:
            print('❌ 테스트 데이터 없음. 먼저 실행하세요:')
            print('   python rag/valid/fetch_mimic_patient.py')
            return

    print('='*60)
    print(f'RAG 파이프라인 검증 — {len(patient_dirs)}명 환자')
    print('='*60)

    # 파이프라인 초기화 (1회만)
    pipeline = RareLinkPipeline(
        vision_model_path='model/chexnet_unet_crop_best.pth',
        orphanet_csv_path='data/orphadata_weighted.csv',
    )

    results = []

    for i, pdir in enumerate(patient_dirs, 1):
        patient_name = os.path.basename(pdir)
        print(f'\n{"="*60}')
        print(f'[{i}/{len(patient_dirs)}] {patient_name}')
        print('='*60)

        # 입력 파일 확인
        xray_path = os.path.join(pdir, 'xray.jpg')
        discharge_path = os.path.join(pdir, 'discharge.txt')
        lab_path = os.path.join(pdir, 'lab_results.json')

        if not os.path.exists(xray_path):
            print(f'  ❌ X-ray 없음 — 건너뜀')
            results.append({'patient': patient_name, 'status': 'SKIP', 'reason': 'no xray'})
            continue

        # 소견서 읽기
        if os.path.exists(discharge_path):
            with open(discharge_path) as f:
                symptom_text = f.read()[:1000]
        else:
            symptom_text = '호흡곤란과 흉통이 있는 환자입니다.'

        # Lab 읽기
        if os.path.exists(lab_path):
            with open(lab_path) as f:
                lab_results = json.load(f)
        else:
            lab_results = {'WBC': 10.0, 'SpO2': 93.0}

        # 파이프라인 실행
        try:
            report = pipeline.run(
                patient_info={
                    "name": "익명",
                    "age": "",
                    "sex": "",
                    "visit_date": "",
                    "visit_type": "외래",
                    "chief_complaint": symptom_text[:100],
                    "allergy": "없음",
                },
                xray_path=xray_path,
                symptom_text=symptom_text,
                negative_text="",
                lab_results=lab_results,
            )

            # 리포트 저장
            report_path = os.path.join(pdir, 'report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f'\n  📄 리포트 저장: {report_path}')

            # PMID 검증 (clinical_notes의 rag_evidence에서 PMID 추출)
            print(f'\n  === PMID 검증 ===')
            report_text = json.dumps(report, ensure_ascii=False)
            pmid_result = verify_pmids(report_text)

            results.append({
                'patient': patient_name,
                'status': 'PASS',
                'report_path': report_path,
                'pmid_total': pmid_result['total'],
                'pmid_valid': pmid_result['valid'],
                'pmid_rate': pmid_result['rate'],
            })

        except Exception as e:
            print(f'  ❌ 오류: {e}')
            results.append({'patient': patient_name, 'status': 'ERROR', 'reason': str(e)})

    # ── 전체 요약 ─────────────────────────────────────────────────
    print(f'\n\n{"="*60}')
    print('검증 결과 요약')
    print('='*60)
    print(f'{"환자":<20} {"상태":<8} {"PMID":<15} {"리포트"}')
    print('-'*60)

    pass_count = 0
    for r in results:
        if r['status'] == 'PASS':
            pass_count += 1
            pmid_str = f"{r['pmid_valid']}/{r['pmid_total']}" if r['pmid_total'] > 0 else "인용없음"
            print(f"  ✅ {r['patient']:<16} PASS     {pmid_str:<13} {r['report_path']}")
        elif r['status'] == 'SKIP':
            print(f"  ⏭️  {r['patient']:<16} SKIP     {r['reason']}")
        else:
            print(f"  ❌ {r['patient']:<16} ERROR    {r['reason'][:30]}")

    print(f'\n결과: {pass_count}/{len(results)} 환자 성공')

    # 요약 파일 저장
    summary_path = os.path.join(OUT_DIR, 'validation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f'RAG 파이프라인 검증 결과\n')
        f.write(f'날짜: {__import__("datetime").datetime.now().isoformat()}\n')
        f.write(f'환자 수: {len(results)}\n')
        f.write(f'성공: {pass_count}/{len(results)}\n\n')
        for r in results:
            f.write(f'{r}\n')
    print(f'\n요약 저장: {summary_path}')


if __name__ == '__main__':
    run_all_patients()
