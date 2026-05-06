"""
RAG 전체 파이프라인 검증 — MIMIC-IV 폐질환 환자
===============================================
팀 보고용: "실제 퇴원 진단 vs 파이프라인 Top 3" 비교

방법:
  1. MIMIC-IV discharge.csv 에서 폐질환 키워드 포함 환자 추출
  2. 해당 환자의 X-ray + 소견서 + 가상 Lab 주입
  3. 5단계 파이프라인 실행 → Top 3 질환 추출
  4. 실제 퇴원 진단(자유 텍스트)와 Top 3 비교 (키워드 매칭)

실행:
    cd aws_say2_project_vision
    python rag/valid/eval_rag_pipeline_mimic.py [--n-patients 10]

결과:
    rag/valid/rag_pipeline_mimic_results.json
"""
import os
import sys
import json
import argparse
import csv
import io
import re
import gzip
from datetime import datetime

import boto3

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

from rag_pipeline import RareLinkPipeline

s3 = boto3.client("s3", region_name="ap-northeast-2")

# 폐질환 관련 키워드 (discharge.csv 에서 필터링용)
LUNG_KEYWORDS = [
    "pneumonia", "pulmonary", "lung", "copd", "emphysema",
    "asthma", "fibrosis", "pleural effusion", "tuberculosis",
    "sarcoidosis", "bronchitis", "respiratory failure",
]

DEFAULT_LAB = {
    "WBC": 11.2, "HGB": 10.8, "LDH": 295, "CRP": 15.3,
    "SpO2": 92.0, "FEV1": 72.0,
}

# MIMIC-IV labevents itemid → 파이프라인 lab_results 키 매핑
LAB_ITEM_MAP = {
    51301: "WBC",          # White Blood Cells (×10³/μL)
    51300: "WBC",          # WBC Count
    51222: "HGB",          # Hemoglobin (g/dL)
    50954: "LDH",          # Lactate Dehydrogenase (U/L)
    50889: "CRP",          # C-Reactive Protein (mg/L)
    51652: "CRP",          # High-Sensitivity CRP
    50817: "SpO2",         # Oxygen Saturation (%) — blood gas
    51196: "D_Dimer",      # D-Dimer (ng/mL)
    50915: "D_Dimer",
    51265: "Platelet",     # Platelet Count (×10³/μL)
    50912: "Creatinine",   # Creatinine (mg/dL)
    50983: "Sodium",       # Sodium (mEq/L)
    50971: "Potassium",    # Potassium (mEq/L)
    51002: "Troponin_I",   # Troponin I (ng/mL)
    51003: "Troponin_T",   # Troponin T (ng/mL)
    50924: "Ferritin",     # Ferritin (ng/mL)
    50963: "NTproBNP",     # NTproBNP (pg/mL)
    51623: "Fibrinogen",   # Fibrinogen (mg/dL)
    51256: "Neutrophil",   # Neutrophils (%)
}

LAB_CACHE_PATH = "/tmp/mimic_lab_cache.json"


def fetch_real_lab_batch(subject_ids: list) -> dict:
    """
    MIMIC-IV labevents.csv.gz 스트리밍으로 여러 환자 실 Lab 수치 추출.

    Parameters
    ----------
    subject_ids : list[str]   대상 환자 ID 목록

    Returns
    -------
    dict  {subject_id: {lab_key: float, ...}}
    캐시 파일이 있으면 S3 재호출 없이 반환.
    """
    sid_set = set(str(s) for s in subject_ids)

    # 캐시 확인
    if os.path.exists(LAB_CACHE_PATH):
        with open(LAB_CACHE_PATH) as f:
            cached = json.load(f)
        if sid_set <= set(cached.keys()):
            print(f"  📂 Lab 캐시 사용 ({LAB_CACHE_PATH})")
            return {sid: cached[sid] for sid in sid_set}

    print(f"  ⬇️  labevents.csv.gz 스트리밍 시작 (대상: {len(sid_set)}명)...")
    print("     ※ 약 3~8분 소요될 수 있습니다.")

    # subject_id → {itemid → (charttime, valuenum)} 수집
    # 각 환자별 가장 최근(마지막) 값 유지
    collected = {sid: {} for sid in sid_set}
    target_items = set(LAB_ITEM_MAP.keys())
    rows_scanned = 0
    rows_matched = 0

    try:
        obj = s3.get_object(Bucket="say2-2team-bucket", Key="lab data/labevents.csv.gz")
        stream = obj["Body"]

        buf = b""
        gz_buf = io.BytesIO()

        # 청크 단위로 읽어 gzip 버퍼에 누적 후 파싱
        CHUNK = 8 * 1024 * 1024  # 8MB
        raw_data = io.BytesIO()

        # 전체 스트리밍 — gzip는 random access 불가라 전체 읽기 필요
        # 메모리 효율을 위해 스트리밍 gzip 디코더 사용
        class StreamingBody:
            def __init__(self, body): self.body = body
            def read(self, n=-1): return self.body.read(n)

        with gzip.GzipFile(fileobj=stream) as gz:
            text_stream = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
            reader = csv.DictReader(text_stream)

            for row in reader:
                rows_scanned += 1
                sid = row.get("subject_id", "").strip()
                if sid not in sid_set:
                    continue

                try:
                    itemid = int(row.get("itemid", 0))
                except (ValueError, TypeError):
                    continue

                if itemid not in target_items:
                    continue

                try:
                    val = float(row.get("valuenum", "") or "")
                except (ValueError, TypeError):
                    continue

                charttime = row.get("charttime", "") or ""
                lab_key = LAB_ITEM_MAP[itemid]

                # 더 최근 값으로 갱신
                prev = collected[sid].get(lab_key)
                if prev is None or charttime >= prev[0]:
                    collected[sid][lab_key] = (charttime, val)

                rows_matched += 1
                if rows_scanned % 5_000_000 == 0:
                    found_count = sum(1 for v in collected.values() if v)
                    print(f"     {rows_scanned:,}행 스캔 / 매칭 {rows_matched}건 / 환자 발견 {found_count}명")

    except Exception as e:
        print(f"  ❌ labevents 스트리밍 오류: {e}")

    print(f"  ✅ 스캔 완료: {rows_scanned:,}행, 매칭 {rows_matched}건")

    # charttime 제거하고 값만 추출
    result = {}
    for sid in sid_set:
        result[sid] = {k: round(v, 2) for k, (_, v) in collected[sid].items()}
        # 없는 항목은 DEFAULT_LAB 값으로 보완
        if "WBC"  not in result[sid]: result[sid]["WBC"]  = DEFAULT_LAB["WBC"]
        if "HGB"  not in result[sid]: result[sid]["HGB"]  = DEFAULT_LAB["HGB"]
        if "LDH"  not in result[sid]: result[sid]["LDH"]  = DEFAULT_LAB["LDH"]
        if "CRP"  not in result[sid]: result[sid]["CRP"]  = DEFAULT_LAB["CRP"]
        if "SpO2" not in result[sid]: result[sid]["SpO2"] = DEFAULT_LAB["SpO2"]

    # 캐시 저장
    try:
        existing = {}
        if os.path.exists(LAB_CACHE_PATH):
            with open(LAB_CACHE_PATH) as f:
                existing = json.load(f)
        existing.update(result)
        with open(LAB_CACHE_PATH, "w") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"  💾 Lab 캐시 저장: {LAB_CACHE_PATH}")
    except Exception as e:
        print(f"  ⚠️ 캐시 저장 실패: {e}")

    return result


def find_xray_for_patient(subject_id: str):
    """MIMIC-CXR S3에서 해당 환자의 첫 frontal JPG 찾기"""
    prefix = f"data/mimic-cxr-jpg/files/p{str(subject_id)[:2]}/p{subject_id}/"
    try:
        resp = s3.list_objects_v2(
            Bucket='say1-pre-project-5', Prefix=prefix, Delimiter='/'
        )
        studies = [p['Prefix'] for p in resp.get('CommonPrefixes', [])]
        if not studies:
            return None
        resp2 = s3.list_objects_v2(Bucket='say1-pre-project-5', Prefix=studies[0])
        for obj in resp2.get('Contents', []):
            if obj['Key'].endswith('.jpg') and obj['Size'] > 10000:
                return obj['Key']
    except Exception:
        pass
    return None


def extract_diagnosis(text: str) -> str:
    """discharge.txt 에서 Discharge Diagnosis 추출"""
    patterns = [
        r'(?:Discharge Diagnosis|DISCHARGE DIAGNOSIS)[:\s]*(.*?)(?=\n[A-Z][a-zA-Z ]+:|\Z)',
        r'(?:Final Diagnosis|PRIMARY DIAGNOSIS)[:\s]*(.*?)(?=\n[A-Z][a-zA-Z ]+:|\Z)',
        r'(?:Chief Complaint)[:\s]*(.*?)(?=\n[A-Z][a-zA-Z ]+:|\Z)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            lines = [l.strip().lstrip('*•-1234567890. ') for l in raw.splitlines() if l.strip()]
            if lines:
                return lines[0][:200]
    return text[:200]


def is_lung_related(diagnosis: str) -> bool:
    d = diagnosis.lower()
    return any(kw in d for kw in LUNG_KEYWORDS)


def find_lung_patients(n_patients: int, max_scan_mb: int = 300):
    """discharge.csv 스트리밍으로 폐질환 환자 N명 찾기
    - 반드시 'Discharge Diagnosis' 섹션 안에 폐질환 키워드 있는 환자만
    """
    print(f"🔍 MIMIC-IV discharge.csv 에서 폐질환 환자 {n_patients}명 검색...")

    obj = s3.get_object(
        Bucket='say1-pre-project-7',
        Key='mimic-iv-note/2.2/note/discharge.csv',
    )

    limit_bytes = max_scan_mb * 1024 * 1024
    raw_bytes = obj['Body'].read(limit_bytes)
    raw_text = raw_bytes.decode('utf-8', errors='replace')

    last_newline = raw_text.rfind('\n')
    if last_newline > 0:
        raw_text = raw_text[:last_newline]

    reader = csv.DictReader(io.StringIO(raw_text))

    patients = []
    seen = set()
    scanned = 0

    for row in reader:
        scanned += 1
        if len(patients) >= n_patients:
            break

        sid = str(row.get('subject_id', '')).strip()
        if not sid or sid in seen:
            continue

        text = (row.get('text', '') or '')[:8000]

        # 반드시 Discharge Diagnosis 섹션 추출
        diag_match = re.search(
            r'Discharge Diagnosis[:\s]*\n?(.{20,1500}?)(?=\n[A-Z][a-zA-Z ]+:|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        if not diag_match:
            continue

        diagnosis_section = diag_match.group(1).strip()

        # Discharge Diagnosis 섹션 안에 폐질환 키워드 있어야 함
        if not is_lung_related(diagnosis_section):
            continue

        # 진단명 첫 줄 추출
        diag_lines = [l.strip().lstrip('*•-1234567890. ')
                      for l in diagnosis_section.splitlines() if l.strip()]
        if not diag_lines:
            continue
        # 폐질환 키워드가 있는 첫 줄 선택
        primary_diag = next(
            (l for l in diag_lines if is_lung_related(l)),
            diag_lines[0]
        )[:200]

        # X-ray 존재 여부 확인
        xray_key = find_xray_for_patient(sid)
        if not xray_key:
            continue

        patients.append({
            'subject_id': sid,
            'diagnosis':  primary_diag,
            'diag_section': diagnosis_section[:500],
            'text':       text[:3000],
            'xray_key':   xray_key,
        })
        seen.add(sid)
        print(f"  [{len(patients)}/{n_patients}] subject_id={sid}  진단: {primary_diag[:70]}")

    print(f"  총 스캔한 note: {scanned}개")
    return patients


def evaluate_top3_hit(actual_diagnosis: str, top3_names: list) -> dict:
    """실제 진단과 Top 3 질환명 키워드 매칭 (개선된 버전)

    Parameters
    ----------
    actual_diagnosis : str     실제 진단 텍스트
    top3_names       : list    [disease_name, disease_name, disease_name]
    """
    actual_lower = actual_diagnosis.lower()

    # 의학적 동의어 매핑 (한↔영, 축약)
    synonyms = {
        "pneumonia": ["pneumonia", "폐렴"],
        "aspiration pneumonia": ["aspiration pneumonia", "흡인성 폐렴", "흡인성폐렴"],
        "pleural effusion": ["pleural effusion", "흉수", "흉막삼출"],
        "pulmonary embolism": ["pulmonary embolism", "폐색전", "폐색전증"],
        "lung cancer": ["lung cancer", "lung mets", "폐암", "전이성 종양", "폐 종양"],
        "metastasis": ["mets", "metastasis", "전이", "전이성"],
        "copd": ["copd", "만성폐쇄성", "폐기종"],
        "pneumothorax": ["pneumothorax", "기흉"],
        "tuberculosis": ["tuberculosis", "tb", "결핵"],
        "sarcoidosis": ["sarcoidosis", "사르코이드", "사르코이도시스"],
        "pulmonary fibrosis": ["pulmonary fibrosis", "폐섬유", "ipf"],
        "lam": ["lymphangioleiomyomatosis", "lam", "림프관근육종"],
        "hospital-acquired pneumonia": ["hospital-acquired", "hap", "vap", "병원획득"],
        "community-acquired pneumonia": ["community-acquired", "cap", "지역사회획득"],
    }

    # 실제 진단을 표준 질환군으로 매핑
    actual_categories = set()
    for category, keywords in synonyms.items():
        for kw in keywords:
            if kw in actual_lower:
                actual_categories.add(category)
                break

    # Top 3 각 질환명도 표준 질환군으로 매핑
    hit_at_1 = False
    hit_at_3 = False
    matched_rank = None

    for rank, name in enumerate(top3_names[:3], 1):
        name_lower = name.lower()
        pred_categories = set()
        for category, keywords in synonyms.items():
            for kw in keywords:
                if kw in name_lower:
                    pred_categories.add(category)
                    break

        # 실제 카테고리와 예측 카테고리가 겹치는가
        if actual_categories & pred_categories:
            if matched_rank is None:
                matched_rank = rank
            if rank == 1:
                hit_at_1 = True
            if rank <= 3:
                hit_at_3 = True

    return {
        "matched_rank": matched_rank,
        "hit@1": hit_at_1,
        "hit@3": hit_at_3,
        "actual_categories": sorted(actual_categories),
    }


def run_evaluation(n_patients: int):
    print("=" * 60)
    print("RAG 전체 파이프라인 검증 — MIMIC-IV 폐질환 환자")
    print(f"대상: {n_patients}명")
    print("=" * 60)

    # ── 1. 환자 찾기 ──────────────────────────────────────────────
    patients = find_lung_patients(n_patients, max_scan_mb=150)
    if not patients:
        print("❌ 폐질환 환자 없음")
        return

    print(f"\n✅ {len(patients)}명 확보 완료\n")

    # ── 1.5. 실 Lab 데이터 배치 추출 ─────────────────────────────
    print("🧪 MIMIC-IV labevents 실 Lab 데이터 추출 중...")
    all_sids = [p["subject_id"] for p in patients]
    real_lab_map = fetch_real_lab_batch(all_sids)

    print("\n  [실 Lab 데이터 요약]")
    for sid in all_sids:
        lab = real_lab_map.get(sid, {})
        keys = list(lab.keys())
        print(f"  {sid}: {len(keys)}개 항목 — {keys}")

    # ── 2. 파이프라인 초기화 (1회만) ─────────────────────────────
    print("\n🔧 파이프라인 초기화 중...")
    pipeline = RareLinkPipeline(
        vision_model_path='model/chexnet_unet_crop_best.pth',
        orphanet_csv_path='data/orphadata_weighted.csv',
    )

    # ── 3. 각 환자 파이프라인 실행 ───────────────────────────────
    results = []
    out_dir = '/tmp/rag_eval_mimic'
    os.makedirs(out_dir, exist_ok=True)

    for i, p in enumerate(patients, 1):
        sid = p['subject_id']
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(patients)}] 환자 {sid}")
        print(f"실제 진단: {p['diagnosis'][:80]}")
        print('─' * 60)

        # X-ray 다운로드
        xray_local = f"{out_dir}/patient_{sid}_xray.jpg"
        try:
            s3.download_file('say1-pre-project-5', p['xray_key'], xray_local)
        except Exception as e:
            print(f"  ⚠️ X-ray 다운로드 실패: {e}")
            continue

        # 파이프라인 실행
        try:
            report = pipeline.run(
                patient_info={
                    "name": "익명",
                    "age": "",
                    "sex": "",
                    "visit_date": "",
                    "visit_type": "입원",
                    "chief_complaint": p['diagnosis'][:100],
                    "allergy": "없음",
                },
                xray_path=xray_local,
                symptom_text=p['text'][:1500],
                negative_text="",
                lab_results=DEFAULT_LAB,
            )
        except Exception as e:
            print(f"  ❌ 파이프라인 오류: {e}")
            continue
        finally:
            if os.path.exists(xray_local):
                os.remove(xray_local)

        # Top 3 질환명 추출 — 신 포맷: general_diagnosis[] + rare_diagnosis[]
        top3_names = []
        if isinstance(report, dict):
            for d in (report.get("general_diagnosis") or [])[:3]:
                name = d.get("disease_name", "")
                if name:
                    top3_names.append(name)
            for d in (report.get("rare_diagnosis") or [])[:3]:
                name = d.get("disease_name", "")
                if name and name not in top3_names:
                    top3_names.append(name)
        top3_names = top3_names[:3]

        # AI 요약 텍스트 (표시용)
        top3_text = ""
        if isinstance(report, dict) and "clinical_notes" in report:
            cn = report["clinical_notes"]
            top3_text = (
                cn.get("summary", "") + " " +
                cn.get("differential_note", "")
            )

        # 실제 진단 vs AI Top 3 매칭 (개선된 카테고리 매칭)
        hit_info = evaluate_top3_hit(p['diagnosis'], top3_names)

        result = {
            "subject_id": sid,
            "actual_diagnosis": p['diagnosis'],
            "ai_top3_names": top3_names,
            "actual_categories": hit_info.get('actual_categories', []),
            "ai_summary": top3_text[:500],
            "matched_rank": hit_info['matched_rank'],
            "hit@1": hit_info['hit@1'],
            "hit@3": hit_info['hit@3'],
            "full_report": report if isinstance(report, dict) else {},
        }
        results.append(result)

        print(f"  AI Top 3: {top3_names}")
        print(f"  매칭 결과: hit@1={hit_info['hit@1']}, hit@3={hit_info['hit@3']}, rank={hit_info['matched_rank']}")

    # ── 4. 종합 결과 ─────────────────────────────────────────────
    total = len(results)
    hit_1 = sum(r['hit@1'] for r in results)
    hit_3 = sum(r['hit@3'] for r in results)

    print("\n" + "=" * 60)
    print(f"검증 결과 요약  ({total}명)")
    print("=" * 60)
    print(f"  Hit@1 (카테고리 매칭): {hit_1}/{total} ({hit_1/total*100:.1f}%)")
    print(f"  Hit@3 (카테고리 매칭): {hit_3}/{total} ({hit_3/total*100:.1f}%)")
    print("=" * 60)

    out_json = "rag/valid/rag_pipeline_mimic_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n_patients": total,
            "hit_at_1": hit_1,
            "hit_at_1_rate": round(hit_1 / max(total, 1), 4),
            "hit_at_3": hit_3,
            "hit_at_3_rate": round(hit_3 / max(total, 1), 4),
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n📄 결과 저장: {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-patients", type=int, default=10)
    args = parser.parse_args()
    run_evaluation(args.n_patients)
