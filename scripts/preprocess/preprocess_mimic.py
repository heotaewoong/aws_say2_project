"""
MIMIC-IV 전처리 스크립트
========================
역할: MIMIC-IV 원본 데이터에서 폐질환 환자의 혈액검사 수치를
      lab_genomic_agent.py가 바로 사용할 수 있는 형태로 변환

입력:
  - data/mimic-iv/hosp/labevents.csv.gz      (혈액검사 원본 - 2.4GB)
  - data/mimic-iv/hosp/d_labitems.csv.gz     (검사명 사전)
  - data/mimic-iv/hosp/diagnoses_icd.csv.gz  (진단 코드)
  - data/mimic-iv/hosp/d_icd_diagnoses.csv.gz (코드→병명)

출력:
  - data/lung_patients.csv    : 폐질환 환자 목록 (subject_id)
  - data/lung_lab_data.csv    : 폐질환 환자 혈액검사 결과 (전처리 완료)
  - data/lab_for_agent.json   : lab_genomic_agent.py 바로 입력 가능한 형태

실행:
  python preprocess_mimic.py

초보자용 설명:
  Step 1: 폐질환 환자만 추출 (36만명 → 수천~수만명)
  Step 2: 그 환자들의 혈액검사 수치만 꺼냄 (수억 건 → 수십만 건)
  Step 3: WBC, CRP 등 우리가 필요한 검사 항목만 필터링
  Step 4: lab_genomic_agent.py 입력 형태로 변환해서 저장
"""

import pandas as pd
import numpy as np
import json
import os
import gzip

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
MIMIC_HOSP = os.path.join(DATA_DIR, "mimic-iv", "hosp")

PATHS = {
    "labevents"      : os.path.join(MIMIC_HOSP, "labevents.csv.gz"),
    "d_labitems"     : os.path.join(MIMIC_HOSP, "d_labitems.csv.gz"),
    "diagnoses_icd"  : os.path.join(MIMIC_HOSP, "diagnoses_icd.csv.gz"),
    "d_icd_diagnoses": os.path.join(MIMIC_HOSP, "d_icd_diagnoses.csv.gz"),
}

OUT_LUNG_PATIENTS  = os.path.join(DATA_DIR, "lung_patients.csv")
OUT_LUNG_LAB       = os.path.join(DATA_DIR, "lung_lab_data.csv")
OUT_LAB_FOR_AGENT  = os.path.join(DATA_DIR, "lab_for_agent.json")

# ─────────────────────────────────────────────
# lab_genomic_agent.py 에서 사용하는 검사 항목
# MIMIC-IV d_labitems의 label 컬럼과 매핑
# ─────────────────────────────────────────────
AGENT_LAB_MAP = {
    # agent 키         : MIMIC-IV label 키워드 (d_labitems.label에서 검색)
    "WBC Count"        : ["White Blood Cells", "WBC"],
    "Platelet Count"   : ["Platelet Count", "Platelets"],
    "CRP"              : ["C-Reactive Protein", "CRP"],
    "Oxygen Saturation": ["Oxygen Saturation", "SaO2"],
}

# 폐질환 ICD-10 코드 (J = 호흡기계 질환)
# ICD-9 코드도 포함 (MIMIC-IV는 ICD-9/10 혼재)
LUNG_ICD10_PREFIX = ["J0", "J1", "J2", "J3", "J4", "J6", "J7", "J8", "J9"]  # J로 시작하는 호흡기계
LUNG_ICD9_CODES   = ["480", "481", "482", "483", "484", "485", "486",  # 폐렴
                      "491", "492", "493", "494", "496",                 # COPD, 천식
                      "510", "511", "512", "513", "514", "515",          # 흉막, 폐섬유화
                      "516", "517", "518", "519"]                        # 기타 폐질환


def step1_extract_lung_patients():
    """
    Step 1: 폐질환 환자 ID 추출
    diagnoses_icd.csv에서 폐질환 ICD 코드를 가진 환자 subject_id 목록 추출
    """
    print("\n" + "="*60)
    print("Step 1: 폐질환 환자 필터링")
    print("="*60)

    print("  📂 diagnoses_icd.csv.gz 로드 중...")
    diag = pd.read_csv(PATHS["diagnoses_icd"])
    print(f"  ✅ 총 진단 기록: {len(diag):,}건")

    # ICD-10: J 코드 (호흡기계)
    mask_icd10 = (diag["icd_version"] == 10) & (
        diag["icd_code"].str.startswith(tuple(LUNG_ICD10_PREFIX), na=False)
    )
    # ICD-9: 폐질환 코드
    mask_icd9 = (diag["icd_version"] == 9) & (
        diag["icd_code"].str.startswith(tuple(LUNG_ICD9_CODES), na=False)
    )

    lung_diag = diag[mask_icd10 | mask_icd9]
    lung_patients = lung_diag["subject_id"].unique()

    print(f"  🫁 폐질환 환자 수: {len(lung_patients):,}명")
    print(f"     (전체 {diag['subject_id'].nunique():,}명 중 "
          f"{len(lung_patients)/diag['subject_id'].nunique()*100:.1f}%)")

    # 저장
    df_patients = pd.DataFrame({"subject_id": lung_patients})
    df_patients.to_csv(OUT_LUNG_PATIENTS, index=False)
    print(f"  💾 저장: {OUT_LUNG_PATIENTS}")

    return set(lung_patients)


def step2_find_lab_itemids():
    """
    Step 2: 필요한 혈액검사 항목의 itemid 찾기
    d_labitems에서 WBC, CRP 등의 itemid 매핑
    """
    print("\n" + "="*60)
    print("Step 2: 혈액검사 항목 itemid 매핑")
    print("="*60)

    d_lab = pd.read_csv(PATHS["d_labitems"])
    print(f"  📋 총 검사 항목: {len(d_lab):,}개")

    itemid_map = {}  # {agent_key: [itemid1, itemid2, ...]}

    for agent_key, keywords in AGENT_LAB_MAP.items():
        matched_ids = []
        for kw in keywords:
            mask = d_lab["label"].str.contains(kw, case=False, na=False)
            ids = d_lab.loc[mask, "itemid"].tolist()
            matched_ids.extend(ids)
            if ids:
                labels = d_lab.loc[mask, "label"].tolist()
                print(f"  ✅ {agent_key:<20} ← {labels[:2]} (itemid: {ids[:2]})")

        itemid_map[agent_key] = list(set(matched_ids))

    all_itemids = set()
    for ids in itemid_map.values():
        all_itemids.update(ids)

    print(f"\n  📌 매핑된 itemid 총 {len(all_itemids)}개")
    return itemid_map, all_itemids


def step3_extract_lab_data(lung_patient_ids, all_itemids):
    """
    Step 3: 폐질환 환자 + 필요한 검사 항목만 labevents에서 추출
    2.4GB 파일을 청크(chunk) 단위로 읽어서 메모리 절약
    """
    print("\n" + "="*60)
    print("Step 3: 혈액검사 데이터 추출 (청크 단위 처리)")
    print("="*60)
    print("  ⏳ labevents.csv.gz 처리 중... (2.4GB, 시간이 걸릴 수 있음)")

    CHUNK_SIZE = 500_000  # 한 번에 50만 행씩 처리
    results = []
    chunk_num = 0
    total_rows = 0
    matched_rows = 0

    for chunk in pd.read_csv(PATHS["labevents"], chunksize=CHUNK_SIZE):
        chunk_num += 1
        total_rows += len(chunk)

        # 필터링: 폐질환 환자 AND 필요한 검사 항목
        filtered = chunk[
            chunk["subject_id"].isin(lung_patient_ids) &
            chunk["itemid"].isin(all_itemids)
        ]

        if len(filtered) > 0:
            # 필요한 컬럼만 유지
            keep_cols = ["subject_id", "hadm_id", "itemid",
                         "charttime", "valuenum", "valueuom",
                         "ref_range_lower", "ref_range_upper", "flag"]
            filtered = filtered[[c for c in keep_cols if c in filtered.columns]]
            results.append(filtered)
            matched_rows += len(filtered)

        if chunk_num % 10 == 0:
            print(f"  처리: {total_rows:,}행 검토 → {matched_rows:,}행 추출 "
                  f"(청크 {chunk_num})")

    print(f"\n  ✅ 처리 완료: 총 {total_rows:,}행 중 {matched_rows:,}행 추출")

    if results:
        df_lab = pd.concat(results, ignore_index=True)
        return df_lab
    else:
        print("  ⚠️ 추출된 데이터 없음")
        return pd.DataFrame()


def step4_merge_and_save(df_lab, itemid_map):
    """
    Step 4: itemid → 검사명 변환 후 저장
    그리고 lab_genomic_agent.py 입력 형태로도 변환
    """
    print("\n" + "="*60)
    print("Step 4: 검사명 매핑 + 저장")
    print("="*60)

    if df_lab.empty:
        print("  ❌ 데이터 없음, 종료")
        return

    # itemid → agent_key 역매핑
    id_to_key = {}
    for key, ids in itemid_map.items():
        for iid in ids:
            id_to_key[iid] = key

    df_lab["lab_name"] = df_lab["itemid"].map(id_to_key)

    # 최종 컬럼 정리
    df_out = df_lab[["subject_id", "hadm_id", "lab_name",
                     "valuenum", "valueuom", "flag",
                     "ref_range_lower", "ref_range_upper", "charttime"]].copy()
    df_out = df_out.dropna(subset=["valuenum"])  # 수치 없는 행 제거

    df_out.to_csv(OUT_LUNG_LAB, index=False)
    print(f"  💾 저장: {OUT_LUNG_LAB} ({len(df_out):,}행)")

    # lab_genomic_agent.py 입력 형태로 변환
    # 환자별 최신 수치 1개씩 (hadm_id 기준 최신)
    print("\n  🔄 lab_genomic_agent.py 입력 형태로 변환 중...")
    df_latest = (df_out.sort_values("charttime", ascending=False)
                       .groupby(["subject_id", "lab_name"])
                       .first()
                       .reset_index())

    # 환자별 딕셔너리 생성
    agent_input = {}
    for subject_id, group in df_latest.groupby("subject_id"):
        lab_dict = {}
        for _, row in group.iterrows():
            if pd.notna(row["valuenum"]):
                lab_dict[row["lab_name"]] = float(row["valuenum"])
        if lab_dict:
            agent_input[int(subject_id)] = lab_dict

    with open(OUT_LAB_FOR_AGENT, "w", encoding="utf-8") as f:
        json.dump(agent_input, f, ensure_ascii=False, indent=2)

    print(f"  💾 저장: {OUT_LAB_FOR_AGENT} ({len(agent_input):,}명)")
    print("\n  📋 샘플 출력 (첫 3명):")
    for sid, labs in list(agent_input.items())[:3]:
        print(f"    환자 {sid}: {labs}")


def run_demo():
    """
    lab_for_agent.json이 있으면 lab_genomic_agent.py로 바로 테스트
    """
    print("\n" + "="*60)
    print("Demo: lab_genomic_agent.py 연동 테스트")
    print("="*60)

    if not os.path.exists(OUT_LAB_FOR_AGENT):
        print("  ❌ lab_for_agent.json 없음. 전처리 먼저 실행하세요.")
        return

    from lab_genomic_agent import LabGenomicAgent
    agent = LabGenomicAgent()

    with open(OUT_LAB_FOR_AGENT, encoding="utf-8") as f:
        data = json.load(f)

    # 첫 번째 환자로 테스트
    first_patient = list(data.keys())[0]
    lab_data = data[first_patient]

    print(f"\n  🧪 환자 {first_patient}의 혈액검사: {lab_data}")
    results = agent.analyze_labs(lab_data)

    if results:
        print("\n  🎯 HPO 코드 추출 결과:")
        for r in results:
            print(f"    {r['finding']} → {r['hpo_id']} (score: {r['score']})")
    else:
        print("  ℹ️  정상 범위 내 (abnormal 없음)")


if __name__ == "__main__":
    print("🏥 MIMIC-IV 전처리 시작")
    print("   폐질환 환자 혈액검사 → lab_genomic_agent.py 입력 형태 변환")

    # 파일 존재 여부 확인
    for name, path in PATHS.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024**3)
            print(f"  ✅ {name}: {size:.2f} GB")
        else:
            print(f"  ❌ {name}: 파일 없음 ({path})")

    print()

    # 전처리 4단계 실행
    lung_patient_ids = step1_extract_lung_patients()
    itemid_map, all_itemids = step2_find_lab_itemids()
    df_lab = step3_extract_lab_data(lung_patient_ids, all_itemids)
    step4_merge_and_save(df_lab, itemid_map)

    # 연동 테스트
    run_demo()

    print("\n✅ 전처리 완료!")
    print(f"   결과 파일: {OUT_LUNG_PATIENTS}")
    print(f"            {OUT_LUNG_LAB}")
    print(f"            {OUT_LAB_FOR_AGENT}")
