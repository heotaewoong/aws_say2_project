"""
Orphanet API / 로컬 데이터 Fetcher
─────────────────────────────────────────────────────────────────
확정 문서 (2026-04-29) §4 API 호출 시스템 — Orphanet 역할:
  OrphaCode (Top 3) →
    - genes_from_orphadata     (유전자 + association_type)
    - phenotypes_from_orphadata (Very frequent / Frequent HPO)
    - epidemiology.prevalence  (유병률 수치/범위)
    - epidemiology.age_of_onset (발병연령)

구현 전략:
  1차: 로컬 orphadata_weighted.csv (4335 질환, 115878행) — 빈도 기반 phenotypes
  2차: en_product6.xml  — 유전자 + association_type
       en_product9_ages.xml — 발병연령 + 유전 양식
       en_product9_prev.xml — 유병률
  3차: Orphanet REST API (https://api.orphacode.org) — 보조

확정 문서 §3.2 출력 필드명을 그대로 따른다.
"""
import os
import xml.etree.ElementTree as ET
from typing import Optional

import pandas as pd

_HERE = os.path.dirname(__file__)
_ORPHANET_CSV      = os.path.join(_HERE, "..", "data", "orphadata_weighted.csv")
_EN_PRODUCT4_XML   = os.path.join(_HERE, "..", "data", "en_product4.xml")
_EN_PRODUCT6_XML   = os.path.join(_HERE, "..", "data", "en_product6.xml")
_EN_PRODUCT9_AGES  = os.path.join(_HERE, "..", "data", "en_product9_ages.xml")
_EN_PRODUCT9_PREV  = os.path.join(_HERE, "..", "data", "en_product9_prev.xml")

# ──────────────────────────────────────────────────────────────
# 캐시 (반복 호출 방지)
# ──────────────────────────────────────────────────────────────
_DISEASE_CACHE: dict = {}
_CSV_CACHE: Optional[pd.DataFrame] = None

# XML 파싱 캐시 (파일 전체를 한 번만 파싱)
_GENE_DB: Optional[dict] = None       # {orpha_code_str: [{"gene", "association_type"}]}
_AGES_DB: Optional[dict] = None       # {orpha_code_str: {"age_of_onset", "inheritance"}}
_PREV_DB: Optional[dict] = None       # {orpha_code_str: prevalence_str}


def _load_csv() -> pd.DataFrame:
    global _CSV_CACHE
    if _CSV_CACHE is None and os.path.exists(_ORPHANET_CSV):
        _CSV_CACHE = pd.read_csv(_ORPHANET_CSV)
    return _CSV_CACHE if _CSV_CACHE is not None else pd.DataFrame()


# ──────────────────────────────────────────────────────────────
# en_product6.xml 파싱 — 유전자 + association_type
# ──────────────────────────────────────────────────────────────
def _load_gene_db() -> dict:
    global _GENE_DB
    if _GENE_DB is not None:
        return _GENE_DB

    _GENE_DB = {}
    if not os.path.exists(_EN_PRODUCT6_XML):
        return _GENE_DB

    try:
        tree = ET.parse(_EN_PRODUCT6_XML)
        root = tree.getroot()
        for disorder in root.findall(".//Disorder"):
            code = disorder.findtext("OrphaCode")
            if not code:
                continue
            genes = []
            for assoc in disorder.findall(".//DisorderGeneAssociation"):
                symbol = assoc.findtext(".//Gene/Symbol") or ""
                assoc_type_node = assoc.find(".//DisorderGeneAssociationType/Name")
                assoc_type = assoc_type_node.text if assoc_type_node is not None else ""
                if symbol:
                    genes.append({"gene": symbol, "association_type": assoc_type})
            _GENE_DB[code] = genes
    except Exception:
        pass

    return _GENE_DB


# ──────────────────────────────────────────────────────────────
# en_product9_ages.xml 파싱 — 발병연령 + 유전 양식
# ──────────────────────────────────────────────────────────────
def _load_ages_db() -> dict:
    global _AGES_DB
    if _AGES_DB is not None:
        return _AGES_DB

    _AGES_DB = {}
    if not os.path.exists(_EN_PRODUCT9_AGES):
        return _AGES_DB

    try:
        tree = ET.parse(_EN_PRODUCT9_AGES)
        root = tree.getroot()
        for disorder in root.findall(".//Disorder"):
            code = disorder.findtext("OrphaCode")
            if not code:
                continue

            # 발병연령 목록
            ages = [
                node.text
                for node in disorder.findall(".//AverageAgeOfOnset/Name")
                if node.text
            ]

            # 유전 양식
            inheritance = [
                node.text
                for node in disorder.findall(".//TypeOfInheritance/Name")
                if node.text
            ]

            _AGES_DB[code] = {
                "age_of_onset": ", ".join(ages) if ages else "",
                "inheritance":  ", ".join(inheritance) if inheritance else "",
            }
    except Exception:
        pass

    return _AGES_DB


# ──────────────────────────────────────────────────────────────
# en_product9_prev.xml 파싱 — 유병률
# Point prevalence 우선, 없으면 Cases/families 사용
# ──────────────────────────────────────────────────────────────
def _load_prev_db() -> dict:
    global _PREV_DB
    if _PREV_DB is not None:
        return _PREV_DB

    _PREV_DB = {}
    if not os.path.exists(_EN_PRODUCT9_PREV):
        return _PREV_DB

    try:
        tree = ET.parse(_EN_PRODUCT9_PREV)
        root = tree.getroot()
        for disorder in root.findall(".//Disorder"):
            code = disorder.findtext("OrphaCode")
            if not code:
                continue

            point_prev = ""
            case_count  = ""

            for prev in disorder.findall(".//Prevalence"):
                prev_type = prev.findtext(".//PrevalenceType/Name") or ""
                prev_class = prev.findtext(".//PrevalenceClass/Name") or ""
                val_moy    = prev.findtext("ValMoy") or ""

                if "Point prevalence" in prev_type and prev_class:
                    point_prev = prev_class
                elif "Cases/families" in prev_type and val_moy and val_moy != "0.0":
                    case_count = f"Cases reported: {val_moy}"

            _PREV_DB[code] = point_prev or case_count or ""
    except Exception:
        pass

    return _PREV_DB


# ──────────────────────────────────────────────────────────────
# 빈도 가중치 ↔ Frequency 라벨 변환
# ──────────────────────────────────────────────────────────────
_WEIGHT_TO_FREQ_LABEL = {
    1.0: "Always",
    0.9: "Very frequent",
    0.5: "Frequent",
    0.1: "Occasional",
    0.3: "Unknown",
}


def get_phenotypes_from_orphadata(orpha_code: str, only_frequent: bool = True) -> list:
    """
    OrphaCode → HPO 표현형 목록 (빈도 라벨 포함)

    확정 문서 §2.4-3: phenotypes_from_orphadata
      "frequency가 'Very frequent' 또는 'Frequent'인 HPO 표현형을 2개 이상 인용"
    """
    df = _load_csv()
    if df.empty:
        return []

    code_str = str(orpha_code).replace("ORPHA:", "").strip()
    rows = df[df["OrphaCode"].astype(str) == code_str]

    if rows.empty:
        return []

    phenotypes = []
    for _, row in rows.iterrows():
        weight = row["Weight"]
        freq_label = _WEIGHT_TO_FREQ_LABEL.get(weight, "Unknown")
        if only_frequent and freq_label not in ("Very frequent", "Frequent", "Always"):
            continue
        phenotypes.append({
            "hpo_id":    row["HPO_ID"],
            "hpo_term":  row["HPO_Term"],
            "frequency": freq_label,
            "weight":    float(weight),
        })

    phenotypes.sort(key=lambda x: x["weight"], reverse=True)
    return phenotypes


# ──────────────────────────────────────────────────────────────
# en_product4.xml 파싱 (질환명 + Disorder Type)
# ──────────────────────────────────────────────────────────────
def _parse_disorder_meta(orpha_code: str) -> dict:
    if not os.path.exists(_EN_PRODUCT4_XML):
        return {}

    if orpha_code in _DISEASE_CACHE:
        return _DISEASE_CACHE[orpha_code]

    code_str = str(orpha_code).replace("ORPHA:", "").strip()
    try:
        tree = ET.parse(_EN_PRODUCT4_XML)
        root = tree.getroot()
        for disorder in root.findall(".//Disorder"):
            if disorder.findtext("OrphaCode") == code_str:
                name = disorder.findtext("Name") or ""
                dtype_node = disorder.find(".//DisorderType/Name")
                dtype = dtype_node.text if dtype_node is not None else ""
                meta = {
                    "orpha_code":    code_str,
                    "disease_name":  name,
                    "disorder_type": dtype,
                }
                _DISEASE_CACHE[orpha_code] = meta
                return meta
    except Exception:
        pass

    return {}


# ──────────────────────────────────────────────────────────────
# 통합 진입점 (확정 문서 §3.2 §8번 섹션 채우기용)
# ──────────────────────────────────────────────────────────────
def get_orphanet_data(orpha_code: str) -> dict:
    """
    확정 문서 §3.2 §8번 섹션 — Top N 질환별 [Orphanet] 블록 데이터 수집

    Returns
    -------
    dict  확정 출력 키:
      - genes_from_orphadata     : list[{"gene": str, "association_type": str}]
      - phenotypes_from_orphadata: list[{"hpo_id", "hpo_term", "frequency", "weight"}]
      - epidemiology.prevalence  : str
      - epidemiology.age_of_onset: str
      - epidemiology.inheritance : str
      - disease_name             : str
      - orpha_code               : str
    """
    code_str = str(orpha_code).replace("ORPHA:", "").strip()

    meta      = _parse_disorder_meta(code_str)
    gene_db   = _load_gene_db()
    ages_db   = _load_ages_db()
    prev_db   = _load_prev_db()

    epi_ages = ages_db.get(code_str, {})

    return {
        "orpha_code":   code_str,
        "disease_name": meta.get("disease_name", ""),
        "disorder_type": meta.get("disorder_type", ""),
        "genes_from_orphadata": gene_db.get(code_str, []),
        "phenotypes_from_orphadata": get_phenotypes_from_orphadata(
            code_str, only_frequent=True
        ),
        "epidemiology": {
            "prevalence":   prev_db.get(code_str, ""),
            "age_of_onset": epi_ages.get("age_of_onset", ""),
            "inheritance":  epi_ages.get("inheritance", ""),
        },
    }


def format_orphanet_for_prompt(data: dict) -> str:
    """
    확정 문서 §3.2 §8번 섹션 [Orphanet] 블록 형식으로 변환

    출력 형식:
      [Orphanet] (희귀질환만, 일반 질환 스킵)
      - 유전자: {genes} (association_type 포함)
      - Very frequent / Frequent HPO: {hpo_frequent}
      - 유병률: {prevalence}
      - 발병연령: {age_of_onset}
    """
    if not data or not data.get("orpha_code"):
        return "[Orphanet] 데이터 없음 (일반 질환 또는 매칭 실패)"

    genes = data.get("genes_from_orphadata", [])
    if genes:
        gene_text = ", ".join(
            f"{g['gene']} ({g['association_type']})" for g in genes
        )
    else:
        gene_text = "정보 없음"

    pheno = data.get("phenotypes_from_orphadata", [])[:5]
    if pheno:
        pheno_text = ", ".join(
            f"{p['hpo_term']} [{p['frequency']}]" for p in pheno
        )
    else:
        pheno_text = "정보 없음"

    epi = data.get("epidemiology", {})
    prev_text = epi.get("prevalence") or "정보 없음"
    age_text  = epi.get("age_of_onset") or "정보 없음"
    inh_text  = epi.get("inheritance") or "정보 없음"

    return (
        f"[Orphanet]\n"
        f"- 유전자: {gene_text}\n"
        f"- Very frequent / Frequent HPO: {pheno_text}\n"
        f"- 유병률: {prev_text}\n"
        f"- 발병연령: {age_text}\n"
        f"- 유전 양식: {inh_text}"
    )


# ──────────────────────────────────────────────────────────────
# 교차검증 유틸 (DB ↔ API 일치/불일치 판정)
# ──────────────────────────────────────────────────────────────
def cross_validate_genes(orphanet_genes: list, monarch_genes: list) -> dict:
    """
    Orphanet genes_from_orphadata ↔ Monarch causal_genes 교차검증

    확정 문서 §2.4-3:
      - 일치 항목: "DB·API 교차검증 일치" 표기
      - 불일치 항목: "DB·API 불일치 — 추가 확인 필요" 경고 표기
    """
    orpha_set   = {g.get("gene", "").upper() for g in orphanet_genes if isinstance(g, dict)}
    orpha_set  |= {str(g).upper() for g in orphanet_genes if isinstance(g, str)}
    monarch_set = {str(g).upper() for g in monarch_genes}

    matched    = sorted(orpha_set & monarch_set)
    mismatched = sorted((orpha_set | monarch_set) - (orpha_set & monarch_set))

    if matched and not mismatched:
        summary = "DB·API 교차검증 일치"
    elif matched and mismatched:
        summary = f"부분 일치 — 일치: {', '.join(matched)} / 불일치: {', '.join(mismatched)}"
    elif mismatched:
        summary = "DB·API 불일치 — 추가 확인 필요"
    else:
        summary = "교차검증 데이터 없음"

    return {
        "matched":    matched,
        "mismatched": mismatched,
        "summary":    summary,
    }


# ──────────────────────────────────────────────────────────────
# 직접 실행 테스트
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Orphanet Fetcher 테스트 ===\n")

    # ORPHA:58 (Alexander disease)
    data = get_orphanet_data("58")
    print(f"OrphaCode: {data['orpha_code']}")
    print(f"질환명:    {data['disease_name']}")
    print(f"분류:      {data['disorder_type']}")

    genes = data["genes_from_orphadata"]
    print(f"\n유전자 ({len(genes)}개):")
    for g in genes[:5]:
        print(f"  - {g['gene']} ({g['association_type']})")

    epi = data["epidemiology"]
    print(f"\n유병률:   {epi['prevalence']}")
    print(f"발병연령: {epi['age_of_onset']}")
    print(f"유전양식: {epi['inheritance']}")

    print(f"\nVery frequent / Frequent HPO ({len(data['phenotypes_from_orphadata'])}개):")
    for p in data["phenotypes_from_orphadata"][:5]:
        print(f"  - {p['hpo_term']} ({p['hpo_id']}) [{p['frequency']}]")

    print("\n=== 프롬프트 형식 ===")
    print(format_orphanet_for_prompt(data))

    print("\n=== 교차검증 테스트 ===")
    result = cross_validate_genes(genes, ["GFAP", "TSC1"])
    print(f"결과: {result['summary']}")
