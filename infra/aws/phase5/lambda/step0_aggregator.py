"""
step0_aggregator.py
작성자  : AWS SAY2기 권미라
작성일  : 2026-05-08
목적    : Phase 5 Step 0 — 다중 모달리티 HPO Aggregator
          Lab 수치 / Micro 결과 → HPO 코드 변환
          Phase 1(증상) + Phase 2(X-ray) HPO와 통합하여 최종 HPO set 반환

근거:
  - 종합 아키텍처 보고서 v2 8장 Step 0 데이터 흐름
  - lab_reference_ranges_v9_5.yaml (MIMIC-IV itemid 기반, 135개 항목)
  - Jacobsen JOB et al. GA4GH Phenopacket, Nat Biotechnol 2022; PMID:35705716
  - Köhler S et al. HPO 2021, Nucleic Acids Res; PMID:33264411

변환 로직:
  [수치형] ranges.lower/upper 기준 → low/high 상태 판별
           critical 필드 있으면 critical_low/critical_high 우선 적용
           → hpo_terms[상태] HPO 코드 추출

  [범주형] categorical 항목 (Micro 등) → 양성/검출 시 hpo_terms["high"] 추출

  [HPO None] hpo_terms 값이 None인 항목 → 스킵 (임상 의미 없음)

audit_trail:
  source_modality 필드로 HPO 출처 기록
  (history/xray/lab/micro — 종합 아키텍처 보고서 Step 0 명세)
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Lab Reference YAML 로드 (싱글턴) ──────────────────────────────────────
_LAB_REF: dict | None = None


def _load_lab_ref() -> dict:
    global _LAB_REF
    if _LAB_REF is not None:
        return _LAB_REF

    # 탐색 경로: 환경변수 → Layer(/opt) → 현재 디렉터리
    candidates = [
        os.environ.get("LAB_REF_PATH", ""),
        "/opt/lab_reference_ranges_v9_5.yaml",      # Lambda Layer
        "/opt/lab_reference_ranges_v9_4.yaml",      # 하위 호환
        str(Path(__file__).parent / "lab_reference_ranges_v9_5.yaml"),
    ]

    for path in candidates:
        if path and Path(path).exists():
            logger.info(f"Lab Reference 로드: {path}")
            with open(path, encoding="utf-8") as f:
                _LAB_REF = yaml.safe_load(f)
            return _LAB_REF

    raise FileNotFoundError(
        "lab_reference_ranges_v9_5.yaml 를 찾을 수 없음. "
        "LAB_REF_PATH 환경변수 또는 Lambda Layer 확인 필요."
    )


# ── 수치 판별 헬퍼 ─────────────────────────────────────────────────────────

def _get_state(value: float, item: dict) -> str | None:
    """
    수치형 Lab 항목의 상태 판별
    반환: "critical_low" | "critical_high" | "low" | "high" | "normal" | None

    우선순위:
      1. critical 필드 (명시적 위험 임계값)
      2. ranges 필드 (정상 범위 이탈)
    """
    # critical 필드 우선 적용
    critical = item.get("critical", {})
    if critical:
        crit_low  = critical.get("low")
        crit_high = critical.get("high")
        if crit_low is not None and value < crit_low:
            return "critical_low"
        if crit_high is not None and value > crit_high:
            return "critical_high"

    # ranges 기준 low/high 판별
    ranges = item.get("ranges", {})
    if not ranges:
        return None

    # 성별/흡연 구분 ranges는 lower/upper 평균으로 단순화
    # (실제 환자 성별 정보 없을 때 보수적 적용)
    lower = ranges.get("lower") or ranges.get("lower_male") or ranges.get("lower_female")
    upper = (ranges.get("upper")
             or ranges.get("upper_male")
             or ranges.get("upper_female")
             or ranges.get("upper_nonsmoker"))

    if lower is not None and value < lower:
        return "low"
    if upper is not None and value > upper:
        return "high"

    return "normal"


# ── 수치형 변환 ───────────────────────────────────────────────────────────

def _numeric_to_hpo(
    itemid: str | int,
    value: float,
    lab_ref: dict,
) -> list[dict]:
    """
    수치 하나 → HPO 코드 목록
    반환: [{"hpo_id": "HP:xxxx", "source": "lab", "itemid": ..., "state": ...}, ...]
    """
    # itemid: 숫자형(MIMIC-IV) 또는 EXT_* 문자형 모두 처리
    key = itemid
    if isinstance(itemid, str) and not itemid.startswith("EXT"):
        try:
            key = int(itemid)
        except ValueError:
            pass
    item = lab_ref.get(key)
    if item is None:
        logger.debug(f"itemid {itemid} — lab_ref에 없음, 스킵")
        return []

    state = _get_state(value, item)
    if state is None or state == "normal":
        return []

    hpo_terms = item.get("hpo_terms", {})
    if not isinstance(hpo_terms, dict):
        return []

    hpo_id = hpo_terms.get(state)
    if not hpo_id:
        # critical_low → low fallback, critical_high → high fallback
        if state == "critical_low":
            hpo_id = hpo_terms.get("low")
        elif state == "critical_high":
            hpo_id = hpo_terms.get("high")

    if not hpo_id or not str(hpo_id).startswith("HP:"):
        return []

    return [{
        "hpo_id":   str(hpo_id),
        "source":   "lab",
        "itemid":   itemid,
        "name":     item.get("name", ""),
        "value":    value,
        "unit":     item.get("unit", ""),
        "state":    state,
        "category": item.get("category", ""),
    }]


# ── 범주형 변환 (Micro / PCR / 항원 등) ──────────────────────────────────

def _categorical_to_hpo(
    itemid: str | int,
    result: str,
    lab_ref: dict,
) -> list[dict]:
    """
    범주형 결과 → HPO 코드
    양성/검출/Positive/Detected → hpo_terms["high"] 적용

    근거: lab_reference_ranges_v9_5.yaml N_Infection_Microbiology 항목
    """
    key = itemid
    if isinstance(itemid, str) and not itemid.startswith("EXT"):
        try:
            key = int(itemid)
        except ValueError:
            pass
    item = lab_ref.get(key)
    if item is None:
        return []

    # 양성 판별 키워드
    POSITIVE_KEYWORDS = {
        "positive", "detected", "present", "growth",
        "양성", "검출", "확인", "동정"
    }

    result_lower = str(result).lower().strip()
    is_positive = any(kw in result_lower for kw in POSITIVE_KEYWORDS)

    if not is_positive:
        return []

    hpo_terms = item.get("hpo_terms", {})
    if not isinstance(hpo_terms, dict):
        return []

    hpo_id = hpo_terms.get("high")
    if not hpo_id or not str(hpo_id).startswith("HP:"):
        return []

    return [{
        "hpo_id":   str(hpo_id),
        "source":   "micro",
        "itemid":   itemid,
        "name":     item.get("name", ""),
        "result":   result,
        "state":    "high",
        "category": item.get("category", ""),
    }]


# ── 핵심 공개 함수 ────────────────────────────────────────────────────────

def aggregate_hpo(
    history_hpo:   list[str],
    xray_hpo:      list[str],
    lab_numeric:   dict[str | int, float],
    lab_categorical: dict[str | int, str],
    phase1_scores: dict[str, float] | None = None,
    phase2_scores: dict[str, float] | None = None,
) -> dict:
    """
    Phase 5 Step 0 — 다중 모달리티 HPO 통합

    입력:
      history_hpo      : Phase 1 증상 HPO 목록 (예: ["HP:0012735", ...])
      xray_hpo         : Phase 2 X-ray HPO 목록 (예: ["HP:0002088", ...])
      lab_numeric      : {itemid: 수치} (예: {220277: 88.0, 51301: 14.2})
      lab_categorical  : {itemid: 결과문자열} (예: {"AFB": "Positive"})
      phase1_scores    : {hpo_id: 확률} — LR 계산에 사용
      phase2_scores    : {hpo_id: 확률} — LR 계산에 사용

    출력:
      {
        "patient_hpo"    : ["HP:xxxx", ...],   # 중복 제거된 통합 HPO set
        "phase1_scores"  : {...},
        "phase2_scores"  : {...},
        "phase3_scores"  : {...},              # lab/micro HPO 확률 (severity 기반)
        "audit_trail"    : [                   # 출처 추적 (종합 아키텍처 Step 0)
          {"hpo_id": "HP:xxxx", "source": "history|xray|lab|micro", ...}
        ]
      }

    근거: 종합 아키텍처 보고서 v2 8장 Step 0 HPO Aggregator
    """
    lab_ref = _load_lab_ref()

    audit_trail: list[dict] = []
    phase3_scores: dict[str, float] = {}

    # ── history HPO (Phase 1) ──────────────────────────────────────────
    for hpo_id in history_hpo:
        if hpo_id and str(hpo_id).startswith("HP:"):
            audit_trail.append({
                "hpo_id": hpo_id,
                "source": "history",
                "source_modality": "Phase1_Symptom",
            })

    # ── X-ray HPO (Phase 2) ───────────────────────────────────────────
    for hpo_id in xray_hpo:
        if hpo_id and str(hpo_id).startswith("HP:"):
            audit_trail.append({
                "hpo_id": hpo_id,
                "source": "xray",
                "source_modality": "Phase2_Xray",
            })

    # ── 수치형 Lab → HPO ──────────────────────────────────────────────
    for itemid, value in lab_numeric.items():
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            logger.warning(f"itemid {itemid} 수치 변환 실패: {value}")
            continue

        converted = _numeric_to_hpo(itemid, value_f, lab_ref)
        for entry in converted:
            audit_trail.append({**entry, "source_modality": "Phase3_Lab"})

            # severity → phase3_scores 확률 배정
            # critical: 0.9, low/high: 0.7 (LR_pipeline_v2.docx severity ordinal 근거)
            hpo_id = entry["hpo_id"]
            score  = 0.9 if "critical" in entry["state"] else 0.7
            phase3_scores[hpo_id] = max(phase3_scores.get(hpo_id, 0.0), score)

    # ── 범주형 Lab/Micro → HPO ────────────────────────────────────────
    for itemid, result in lab_categorical.items():
        converted = _categorical_to_hpo(itemid, result, lab_ref)
        for entry in converted:
            audit_trail.append({**entry, "source_modality": "Phase3_Micro"})
            hpo_id = entry["hpo_id"]
            phase3_scores[hpo_id] = max(phase3_scores.get(hpo_id, 0.0), 0.8)

    # ── 통합 HPO set 구성 (중복 제거) ─────────────────────────────────
    patient_hpo = list({entry["hpo_id"] for entry in audit_trail
                        if entry["hpo_id"].startswith("HP:")})

    logger.info(
        f"Step 0 완료: history={len(history_hpo)}, xray={len(xray_hpo)}, "
        f"lab_numeric={len(lab_numeric)}, lab_categorical={len(lab_categorical)} "
        f"→ 통합 HPO {len(patient_hpo)}개"
    )

    return {
        "patient_hpo":   patient_hpo,
        "phase1_scores": phase1_scores or {},
        "phase2_scores": phase2_scores or {},
        "phase3_scores": phase3_scores,
        "audit_trail":   audit_trail,
    }


# ── 로컬 테스트 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import os

    os.environ["LAB_REF_PATH"] = "/tmp/lab_ref_v95.yaml"
    logging.basicConfig(level=logging.INFO)

    result = aggregate_hpo(
        history_hpo   = ["HP:0012735", "HP:0002094"],   # Phase 1: 기침, 호흡곤란
        xray_hpo      = ["HP:0002088"],                  # Phase 2: Abnormal lung morphology
        lab_numeric   = {
            220277: 82.0,    # SpO2 82% → critical_low → HP:0002878 (Respiratory failure)
            51301:  14.2,    # WBC 14.2 K/uL → high → HP:0011897 (Leukocytosis)
            50821:  55.0,    # pO2 55 mmHg → critical_low
            50818:  55.0,    # pCO2 55 mmHg → critical_high
            50813:  3.5,     # Lactate 3.5 → high
        },
        lab_categorical = {
            "EXT_B":  "Positive",    # AFB Smear 양성 → HP:0032243 (TB)
            "EXT_I":  "Growth",      # Blood Culture 양성 → HP:0034196
        },
        phase1_scores = {"HP:0012735": 0.9, "HP:0002094": 0.85},
        phase2_scores = {"HP:0002088": 0.78},
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
