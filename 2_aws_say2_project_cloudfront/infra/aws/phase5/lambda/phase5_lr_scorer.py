"""
phase5_lr_scorer.py
작성자  : AWS SAY2기 권미라
작성일  : 2026-05-08
목적    : Rare-Link AI Phase 5 — 희귀질환 LR 스코어링 Lambda 핸들러
          Phase 1·2·3 통합 HPO set 입력 → 희귀질환 Listing JSON 출력

수식 근거 (LR_pipeline_v2.docx, 경로 C):
  soft_LR     = Σ w_axis × log(frequency_p / background_p)
  final_score = w_rad × log_LR_rad + w_sym × log_LR_sym
              + w_lab × log_LR_lab + w_mic × log_LR_mic
  posterior   = prior × exp(final_score)
  Listing 조건: exp(final_score) > LR_THRESHOLD (기본값 5.0)

참고 문헌:
  - Robinson et al. LIRICAL, Am J Hum Genet 2020; PMID:32755546
  - Köhler et al. HPO 2024, Nucleic Acids Res; PMID:37953324
  - LR_pipeline_v2.docx (SKKU AWS SAY2 2팀 내부 기술문서, 2026-04-19)

실행 환경:
  AWS Lambda (Python 3.12)
  환경변수: YAML_PATH (로컬) 또는 S3_BUCKET / S3_KEY (S3 로드)
  로컬 테스트: python3 phase5_lr_scorer.py
"""

import os
import json
import math
import yaml
import logging
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── 상수 ──────────────────────────────────────────────────────────────────

# Listing 출력 임계값 (기획서 3.5절, LR_pipeline_v2.docx 문제4)
LR_THRESHOLD = 5.0

# prevalence_numeric 없는 질환의 LR 분모 default
# 근거: LR_pipeline_v2.docx 4.3 — LAM prior = 0.15/1,000,000
# 희귀질환 정의(Orphanet) 상한 = 5/10,000 = 5×10⁻⁴
# 보수적 default = 1/1,000,000 (최소 사전확률)
DEFAULT_PREVALENCE = 1.0 / 1_000_000

# p < 0.05 exclusion threshold (LR_pipeline_v2.docx 2.2절)
MIN_PHASE_SCORE = 0.05

# MIMIC-CXR 기반 HPO 배경 유병률
# 근거: LR_pipeline_v2.docx 2.1 — 분모 P(x|배경)
# 정확한 값은 MIMIC-CXR 227K 집계로 교체 예정 (Final phase)
# 현재는 보수적 default 사용
DEFAULT_BACKGROUND_P = 0.05


# ── YAML 로더 ─────────────────────────────────────────────────────────────

_DISEASE_DB: dict[str, Any] | None = None


def load_disease_db() -> dict[str, Any]:
    """
    rare_disease_profiles_v3_1.yaml 로드 (싱글턴)
    로컬: YAML_PATH 환경변수
    S3  : S3_BUCKET / S3_KEY 환경변수 (Lambda 배포 시)
    """
    global _DISEASE_DB
    if _DISEASE_DB is not None:
        return _DISEASE_DB

    yaml_path = os.environ.get("YAML_PATH", "")
    s3_bucket = os.environ.get("S3_BUCKET", "")
    s3_key    = os.environ.get("S3_KEY", "kb/v7/rare_disease_profiles_v3_1.yaml")

    if yaml_path and os.path.exists(yaml_path):
        logger.info(f"YAML 로컬 로드: {yaml_path}")
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif s3_bucket:
        import boto3
        logger.info(f"YAML S3 로드: s3://{s3_bucket}/{s3_key}")
        s3  = boto3.client("s3")
        obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        data = yaml.safe_load(obj["Body"].read().decode("utf-8"))
    else:
        # 로컬 테스트: 현재 디렉터리에서 탐색
        import glob
        candidates = (
            glob.glob("rare_disease_profiles_v3*.yaml")
            + glob.glob("data/rare_disease_profiles_v3*.yaml")
            + glob.glob("**/rare_disease_profiles_v3*.yaml")
        )
        if not candidates:
            raise FileNotFoundError(
                "YAML_PATH 환경변수 또는 rare_disease_profiles_v3*.yaml 필요"
            )
        logger.info(f"YAML 자동 탐색: {candidates[0]}")
        with open(candidates[0], encoding="utf-8") as f:
            data = yaml.safe_load(f)

    _DISEASE_DB = data.get("rare_diseases", {})
    logger.info(f"DB 로드 완료: {len(_DISEASE_DB)}개 질환")
    return _DISEASE_DB


# ── Stage 1: HPO 교집합 필터 ──────────────────────────────────────────────

def stage1_hpo_filter(
    patient_hpo: set[str],
    db: dict[str, Any],
) -> list[tuple[str, Any, set[str]]]:
    """
    Phase 5 Stage 1: HPO 교집합 필터
    조건: patient_hpo ∩ disease_hpo ≠ ∅

    근거: 기획서 3.5절 Stage 1
      376개 희귀질환 → 교집합 필터 → ~30개 후보로 축소

    Skip 조건:
      - hpo_id가 '—' (Orphanet HPO 미기재)
      - Category / Clinical group (상위노드, LR 계산 의미 없음)

    반환: [(disease_key, disease_profile, matched_hpo_set), ...]
    """
    SKIP_DTYPE = {"Category", "Clinical group"}
    candidates = []

    for key, profile in db.items():
        # 상위노드 스킵
        if profile.get("disorder_type", "") in SKIP_DTYPE:
            continue

        # 질환 HPO set 구성 (실제 HP: 코드만)
        disease_hpo = {
            h["hpo_id"]
            for h in profile.get("hpo_symptoms", [])
            if h.get("hpo_id", "—") != "—"
            and str(h.get("hpo_id", "")).startswith("HP:")
        }

        if not disease_hpo:
            continue

        matched = patient_hpo & disease_hpo
        if matched:
            candidates.append((key, profile, matched))

    logger.info(f"Stage 1 통과: {len(candidates)}개 / {len(db)}개")
    return candidates


# ── Stage 2: LR 계산 ──────────────────────────────────────────────────────

def _hpo_to_axis(hpo_id: str, profile: dict) -> str:
    """
    HPO 코드가 어느 모달리티 축에 속하는지 판단
    근거: LR_pipeline_v2.docx 2.4 — 모달리티 가중치
    현재는 symptoms 축으로 통합 처리
    (Phase 2 X-ray HPO, Phase 3 Lab HPO 분리는 추후 확장)
    """
    return "symptoms"


def _compute_axis_log_lr(
    matched_hpos: set[str],
    patient_scores: dict[str, float],
    profile: dict,
    axis: str,
) -> float:
    """
    단일 모달리티 축의 log LR 계산

    수식 (LR_pipeline_v2.docx 2.2 Soft LR):
      effective_LR_i = p_i × LR_pos_i + (1-p_i) × LR_neg_i
      LR_pos = frequency_p / background_p
      LR_neg = (1 - frequency_p) / (1 - background_p)
      log_LR_axis = Σ log(effective_LR_i)

    p_i: Phase 1·2·3에서 넘어온 HPO 확률 점수
         없으면 1.0 (환자에게 해당 HPO가 관찰됨)
    """
    log_lr_sum = 0.0

    hpo_map = {
        h["hpo_id"]: h
        for h in profile.get("hpo_symptoms", [])
        if h.get("hpo_id", "—") != "—"
    }

    for hpo_id in matched_hpos:
        h = hpo_map.get(hpo_id)
        if h is None:
            continue

        freq_p = h.get("frequency_p")
        if freq_p is None:
            continue
        freq_p = max(1e-6, min(1 - 1e-6, float(freq_p)))

        bg_p = DEFAULT_BACKGROUND_P
        bg_p = max(1e-6, min(1 - 1e-6, bg_p))

        # Phase 1·2·3 에서 넘어온 HPO 확률 점수
        p_i = float(patient_scores.get(hpo_id, 1.0))

        # p < 0.05 exclusion (LR_pipeline_v2.docx 2.2)
        if p_i < MIN_PHASE_SCORE:
            continue

        lr_pos = freq_p / bg_p
        lr_neg = (1 - freq_p) / (1 - bg_p)

        # Soft LR
        effective_lr = p_i * lr_pos + (1 - p_i) * lr_neg
        effective_lr = max(1e-9, effective_lr)

        log_lr_sum += math.log(effective_lr)

    return log_lr_sum


def stage2_lr_compute(
    candidates: list[tuple[str, Any, set[str]]],
    patient_hpo: set[str],
    phase1_scores: dict[str, float],
    phase2_scores: dict[str, float],
    phase3_scores: dict[str, float],
) -> list[dict]:
    """
    Phase 5 Stage 2: 4축 가중치 LR 계산

    수식 (LR_pipeline_v2.docx 2.4):
      final_score = w_rad × log_LR_rad
                  + w_sym × log_LR_sym
                  + w_lab × log_LR_lab
                  + w_mic × log_LR_mic

    LR_threshold 조건 (기획서 3.5절):
      exp(final_score) > LR_THRESHOLD (5.0)

    출력: LR 내림차순 정렬된 Listing 후보
    """
    # 통합 환자 점수 (Phase 1·2·3 합산, 최대값 우선)
    all_scores: dict[str, float] = {}
    for scores in (phase1_scores, phase2_scores, phase3_scores):
        for hpo_id, score in scores.items():
            all_scores[hpo_id] = max(all_scores.get(hpo_id, 0.0), score)

    results = []

    for key, profile, matched_hpo in candidates:
        weights = profile.get("lr_weights", {
            "radiology": 0.35, "symptoms": 0.25, "lab": 0.25, "micro": 0.15
        })
        w_rad = float(weights.get("radiology", 0.35))
        w_sym = float(weights.get("symptoms",  0.25))
        w_lab = float(weights.get("lab",       0.25))
        w_mic = float(weights.get("micro",     0.15))

        # 현재 Phase 5는 통합 HPO set 기준 단일 축 계산
        # Phase 2·3 분리 점수가 들어오면 축별 분리 가능 (확장 포인트)
        log_lr_sym = _compute_axis_log_lr(matched_hpo, all_scores, profile, "symptoms")
        log_lr_rad = _compute_axis_log_lr(matched_hpo, phase2_scores, profile, "radiology")
        log_lr_lab = _compute_axis_log_lr(matched_hpo, phase3_scores, profile, "lab")
        log_lr_mic = 0.0  # Phase 3 micro HPO 추후 연동

        final_score = (
            w_rad * log_lr_rad
            + w_sym * log_lr_sym
            + w_lab * log_lr_lab
            + w_mic * log_lr_mic
        )

        lr_value = math.exp(final_score)

        # prior 계산 (LR_pipeline_v2.docx 7.3)
        prev = profile.get("prevalence_numeric")
        prior = float(prev) / 1_000_000 if prev else DEFAULT_PREVALENCE
        posterior = prior * lr_value

        results.append({
            "disease_key":    key,
            "orphacode":      profile.get("orphacode", ""),
            "disease_en":     profile.get("disease_en", ""),
            "disease_kr":     profile.get("disease_kr", ""),
            "icd10":          profile.get("icd10", []),
            "lr_category":    profile.get("lr_category", "G"),
            "disorder_type":  profile.get("disorder_type", ""),
            "lr_value":       round(lr_value, 4),
            "final_score":    round(final_score, 6),
            "posterior":      round(posterior, 10),
            "prior":          round(prior, 12),
            "matched_hpo":    sorted(matched_hpo),
            "matched_count":  len(matched_hpo),
            "log_lr_rad":     round(log_lr_rad, 4),
            "log_lr_sym":     round(log_lr_sym, 4),
            "log_lr_lab":     round(log_lr_lab, 4),
            "passes_threshold": lr_value > LR_THRESHOLD,
        })

    # LR 내림차순 정렬
    results.sort(key=lambda x: x["lr_value"], reverse=True)
    return results


# ── 출력 포맷팅 ───────────────────────────────────────────────────────────

def build_listing_output(
    scored: list[dict],
    patient_hpo: list[str],
    threshold: float = LR_THRESHOLD,
) -> dict:
    """
    Phase 5 최종 출력 JSON 구성
    RAG Step 7 입력 형식

    출력 구조 (기획서 3.6.2 Listing):
      primary key = Orphacode
      조건: LR > 5.0 AND matched_hpo ⊇ minimum_HPO_set(d) 80%
    """
    listing = [r for r in scored if r["passes_threshold"]]

    return {
        "phase":               5,
        "output_type":         "listing",
        "lr_threshold":        threshold,
        "total_evaluated":     len(scored),
        "listing_count":       len(listing),
        "patient_hpo":         patient_hpo,
        "listing": [
            {
                "orphacode":     r["orphacode"],
                "disease_en":    r["disease_en"],
                "disease_kr":    r["disease_kr"],
                "icd10":         r["icd10"],
                "lr_value":      r["lr_value"],
                "lr_category":   r["lr_category"],
                "matched_hpo":   r["matched_hpo"],
                "matched_count": r["matched_count"],
                "evidence": {
                    "log_lr_radiology": r["log_lr_rad"],
                    "log_lr_symptoms":  r["log_lr_sym"],
                    "log_lr_lab":       r["log_lr_lab"],
                },
            }
            for r in listing
        ],
        # 임계값 미달 후보 (RAG 참고용)
        "sub_threshold": [
            {
                "orphacode":   r["orphacode"],
                "disease_en":  r["disease_en"],
                "lr_value":    r["lr_value"],
                "matched_hpo": r["matched_hpo"],
            }
            for r in scored
            if not r["passes_threshold"]
        ][:10],
    }


# ── Lambda 핸들러 ─────────────────────────────────────────────────────────

def lambda_handler(event: dict, context: Any = None) -> dict:
    """
    AWS Lambda 핸들러 (Phase 5 진입점)

    입력 event (Phase 1·2·3 통합 HPO set):
    {
      "patient_hpo":    ["HP:0002088", "HP:0001626", ...],
      "phase1_scores":  {"HP:0002088": 0.9, ...},
      "phase2_scores":  {"HP:0002088": 0.75, ...},
      "phase3_scores":  {"HP:0002088": 0.6, ...}
    }

    출력: Listing JSON (RAG Step 7 입력)
    """
    logger.info("Phase 5 LR Scorer 시작")

    # 입력 파싱
    patient_hpo_list = event.get("patient_hpo", [])
    phase1_scores    = event.get("phase1_scores", {})
    phase2_scores    = event.get("phase2_scores", {})
    phase3_scores    = event.get("phase3_scores", {})

    if not patient_hpo_list:
        return {
            "statusCode": 400,
            "error": "patient_hpo 필드 필요",
        }

    patient_hpo = set(patient_hpo_list)

    # YAML DB 로드
    db = load_disease_db()

    # Stage 1: HPO 교집합 필터
    candidates = stage1_hpo_filter(patient_hpo, db)

    if not candidates:
        logger.info("Stage 1 통과 후보 없음 → 빈 Listing 반환")
        return {
            "statusCode": 200,
            "body": build_listing_output([], patient_hpo_list),
        }

    # Stage 2: LR 계산
    scored = stage2_lr_compute(
        candidates, patient_hpo,
        phase1_scores, phase2_scores, phase3_scores,
    )

    # 출력 구성
    output = build_listing_output(scored, patient_hpo_list)

    logger.info(
        f"Phase 5 완료: 평가 {output['total_evaluated']}개, "
        f"Listing {output['listing_count']}개 (LR > {LR_THRESHOLD})"
    )

    return {
        "statusCode": 200,
        "body": output,
    }


# ── 로컬 테스트 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    로컬 실행 예시
    YAML_PATH 환경변수 설정 후:
      YAML_PATH=./rare_disease_profiles_v3_1.yaml python3 phase5_lr_scorer.py
    """
    test_event = {
        "patient_hpo": [
            "HP:0002088",   # Abnormal lung morphology
            "HP:0001626",   # Abnormal heart morphology
            "HP:0002113",   # Pulmonary cysts
            "HP:0000961",   # Cyanosis
            "HP:0002878",   # Respiratory failure
            "HP:0006536",   # Pulmonary obstruction
        ],
        "phase1_scores": {
            "HP:0002088": 0.90,
            "HP:0002113": 0.85,
            "HP:0002878": 0.75,
        },
        "phase2_scores": {
            "HP:0002088": 0.78,
            "HP:0001626": 0.60,
        },
        "phase3_scores": {
            "HP:0000961": 0.70,
        },
    }

    result = lambda_handler(test_event)
    print(json.dumps(result, ensure_ascii=False, indent=2))
