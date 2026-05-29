"""
lambda/handler_v3.py — Phase 5 Lambda 핸들러 (v4 기준)
작성자  : AWS SAY2기 권미라
작성일  : 2026-05-14
목적    : Phase 5 진입점
          Phase 1·2·3와 동일한 raw 소스 직접 읽기
          → Step 0 HPO 변환 → LR 계산 → DB 저장

변경 이력:
  v1: phase3_integrated_ranking 단일 테이블 읽기
  v2: v4 문서 기준 — Phase 1·2·lab·patient 직접 읽기
  v3: SLRM_WEIGHTS 제거, audit_trail/step0_log DB 저장 추가
      → 실제 계산에 쓰인 YAML lr_weights를 listed_diseases에 기록
      → top_lr_score / top_lr_orphacode 컬럼 추가 (빠른 조회용)

흐름:
  [Step Functions → session_id + patient_id]
          ↓
  [_read_phase1()]  phase1_hpo_extraction
  [_read_phase2()]  phase2_xray_processing
  [_read_lab()]     lab_result
  [_read_patient()] patient_profile
          ↓
  [step0_aggregator.aggregate_hpo()]  lab raw → HPO 변환
          ↓
  [phase5_lr_scorer.lambda_handler()]  LR 계산
          ↓
  [_write_phase5()]  phase5_rare_disease_listing INSERT

수정이 필요한 항목:
  - DB_HOST       : Aurora 클러스터 엔드포인트
  - DB_SECRET_ARN : Secrets Manager ARN
  - S3_BUCKET / S3_KEY : YAML KB S3 경로 (허태웅 확인)
  - VPC Subnet ID / SG ID : template.yaml (허태웅 확인)

참고:
  - rarelinkai_phase5_rare_disease_listing.docx (v4)
  - LR_pipeline_v2.docx 경로 C (희귀질환 카테고리 가중치 A~G)
  - 4-layer-db-team-guide.md
  - Robinson et al. LIRICAL, PMID:32755546
"""

import json
import logging
import os
import time
from typing import Any

import boto3
import psycopg2
from psycopg2.extras import RealDictCursor

from step0_aggregator import aggregate_hpo
from phase5_lr_scorer import lambda_handler as _lr_scorer

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# ── 환경변수 ──────────────────────────────────────────────────────────────
DB_HOST       = os.environ.get("DB_HOST",
    "patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com")
DB_PORT       = int(os.environ.get("DB_PORT", "5432"))
DB_NAME       = os.environ.get("DB_NAME", "rarelink")
DB_SCHEMA     = os.environ.get("DB_SCHEMA", "rarelinkai")
DB_SECRET_ARN = os.environ.get("DB_SECRET_ARN", "rare-link-ai/aurora/app-user")
LR_THRESHOLD  = float(os.environ.get("LR_THRESHOLD", "5.0"))
RARE_DB_VER   = os.environ.get("RARE_DB_VER", "rare_disease_profiles_v3_1")
LR_PIPELINE_VER = os.environ.get("LR_PIPELINE_VER", "LR_pipeline_v2")


# ── DB 연결 ───────────────────────────────────────────────────────────────

def _get_db_password() -> str:
    """
    Secrets Manager에서 DB 비밀번호 조회
    주의: rare-link-ai/aurora/app-user 는 plain text 방식
          json.loads() 불가 — SecretString 직접 사용
    (resource_ids.md 확인)
    """
    client = boto3.client("secretsmanager")
    resp   = client.get_secret_value(SecretId=DB_SECRET_ARN)
    return resp["SecretString"].strip()


def _get_conn():
    pw = _get_db_password()
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user="app_user", password=pw, connect_timeout=5,
        options=f"-c search_path={DB_SCHEMA}",
        cursor_factory=RealDictCursor,
    )


# ── DB READ ───────────────────────────────────────────────────────────────

def _read_phase1(conn, session_id: str) -> dict:
    """
    phase1_hpo_extraction 읽기
    positive_hpo[*].confidence → phase1_scores (p_i)
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT executed_at, positive_hpo, negative_hpo
            FROM phase1_hpo_extraction
            WHERE session_id = %s
            ORDER BY executed_at DESC LIMIT 1
        """, (session_id,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"phase1_hpo_extraction 없음: session_id={session_id}")
    return dict(row)


def _read_phase2(conn, session_id: str) -> list[dict]:
    """
    phase2_xray_processing 읽기 (study별 최신)
    xray_hpo_inferred[*].confidence → phase2_scores (p_i)
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT study_id, executed_at,
                   xray_hpo_inferred, densenet_findings, mask_quality_flag
            FROM phase2_xray_processing p2
            WHERE session_id = %s
              AND executed_at = (
                  SELECT MAX(executed_at)
                  FROM phase2_xray_processing
                  WHERE session_id = %s AND study_id = p2.study_id
              )
        """, (session_id, session_id))
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def _read_lab(conn, patient_id: str) -> list[dict]:
    """
    lab_result 직접 읽기 (raw 수치)
    → step0_aggregator가 HPO + phase3_scores(p_i) 변환
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT observation_category, loinc_code,
                   test_name_ko, value_numeric, value_text,
                   abnormal_flag, severity, micro_result,
                   measured_at
            FROM lab_result
            WHERE patient_id = %s
              AND observation_category IN ('lab', 'vital_sign', 'microbiology')
            ORDER BY observation_category, measured_at DESC
        """, (patient_id,))
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def _read_patient(conn, patient_id: str) -> dict:
    """patient_profile 읽기 (input_data_meta 컨텍스트용)"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT age_years, sex, smoking_status, occupation
            FROM patient_profile
            WHERE patient_id = %s
        """, (patient_id,))
        row = cur.fetchone()
    return dict(row) if row else {}


# ── 입력 변환 헬퍼 ────────────────────────────────────────────────────────

def _build_phase1_scores(positive_hpo: list[dict]) -> dict[str, float]:
    """
    Phase 1 positive_hpo.confidence → phase1_scores
    Soft LR 수식의 p_i로 사용
    [{hpo: "HP:0002094", confidence: 0.92}] → {"HP:0002094": 0.92}
    """
    scores: dict[str, float] = {}
    for item in positive_hpo:
        hpo_id = item.get("hpo", "")
        conf   = float(item.get("confidence", 1.0))
        if hpo_id.startswith("HP:"):
            scores[hpo_id] = max(scores.get(hpo_id, 0.0), conf)
    return scores


def _build_phase2_scores(xray_studies: list[dict]) -> dict[str, float]:
    """
    Phase 2 xray_hpo_inferred.confidence → phase2_scores
    Soft LR 수식의 p_i로 사용
    """
    scores: dict[str, float] = {}
    for study in xray_studies:
        for item in study.get("xray_hpo_inferred", []):
            hpo_id = item.get("hpo", "")
            conf   = float(item.get("confidence", item.get("prob", 1.0)))
            if hpo_id.startswith("HP:"):
                scores[hpo_id] = max(scores.get(hpo_id, 0.0), conf)
    return scores


def _build_lab_inputs(lab_rows: list[dict]) -> tuple[dict, dict]:
    """
    lab_result rows → step0_aggregator 입력 형태 변환
    수치형 → lab_numeric {loinc: value}
    범주형/micro → lab_categorical {key: organism}
    """
    lab_numeric:     dict[str | int, float] = {}
    lab_categorical: dict[str, str]          = {}

    for row in lab_rows:
        cat   = row.get("observation_category", "")
        loinc = row.get("loinc_code", "")
        val_n = row.get("value_numeric")
        val_t = row.get("value_text", "")
        micro = row.get("micro_result")

        if cat == "microbiology" and micro:
            organism = micro.get("organism", "")
            if organism:
                key = f"MICRO_{loinc}" if loinc else f"MICRO_{organism[:20]}"
                lab_categorical[key] = organism
        elif val_n is not None:
            if loinc:
                lab_numeric[loinc] = float(val_n)
        elif val_t:
            if loinc:
                lab_categorical[loinc] = val_t

    return lab_numeric, lab_categorical


def _build_hpo_lists(
    phase1: dict,
    phase2_studies: list[dict],
) -> tuple[list[str], list[str]]:
    """Phase 1·2 HPO → history_hpo, xray_hpo (step0_aggregator 입력)"""
    history_hpo = [
        item["hpo"]
        for item in phase1.get("positive_hpo", [])
        if item.get("hpo", "").startswith("HP:")
    ]
    xray_hpo = [
        item["hpo"]
        for study in phase2_studies
        for item in study.get("xray_hpo_inferred", [])
        if item.get("hpo", "").startswith("HP:")
    ]
    return history_hpo, xray_hpo


# ── DB WRITE ──────────────────────────────────────────────────────────────

def _write_phase5(
    conn,
    session_id:       str,
    input_hpo_used:   dict,
    listing:          list,
    sub_threshold:    list,
    total_evaluated:  int,
    inference_ms:     int,
    input_data_meta:  dict,
    audit_trail:      list,
    step0_log:        dict,
) -> None:
    """
    phase5_rare_disease_listing INSERT (v4 스키마)

    LR 기록:
      - listed_diseases[] 각 항목에 lr_value + weights_applied 포함
        (실제 계산에 쓰인 YAML lr_weights가 weights_applied에 기록됨)
      - top_lr_score  : 1위 질환 LR값 (빠른 조회용)
      - top_lr_orphacode : 1위 질환 Orphacode (빠른 조회용)

    4-layer-db-team-guide.md 규칙4: 모델 버전 필수 기록
    """
    # 1위 LR 값 추출
    top_lr_score    = listing[0]["lr_value"]    if listing else None
    top_lr_orphacode = listing[0]["orphacode"]  if listing else None

    listing_criteria = {
        "sort_by":        "lr_value",
        "lr_threshold":   LR_THRESHOLD,
        "rare_db_ver":    RARE_DB_VER,
        "lr_pipeline_ver": LR_PIPELINE_VER,
        "note": "lr_weights는 질환별 YAML lr_weights (A~G 카테고리) 사용"
    }

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO phase5_rare_disease_listing (
                session_id,
                input_phase4_top_orphas,
                input_hpo_used,
                rare_db_ver,
                rare_db_source,
                listed_diseases,
                listing_criteria,
                total_listed_count,
                top_lr_score,
                top_lr_orphacode,
                external_api_called,
                external_api_versions,
                inference_time_ms,
                input_data_meta
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """, (
            session_id,
            json.dumps([]),                               # DEPRECATED
            json.dumps(input_hpo_used, ensure_ascii=False),
            RARE_DB_VER,
            "local_orphadata",
            json.dumps(listing,          ensure_ascii=False),
            json.dumps(listing_criteria, ensure_ascii=False),
            len(listing),
            top_lr_score,
            top_lr_orphacode,
            False,
            None,
            inference_ms,
            json.dumps(input_data_meta,  ensure_ascii=False),
        ))
    conn.commit()


# ── 응답 포맷 ─────────────────────────────────────────────────────────────

def _ok(body: dict) -> dict:
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, ensure_ascii=False),
    }


def _err(status: int, message: str) -> dict:
    logger.error(f"Phase 5 오류 ({status}): {message}")
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": message}, ensure_ascii=False),
    }


# ── Lambda 핸들러 ─────────────────────────────────────────────────────────

def lambda_handler(event: dict, context: Any = None) -> dict:
    """
    Phase 5 Lambda 진입점 (v4)

    입력: {"session_id": "UUID", "patient_id": "str"}
    출력: {"statusCode": 200, "listing_count": N, "top_lr_score": X, ...}

    Step 0 실패 시:
      - lab 변환 실패해도 Phase 1+2 HPO만으로 LR 계산 시도
      - 200 유지, step0_ok=False로 기록
    """
    t_start    = time.time()
    request_id = getattr(context, "aws_request_id", "local")
    logger.info(f"Phase 5 시작 (v4) | requestId: {request_id}")

    # ── 1. 입력 파싱 ──────────────────────────────────────────────────
    if isinstance(event.get("body"), str):
        body = json.loads(event["body"])
    elif isinstance(event.get("body"), dict):
        body = event["body"]
    else:
        body = event

    session_id = body.get("session_id")
    patient_id = body.get("patient_id")

    if not session_id:
        return _err(400, "session_id 필드 필요")
    if not patient_id:
        return _err(400, "patient_id 필드 필요")

    logger.info(f"session_id: {session_id} | patient_id: {patient_id}")

    # ── 2. DB 연결 ────────────────────────────────────────────────────
    try:
        conn = _get_conn()
    except Exception as e:
        return _err(503, f"DB 연결 실패: {e}")

    # ── 3. 입력 읽기 ──────────────────────────────────────────────────
    try:
        phase1         = _read_phase1(conn, session_id)
        phase2_studies = _read_phase2(conn, session_id)
        lab_rows       = _read_lab(conn, patient_id)
        patient        = _read_patient(conn, patient_id)
    except ValueError as e:
        conn.close()
        return _err(404, str(e))
    except Exception as e:
        conn.close()
        return _err(500, f"입력 읽기 실패: {e}")

    logger.info(
        f"입력 읽기 완료: Phase1 HPO {len(phase1.get('positive_hpo', []))}개 | "
        f"Phase2 study {len(phase2_studies)}개 | Lab {len(lab_rows)}개"
    )

    # ── 4. scores 구성 (Soft LR p_i 값) ──────────────────────────────
    phase1_scores = _build_phase1_scores(phase1.get("positive_hpo", []))
    phase2_scores = _build_phase2_scores(phase2_studies)

    # ── 5. HPO 리스트 구성 ────────────────────────────────────────────
    history_hpo, xray_hpo = _build_hpo_lists(phase1, phase2_studies)

    # ── 6. lab raw → lab_numeric / lab_categorical 변환 ───────────────
    lab_numeric, lab_categorical = _build_lab_inputs(lab_rows)

    # ── 7. Step 0 — HPO Aggregator ────────────────────────────────────
    step0_ok    = True
    step0_error = None

    try:
        aggregated = aggregate_hpo(
            history_hpo      = history_hpo,
            xray_hpo         = xray_hpo,
            lab_numeric      = lab_numeric,
            lab_categorical  = lab_categorical,
            phase1_scores    = phase1_scores,
            phase2_scores    = phase2_scores,
        )
        logger.info(
            f"Step 0 완료: 통합 HPO {len(aggregated['patient_hpo'])}개 "
            f"(lab {len(lab_numeric)}개 수치)"
        )
    except Exception as e:
        # Step 0 실패 → Phase 1+2 HPO만으로 fallback
        step0_ok    = False
        step0_error = str(e)
        logger.warning(f"Step 0 실패 — fallback: {e}")
        aggregated = {
            "patient_hpo":   history_hpo + xray_hpo,
            "phase1_scores": phase1_scores,
            "phase2_scores": phase2_scores,
            "phase3_scores": {},
            "audit_trail":   [],
        }

    if not aggregated["patient_hpo"]:
        conn.close()
        return _err(400, "통합 HPO 없음 — Phase 1/2 positive HPO 확인 필요")

    # ── 8. LR 스코어링 ────────────────────────────────────────────────
    # 가중치는 phase5_lr_scorer 내부에서
    # 질환별 YAML lr_weights (A~G 카테고리) 사용
    scorer_input = {
        "patient_hpo":   aggregated["patient_hpo"],
        "phase1_scores": aggregated["phase1_scores"],   # Phase 1 confidence → p_i
        "phase2_scores": aggregated["phase2_scores"],   # Phase 2 prob → p_i
        "phase3_scores": aggregated["phase3_scores"],   # lab severity → p_i
    }

    try:
        scorer_result = _lr_scorer(scorer_input, context)
    except Exception as e:
        conn.close()
        return _err(500, f"LR 스코어러 오류: {e}")

    result_body   = scorer_result.get("body", {})
    listing       = result_body.get("listing", [])
    sub_threshold = result_body.get("sub_threshold", [])

    logger.info(
        f"LR 계산 완료: 평가 {result_body.get('total_evaluated', 0)}개 "
        f"→ Listing {len(listing)}개 (LR > {LR_THRESHOLD})"
    )
    if listing:
        logger.info(
            f"1위: {listing[0].get('disease_en')} "
            f"(LR={listing[0].get('lr_value')}, "
            f"ORPHA:{listing[0].get('orphacode')})"
        )

    # ── 9. input_hpo_used 구성 (phase1/phase2 분리 보관) ─────────────
    input_hpo_used = {
        "phase1_positive": phase1.get("positive_hpo", []),
        "phase1_negative": phase1.get("negative_hpo", []),
        "phase2_xray": [
            item
            for study in phase2_studies
            for item in study.get("xray_hpo_inferred", [])
        ],
    }

    # ── 10. input_data_meta 구성 ──────────────────────────────────────
    input_data_meta = {
        "input_collected_at": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "source_phase_executed_at": {
            "phase1": str(phase1.get("executed_at", "")),
            "phase2_studies": {
                str(s["study_id"]): str(s["executed_at"])
                for s in phase2_studies
            },
        },
        "count_summary": {
            "phase1_positive_count": len(phase1.get("positive_hpo", [])),
            "phase2_study_count":    len(phase2_studies),
            "lab_count":             len(lab_rows),
            "patient_hpo_count":     len(aggregated["patient_hpo"]),
            "total_evaluated":       result_body.get("total_evaluated", 0),
            "listed_count":          len(listing),
        },
        "step0": {
            "ok":    step0_ok,
            "error": step0_error,
        },
        "patient_context": patient,
        "lr_weights_source": "YAML lr_weights (카테고리 A~G, LR_pipeline_v2.docx 8장)",
    }

    # ── 11. DB 저장 ───────────────────────────────────────────────────
    inference_ms = int((time.time() - t_start) * 1000)
    try:
        _write_phase5(
            conn, session_id,
            input_hpo_used  = input_hpo_used,
            listing         = listing,
            sub_threshold   = sub_threshold,
            total_evaluated = result_body.get("total_evaluated", 0),
            inference_ms    = inference_ms,
            input_data_meta = input_data_meta,
            audit_trail     = aggregated.get("audit_trail", []),
            step0_log       = {
                "ok":             step0_ok,
                "error":          step0_error,
                "lab_converted":  sum(
                    1 for t in aggregated.get("audit_trail", [])
                    if t.get("source") == "lab"
                ),
                "micro_converted": sum(
                    1 for t in aggregated.get("audit_trail", [])
                    if t.get("source") == "micro"
                ),
                "patient_hpo_count": len(aggregated["patient_hpo"]),
                "source_breakdown": {
                    "history": sum(1 for t in aggregated.get("audit_trail", []) if t.get("source") == "history"),
                    "xray":    sum(1 for t in aggregated.get("audit_trail", []) if t.get("source") == "xray"),
                    "lab":     sum(1 for t in aggregated.get("audit_trail", []) if t.get("source") == "lab"),
                    "micro":   sum(1 for t in aggregated.get("audit_trail", []) if t.get("source") == "micro"),
                },
            },
        )
        logger.info(f"phase5_rare_disease_listing 저장 완료 | {inference_ms}ms")
    except Exception as e:
        conn.close()
        return _err(500, f"DB 저장 실패: {e}")

    conn.close()

    return _ok({
        "session_id":      session_id,
        "listing_count":   len(listing),
        "total_evaluated": result_body.get("total_evaluated", 0),
        "top_lr_score":    listing[0]["lr_value"]  if listing else None,
        "top_lr_orphacode": listing[0]["orphacode"] if listing else None,
        "patient_hpo":     len(aggregated["patient_hpo"]),
        "inference_ms":    inference_ms,
        "step0_ok":        step0_ok,
        "message":         "phase5_rare_disease_listing 저장 완료",
    })
