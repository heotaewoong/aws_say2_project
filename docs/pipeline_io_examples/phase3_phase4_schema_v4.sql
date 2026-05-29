-- ════════════════════════════════════════════════════════════════════════
-- Rare-Link AI — Phase 3 & Phase 4 Schema (v4, owner-aligned)
-- Version : v4.0      |  Date : 2026-05-14
-- Engine  : Aurora PostgreSQL 16.4 (ap-northeast-2)
-- Schema  : rarelinkai
-- ════════════════════════════════════════════════════════════════════════
--
-- 본 schema는 팀원 v1.2-final 구조(단일 테이블 + JSONB)를 채택하되,
-- Phase 3/4 owner의 의학적 결정을 내부에 보존합니다.
--
-- ── Owner Decisions (반드시 보존) ──────────────────────────────────────
--   ① Scoring 공식    : total = clip(weighted + W4 adjustments, 0, 1)
--                       — 0~1 capped, 비-additive
--   ② Modality weights: YAML/Excel v3_7/v9 SSOT (disease별 다름)
--                       — GOLD 2026 v1.3 기반 J41/J43/J44 재설계 (v3_3)
--                       — v3_6 sub_code_radiology_findings B 옵션 schema
--                         (10 카테고리/35 sub-codes, 2026-05-19)
--                       — v3_7 2025-26 신규 4 가이드라인 references 부착
--                         (CAP ATS 2026 40679934 / ILD Fleischner 2025 40758555 /
--                          PE AHA/ACC 2026 41712677 / TB WHO 2025 40388555;
--                          토큰 변경 0건, references 메타만, 2026-05-19)
--   ③ Disease 범위    : 104 (일반 53 + 기타 51, Q22 CHD 제거 후)
--                       — v8 NO 36건 삭제 → v9 추가 Q22 (CHD) 1건 제거 (의학적 fact, 2026-05-19)
--   ④ Option E        : Phase 2 = CheXpert 14 카테고리만, HPO 변환 X
--                       — xray_hpo_inferred 컬럼이 DB에 있어도 lung_dx는 무시
--   ⑤ W4(a-g) 7 보정  : critical_lab_bonus, news2_high_bonus,
--                       negative_pathognomonic, modality_redistribution,
--                       min_criteria_normalization, coverage_factor,
--                       severity_weighting
--   ⑥ Positive HPO only: Phase 1에서 negative_hpo 추출 안 함
--                       → unified_negative_hpo 컬럼 제외
--   ⑦ Single-stage     : two-stage screening 미사용
--                       → stage1_filtered_count, stage2_full_lr_count 제외
--
-- ── Teammate 구조 채택 ────────────────────────────────────────────────
--   • 테이블 이름: phase3_integrated_ranking, phase4_llm_rerank
--   • session_id UUID (phase별)
--   • JSONB 위주 저장 (메모리 효율)
--   • inference_time_ms, input_data_meta
--   • agrees_with_top1, flagged_concerns, reranked, rank_changes
--   • reasoning_summary + s3_reasoning_full 분리
--   • inference_cost_usd
-- ════════════════════════════════════════════════════════════════════════

CREATE SCHEMA IF NOT EXISTS rarelinkai;

-- ─────────────────────────────────────────────────────────────────────
-- READ CONTRACT (Phase 3 입력) — 다른 팀이 OWN, 본 파일은 명세만
-- ─────────────────────────────────────────────────────────────────────
--   rarelinkai.patient_profile
--     READ: patient_id, age_years, sex, smoking_status, occupation
--   rarelinkai.phase1_hpo_extraction
--     READ: session_id, executed_at, positive_hpo
--     SKIP: negative_hpo (owner 결정 — 추출 안 함)
--   rarelinkai.phase2_xray_processing
--     READ: session_id, study_id, executed_at, densenet_findings, mask_quality_flag
--     SKIP: xray_hpo_inferred (owner Option E — lung_dx Phase 3 무시)
--   rarelinkai.lab_result
--     READ: observation_category, loinc_code, test_name_ko/en,
--           value_numeric/text, value_unit, reference_low/high, reference_ver,
--           abnormal_flag, severity, micro_result, measured_at, lab_value_meta
--     NOTE: severity가 ingestion 단계에서 사전 계산되어 있음 (팀원 방식)
--
-- READ CONTRACT (Phase 4 입력)
--   rarelinkai.phase3_integrated_ranking (이 파일 정의)
--     READ: executed_at, scoring, ranking, scoring_process,
--           modality_weights, thresholds_bonus_config
--   + Phase 3와 동일한 상류 4개 테이블 (LLM 컨텍스트)
-- ════════════════════════════════════════════════════════════════════════


-- ════════════════════════════════════════════════════════════════════════
-- TABLE: phase3_integrated_ranking
--   Phase 3 OUTPUT — 1 session = 1 row
-- ════════════════════════════════════════════════════════════════════════

CREATE TABLE rarelinkai.phase3_integrated_ranking (
  -- ── Identification ────────────────────────────────────────────────
  session_id              UUID         PRIMARY KEY,
  phase                   SMALLINT     NOT NULL DEFAULT 3 CHECK (phase = 3),
  executed_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

  -- ── Phase 1 audit snapshot ────────────────────────────────────────
  -- owner: positive HPO only (negative 미수집)
  unified_positive_hpo    JSONB        NOT NULL,
  -- unified_negative_hpo  JSONB        ❌ 제외 — owner 결정

  -- ── Lab 이상치 audit ──────────────────────────────────────────────
  lab_anomalies           JSONB        NOT NULL DEFAULT '[]'::jsonb,
  lab_ref_ver             TEXT,                                  -- e.g., "mimic_iv_v3_0"

  -- ── Scoring 설정 (owner SSOT) ─────────────────────────────────────
  modality_weights        JSONB        NOT NULL,                 -- {"S":0.30,"L":0.25,"R":0.30,"M":0.15} disease별 또는 default
  thresholds_bonus_config JSONB        NOT NULL,                 -- owner W4(a-g) config: see below

  -- ── 버전 트래킹 (재현성) ───────────────────────────────────────────
  yaml_ssot_ver           TEXT         NOT NULL,                 -- "v3_7" (2026-05-19: 2025-26 신규 4 가이드라인 references 부착 — CAP/ILD/PE/TB; v3_6 = sub_code_radiology_findings B 옵션 schema)
  rare_db_ver             TEXT,                                  -- (희귀 사용 시)
  excel_db_ver            TEXT,                                  -- "v9" (2026-05-19: Q22 CHD 제거. owner 추가 컬럼)
  scorer_code_sha         VARCHAR(40),                           -- git SHA (owner 추가)

  -- ── 두 단계 screening 카운트 ──────────────────────────────────────
  -- stage1_filtered_count INT          ❌ 제외 — owner 결정 (two-stage 미사용)
  -- stage2_full_lr_count  INT          ❌ 제외
  evaluated_disease_count INT          NOT NULL DEFAULT 104,     -- owner: 104 활성 질환 (v9: Q22 제거 후)

  -- ── Scoring 결과 (3 JSONB) — owner 정직성 보존 ────────────────────
  scoring                 JSONB        NOT NULL,                 -- top-K disease score objects
  ranking                 JSONB        NOT NULL,                 -- [{"rank":1,"disease_key":"...","icd10":[...]}]
  scoring_process         JSONB        NOT NULL,                 -- evidence + W4 adjustments 통합 trace

  -- ── 운영 메타 ────────────────────────────────────────────────────
  inference_time_ms       INT,
  input_data_meta         JSONB        NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX idx_p3_executed
  ON rarelinkai.phase3_integrated_ranking (executed_at DESC);

CREATE INDEX idx_p3_scoring_disease
  ON rarelinkai.phase3_integrated_ranking
  USING GIN ((scoring -> 0 -> 'disease_key'));

COMMENT ON TABLE rarelinkai.phase3_integrated_ranking IS
  'Phase 3 출력 — 1 session = 1 row. 팀원 v4 구조 + owner 의학 결정 보존.';

COMMENT ON COLUMN rarelinkai.phase3_integrated_ranking.scoring IS
  'JSONB: top-K disease score objects. 각 element 구조:
   {
     "disease_key": "community_acquired_pneumonia",
     "name_en": "Community-acquired pneumonia",
     "name_kr": "지역사회획득폐렴",
     "category": "common",
     "icd10": ["J18.9", "J15.9"],
     "total_score": 0.8421,           -- owner: clip(weighted+adj, 0, 1)
     "base_score": 0.7600,            -- weighted sum 전
     "weighted_score": 0.7600,        -- modality_weights 적용 후, 보정 전
     "adjustments_total": 0.0821,     -- W4 합산 delta
     "confidence": "STRONG",          -- STRONG/MODERATE/WEAK
     "modality_scores": {"S":0.667,"L":0.785,"R":1.000,"M":0.333},
     "weights_applied": {"S":0.30,"L":0.25,"R":0.30,"M":0.15},
     "matched_count": 11,
     "total_criteria": 17
   }
   참고: total_score는 owner 공식 (0~1 capped). 팀원 v4의 비-cap additive 공식 *미사용*.';

COMMENT ON COLUMN rarelinkai.phase3_integrated_ranking.scoring_process IS
  'JSONB: per-disease scoring evidence + W4 adjustments 통합 trace. 각 element:
   {
     "disease_key": "community_acquired_pneumonia",
     "evidence": [
       {"modality":"symptoms","finding":"Fever","source_hpo":"HP:0001945","matched":true},
       {"modality":"radiology","finding":"consolidation","source_chexpert":"Consolidation","prob":0.78},
       {"modality":"lab","finding":"Markedly Elevated CRP","severity":"critical","severity_weight":2.0,"source_loinc":"1988-5"},
       {"modality":"micro","finding":"S.pneumoniae","source":"sputum_culture"}
     ],
     "adjustments": [
       {"kind":"critical_lab_bonus","delta":0.05,"reason":"CRP 184 critical"},
       {"kind":"news2_high_bonus","delta":0.03,"reason":"NEWS2=8 + infectious"},
       {"kind":"coverage_factor","delta":0.0,"reason":"sqrt(4/4)=1.0"}
     ],
     "matched_hpo_phase1": ["HP:0001945","HP:0012735"],
     "matched_chexpert_labels": ["Consolidation","Pneumonia","Lung Opacity"]
   }
   참고: Option E — matched_chexpert_labels는 14 카테고리 raw, xray_hpo_inferred 사용 안 함.';

COMMENT ON COLUMN rarelinkai.phase3_integrated_ranking.thresholds_bonus_config IS
  'JSONB: owner W4(a-g) 7 종 보정 config. 구조:
   {
     "W4a_critical_lab_bonus": 0.05,
     "W4b_news2_high_bonus": 0.03,
     "W4c_negative_pathognomonic": -0.10,
     "W4d_modality_redistribution": "enabled",
     "W4e_min_criteria_normalization": 3,
     "W4f_coverage_factor": "sqrt(active/patient)",
     "W4g_severity_weighting": {"critical": 2.0, "abnormal": 1.0}
   }
   참고: 팀원 v4의 mild/moderate/severe/critical bonus (additive) *미사용* — owner 의학 evidence W4 보정 채택.';

COMMENT ON COLUMN rarelinkai.phase3_integrated_ranking.unified_positive_hpo IS
  'JSONB: Phase 1 positive HPO audit snapshot. owner 결정 — negative HPO 미수집.';


-- ════════════════════════════════════════════════════════════════════════
-- TABLE: phase4_llm_rerank
--   Phase 4 OUTPUT — 1 session = 1 row (팀원 schema 그대로 채택)
-- ════════════════════════════════════════════════════════════════════════

CREATE TABLE rarelinkai.phase4_llm_rerank (
  -- ── Identification ────────────────────────────────────────────────
  session_id              UUID         PRIMARY KEY,
  phase                   SMALLINT     NOT NULL DEFAULT 4 CHECK (phase = 4),
  executed_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

  -- ── Phase 3 source link ───────────────────────────────────────────
  p3_executed_at          TIMESTAMPTZ  NOT NULL,
  -- session_id가 Phase 3와 동일 (같은 case 한 흐름)

  -- ── 핵심 결과 ────────────────────────────────────────────────────
  agrees_with_top1        BOOLEAN      NOT NULL,                 -- Phase 3 top1과 동의 여부 (KPI)
  reranked                JSONB        NOT NULL,                 -- LLM 재정렬된 ranking
  flagged_concerns        JSONB        NOT NULL DEFAULT '[]'::jsonb, -- 안전 경고 / 감별 권고
  rank_changes            JSONB        NOT NULL DEFAULT '[]'::jsonb, -- Phase 3→4 rank 변동

  -- ── LLM 추론 ─────────────────────────────────────────────────────
  reasoning_summary       TEXT         NOT NULL,                 -- 한국어 요약 (DB 내)
  s3_reasoning_full       TEXT,                                  -- S3 URI (전체 추론 본문, 비용 효율)

  -- ── LLM 호출 메타 ─────────────────────────────────────────────────
  llm_model               TEXT         NOT NULL,                 -- "claude-sonnet-4-6"
  prompt_ver              TEXT         NOT NULL,                 -- "v4_..."
  input_tokens            INT          NOT NULL,
  output_tokens           INT          NOT NULL,
  inference_cost_usd      NUMERIC(10,6),                         -- 비용 추적
  inference_time_ms       INT,

  -- ── 운영 메타 ────────────────────────────────────────────────────
  input_data_meta         JSONB        NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX idx_p4_executed
  ON rarelinkai.phase4_llm_rerank (executed_at DESC);

CREATE INDEX idx_p4_p3_executed
  ON rarelinkai.phase4_llm_rerank (p3_executed_at DESC);

CREATE INDEX idx_p4_agrees
  ON rarelinkai.phase4_llm_rerank (agrees_with_top1);

COMMENT ON TABLE rarelinkai.phase4_llm_rerank IS
  'Phase 4 출력 — 1 session = 1 row. 팀원 v4 schema 그대로 채택.
   reranked/scoring 본문은 owner Phase 3 결과 기반 (0~1 capped scores).';

COMMENT ON COLUMN rarelinkai.phase4_llm_rerank.reranked IS
  'JSONB: LLM 재정렬된 ranking. 각 element:
   {
     "rank": 1,
     "disease_key": "community_acquired_pneumonia",
     "name_kr": "지역사회획득폐렴 (CAP)",
     "icd10": ["J18.9"],
     "score_phase3": 0.8421,         -- owner Phase 3 score (0~1)
     "score_phase4": 0.8421,         -- LLM 조정 후 (보통 유지, 감별 시 조정)
     "delta": 0.0,
     "reason": "ATS/IDSA CAP 2019 진단 기준 충족..."
   }';

COMMENT ON COLUMN rarelinkai.phase4_llm_rerank.flagged_concerns IS
  'JSONB array: 안전 경고 / 감별 권고 / 추가 검사. 각 element:
   {
     "disease_key": "viral_pneumonia",
     "concern": "PCT 2.8 > 0.5 → 세균성 우세, 바이러스성 가능성 감소",
     "severity": "info",          -- info/warning/critical
     "reference": "Schuetz 2017 PMID 29025194"
   }';

COMMENT ON COLUMN rarelinkai.phase4_llm_rerank.s3_reasoning_full IS
  'S3 URI for full LLM reasoning text (비용 효율). DB에는 reasoning_summary만, 전체 본문은 S3.
   예: "s3://say2-2team-bucket/llm_reasoning/2026/05/14/session_<uuid>.txt"';


-- ════════════════════════════════════════════════════════════════════════
-- 검증 쿼리 sample
-- ════════════════════════════════════════════════════════════════════════

-- 1) 환자 session의 Phase 3 → Phase 4 흐름
-- SELECT
--   p3.session_id,
--   p3.executed_at AS p3_executed,
--   p4.executed_at AS p4_executed,
--   p3.evaluated_disease_count,
--   jsonb_array_length(p3.scoring) AS top_k_count,
--   p4.agrees_with_top1,
--   p4.inference_cost_usd
-- FROM rarelinkai.phase3_integrated_ranking p3
-- LEFT JOIN rarelinkai.phase4_llm_rerank p4 USING (session_id)
-- WHERE p3.session_id = '<uuid>';

-- 2) Phase 3 Top 1 disease 추출 (JSONB query)
-- SELECT
--   scoring -> 0 ->> 'disease_key'  AS top1_disease,
--   (scoring -> 0 ->> 'total_score')::numeric AS top1_score,
--   scoring -> 0 ->> 'confidence'   AS top1_conf
-- FROM rarelinkai.phase3_integrated_ranking
-- WHERE session_id = '<uuid>';

-- 3) Phase 4 rerank가 Phase 3와 다른 케이스
-- SELECT
--   p4.session_id, p4.executed_at,
--   jsonb_array_elements(p4.rank_changes) ->> 'disease_key' AS disease,
--   (jsonb_array_elements(p4.rank_changes) ->> 'delta')::int AS rank_delta
-- FROM rarelinkai.phase4_llm_rerank p4
-- WHERE NOT p4.agrees_with_top1;

-- 4) W4 보정이 적용된 케이스 통계
-- SELECT
--   jsonb_array_elements(scoring_process) -> 'adjustments' AS adjustments
-- FROM rarelinkai.phase3_integrated_ranking
-- WHERE jsonb_array_length(scoring_process) > 0;
