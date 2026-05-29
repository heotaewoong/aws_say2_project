# Phase 3 / Phase 4 — Aurora DB I/O 명세 (v4, owner-aligned)

**Version**: v4.0
**Date**: 2026-05-14
**Schema**: `rarelinkai` (Aurora PostgreSQL 16.4, ap-northeast-2)
**Scope**: lung_dx의 Phase 3 (Multimodal Scoring) + Phase 4 (LLM Rerank). 다른 phase는 각 phase 담당 팀원이 별도 정의.

---

## 0. v4 변경 요약 (이전 v1.0 대비)

| 항목 | v1.0 (이전) | v4.0 (현재) | 결정 근거 |
|---|---|---|---|
| 테이블 수 | Phase 3: 3개 / Phase 4: 2개 | **Phase 3: 1개 / Phase 4: 1개** | 팀원 v4 schema 채택 (메모리 효율) |
| 저장 패러다임 | 정규 컬럼 위주 | **JSONB 위주** | 팀원 v4 |
| 테이블 이름 | `phase3_disease_scores`, etc. | **`rarelinkai.phase3_integrated_ranking`, `rarelinkai.phase4_llm_rerank`** | 팀 컨벤션 |
| Run ID | `scoring_run_id` + `verification_run_id` | **`session_id`** (단일, phase 공유) | 팀원 v4 |
| Phase 4 KPI | `guard_rails_triggered` (TEXT[]) | **`agrees_with_top1` (BOOL) + `flagged_concerns` (JSONB)** | 팀원 v4 |
| LLM 추론 본문 | DB TEXT 전체 | **요약(TEXT) + 전체(S3 URI)** | 비용 효율 |
| 비용 추적 | 없음 | **`inference_cost_usd NUMERIC`** | 팀원 v4 |
| **Scoring 공식** | clip(weighted + W4 adj, 0, 1) | **그대로 유지 (owner 결정)** | 이전 의학 검증 |
| **W4(a-g) 7 보정** | 유지 | **그대로 유지 (owner 결정)** | 의학 evidence |
| **Disease 범위** | 105 | **104 (owner 결정, v9 갱신)** | v8 NO 36건 삭제 → v9 추가 Q22 CHD 1건 제거 (2026-05-19) |
| **Option E** | CheXpert 14만 | **CheXpert 14만 (owner 결정)** | xray_hpo_inferred는 DB에 있어도 lung_dx 무시 |
| **Positive HPO only** | (gap) | **Phase 1 positive만 수집 (owner 결정)** | negative HPO 사용 안 함 |
| **Two-stage screening** | 없음 | **사용 안 함 (owner 결정)** | single-stage 유지 |

---

## 1. 흐름 다이어그램

```
[EMR UI/UX] → Aurora DB ── rarelinkai.patient_profile          (EMR 팀)
                          ├── rarelinkai.clinical_note         (EMR 팀)
                          ├── rarelinkai.imaging_study         (EMR 팀)
                          ├── rarelinkai.lab_result            (EMR 팀 — severity 사전 계산)
                          ├── rarelinkai.raw_emr_bundle        (Phase 0)
                          ├── rarelinkai.phase1_hpo_extraction (Phase 1 팀)
                          └── rarelinkai.phase2_xray_processing(Phase 2 SooNet 팀)
                                       │
                                       │  READ (session_id로 join)
                                       ▼
                            ┌──────────────────────────────┐
                            │ lung_dx Phase 3 (owner)      │
                            │   - DiagnosticScorer         │
                            │   - 4축 (S/L/R/M) scoring    │
                            │   - W4(a-g) 보정              │
                            │   - 105 활성 질환 평가         │
                            │   - Option E (CheXpert 14)   │
                            │   - clip(0~1) total_score    │
                            └──────────────────────────────┘
                                       │   WRITE (single row)
                                       ▼
                            ┌──────────────────────────────┐
                            │ rarelinkai.phase3_integrated │
                            │   _ranking                   │
                            │     - scoring (JSONB)        │
                            │     - ranking (JSONB)        │
                            │     - scoring_process (JSONB)│
                            │     - modality_weights       │
                            │     - thresholds_bonus_config│
                            │     - unified_positive_hpo   │
                            └──────────────────────────────┘
                                       │   READ (same session_id)
                                       ▼
                            ┌──────────────────────────────┐
                            │ lung_dx Phase 4 (owner)      │
                            │   - Bedrock Claude rerank    │
                            │   - agrees_with_top1 판정     │
                            │   - flagged_concerns         │
                            └──────────────────────────────┘
                                       │   WRITE (single row)
                                       ▼
                            ┌──────────────────────────────┐
                            │ rarelinkai.phase4_llm_rerank │
                            │     - agrees_with_top1 (BOOL)│
                            │     - reranked (JSONB)       │
                            │     - flagged_concerns       │
                            │     - rank_changes           │
                            │     - reasoning_summary +    │
                            │       s3_reasoning_full      │
                            │     - inference_cost_usd     │
                            └──────────────────────────────┘
```

---

## 2. Phase 3 I/O 명세

### Input — Phase 3가 READ

| Table | OWN | Phase 3가 읽는 필드 | 사용 |
|---|---|---|---|
| `rarelinkai.patient_profile` | EMR 팀 | `patient_id`, `age_years`, `sex`, `smoking_status`, `occupation` | 환자 컨텍스트 |
| `rarelinkai.phase1_hpo_extraction` | Phase 1 팀 | `session_id`, `executed_at`, **`positive_hpo`** | S축 (Symptoms) 매칭 |
| `rarelinkai.phase2_xray_processing` | Phase 2 (SooNet) 팀 | `session_id`, `study_id`, `executed_at`, **`densenet_findings`**, `mask_quality_flag` | R축 (Radiology) 매칭 |
| `rarelinkai.lab_result` | EMR 팀 | `observation_category`, `loinc_code`, `test_name_ko/en`, `value_numeric/text`, `value_unit`, `reference_low/high`, `reference_ver`, `abnormal_flag`, `severity`, `micro_result`, `measured_at`, `lab_value_meta` | L축 + M축 매칭 |

### Phase 3 READ 명시적 제외 (owner 결정)

| 컬럼 | 위치 | 제외 이유 |
|---|---|---|
| `phase1_hpo_extraction.negative_hpo` | Phase 1 출력 | owner 결정: Phase 1에서 negative HPO 추출 안 함. Phase 3는 사용 안 함 |
| `phase2_xray_processing.xray_hpo_inferred` | Phase 2 출력 | **Option E**: CheXpert→HPO 변환은 의학 inference (CheXpert paper §4.2 + 표준 crosswalk 부재). lung_dx Phase 3는 densenet_findings만 READ |

### Output — Phase 3가 WRITE

**테이블**: `rarelinkai.phase3_integrated_ranking` (1 row / session)

**핵심 컬럼**:
- `session_id` UUID (PK)
- `unified_positive_hpo` JSONB — Phase 1 audit snapshot (positive only)
- `lab_anomalies` JSONB
- `modality_weights` JSONB — disease별 또는 default (owner SSOT)
- `thresholds_bonus_config` JSONB — owner W4(a-g) 7종
- `yaml_ssot_ver` **"v3_6"** (2026-05-19: sub_code_radiology_findings B 옵션 schema), `excel_db_ver` **"v9"** (2026-05-19: Q22 Congenital Heart Disease 제거 — 의학적 fact 기반)
- `evaluated_disease_count` = 104 (일반 53 + 기타 51, Q22 제거 후)
- **`scoring`** JSONB — top-K disease (owner 0-1 capped scores)
- **`ranking`** JSONB — rank list
- **`scoring_process`** JSONB — evidence + W4 adjustments trace
- `inference_time_ms`, `input_data_meta`

→ 상세 schema: [`phase3_phase4_schema_v4.sql`](phase3_phase4_schema_v4.sql)

---

## 3. Phase 4 I/O 명세

### Input — Phase 4가 READ

| Table | OWN | Phase 4가 읽는 필드 | 사용 |
|---|---|---|---|
| `rarelinkai.phase3_integrated_ranking` | **Phase 3 (lung_dx)** | `executed_at`, `scoring`, `ranking`, `scoring_process`, `modality_weights`, `thresholds_bonus_config` | 기본 ranking + 컨텍스트 |
| `rarelinkai.phase1_hpo_extraction` | Phase 1 팀 | `positive_hpo` | LLM 프롬프트 컨텍스트 |
| `rarelinkai.phase2_xray_processing` | Phase 2 팀 | `densenet_findings` | LLM 프롬프트 컨텍스트 |
| `rarelinkai.patient_profile` | EMR 팀 | `age_years`, `sex`, `smoking_status`, `occupation` | LLM 컨텍스트 |

### Output — Phase 4가 WRITE

**테이블**: `rarelinkai.phase4_llm_rerank` (1 row / session)

**핵심 컬럼**:
- `session_id` UUID (PK, Phase 3와 동일)
- `p3_executed_at` — Phase 3 결과 timestamp
- **`agrees_with_top1`** BOOL — KPI
- **`reranked`** JSONB — LLM 재정렬
- **`flagged_concerns`** JSONB — 안전 경고
- **`rank_changes`** JSONB — Phase 3→4 변동
- `reasoning_summary` TEXT — 한국어 요약 (DB)
- **`s3_reasoning_full`** TEXT — 전체 본문 S3 URI
- `llm_model`, `prompt_ver`, `input_tokens`, `output_tokens`, **`inference_cost_usd`**, `inference_time_ms`
- `input_data_meta` JSONB

→ 상세 schema: [`phase3_phase4_schema_v4.sql`](phase3_phase4_schema_v4.sql)

---

## 4. 형식 분류

| 항목 | 형식 |
|---|---|
| Phase 3/4 입력 (DB read) | PostgreSQL row (각 컬럼 타입) |
| Phase 3/4 처리 중 (메모리) | Python dataclass / dict |
| Phase 3/4 출력 (DB write) | **PostgreSQL row + JSONB 위주** |
| Phase 4 LLM 호출 (외부) | JSON (HTTP body) |
| Phase 4 전체 LLM 추론 본문 | S3 객체 (URI만 DB) |
| 사람 가독 출력 | ❌ Phase 3/4는 PDF/txt 직접 생성 안 함 (Final Report 팀 책임) |

---

## 5. Owner 결정 보존 매트릭스 (v4에서 모두 유지)

| Owner 결정 | v4 schema에서 어떻게 보존되는가 |
|---|---|
| Scoring 공식 (clip 0~1) | `scoring[*].total_score` JSONB에 0~1 값. 비-additive |
| W4(a-g) 7 보정 | `thresholds_bonus_config` JSONB + `scoring_process[*].adjustments[*].kind` |
| Disease 104 | `evaluated_disease_count = 104` 컬럼 + `scoring` 배열에 top-K (v9: Q22 제거 후) |
| Option E (CheXpert 14만) | Phase 3 READ contract에 명시 (xray_hpo_inferred 사용 안 함). `scoring_process[*].matched_chexpert_labels`는 raw text |
| YAML/Excel SSOT 가중치 | `modality_weights` JSONB + `yaml_ssot_ver/excel_db_ver` 컬럼 |
| GOLD 2026 J41/J43/J44 weights | YAML v3_6 SSOT에서 자동 적용 — scoring 결과에 반영 (v3_3 GOLD 2026 base + v3_6 sub_code_radiology_findings B 옵션 추가) |
| Sub-code 영상 매칭 (v3_6) | YAML `sub_code_radiology_findings` (10 카테고리/35 sub-codes) — Phase 2 candidate_icd_codes 우선 매칭. `scoring_process[*].evidence[*]`에 sub-code 매칭 trace 포함 가능 (JSONB 구조라 schema 변경 없음) |
| Positive HPO only | `unified_positive_hpo` 컬럼만 (negative 컬럼 자체 제외) |
| Single-stage | `stage1_filtered_count`/`stage2_full_lr_count` 컬럼 제외 |

---

## 6. 파일 일람 (본 폴더, v4)

| 파일 | 내용 |
|---|---|
| **`README_v4.md`** | 본 문서 (v4 명세) |
| **`phase3_phase4_schema_v4.sql`** | Aurora DB DDL (rarelinkai.phase3_integrated_ranking + phase4_llm_rerank) |
| **`phase3_output_example_v4.json`** | Phase 3 출력 예시 (CAP P00123) — 1 row JSON view |
| **`phase4_output_example_v4.json`** | Phase 4 출력 예시 (CAP P00123) — 1 row JSON view |
| (legacy) `README.md` | v1.0 — 참고용 보존 |
| (legacy) `phase3_phase4_schema.sql` | v1.0 정규화 schema — 참고용 보존 |
| (legacy) `phase3_input_*.json` `phase3_output_*.json` `phase4_*.json` | v1.0 예시 — 참고용 보존 |

---

## 7. 다음 단계 (lung_dx 코드 정합)

| 단계 | 작업 | 영향 받는 파일 |
|---|---|---|
| 1 | aurora_reader.py 갱신 — 4 테이블 SELECT (patient/phase1/phase2/lab) | `lung_dx/phase2_xray/aurora_reader.py` |
| 2 | chexpert_adapter.py — densenet_findings만 read, xray_hpo_inferred 무시 명시 | `lung_dx/phase3_multimodal/chexpert_adapter.py` |
| 3 | Phase 3 writer 함수 — DiagnosticScorer 결과 → JSONB INSERT | (신규) `lung_dx/phase3_multimodal/phase3_writer.py` |
| 4 | Phase 4 client + writer — Bedrock 호출 + phase4_llm_rerank INSERT | (신규) `lung_dx/phase4_llm_verify/phase4_writer.py` |
| 5 | regression test 갱신 | `scripts/test_phase3_v4.py`, `test_phase4_v4.py` |
| 6 | ARCHITECTURE.md / disease_registry docstring 갱신 | 문서 정합 |

---

## 8. 핵심 한 줄 요약

> **팀원 v4 schema 구조(이름·JSONB·session_id·KPI 컬럼) 채택 + Owner의 의학 결정(W4 보정·Option E·105 질환·positive HPO only·clip 0-1 scoring) JSONB 내부에 보존.**
> **두 영역이 깔끔히 분리되어 팀 컨벤션과 owner 의학 정직성이 동시 충족됨.**
