# Rare-Link AI · Phase별 입출력 매핑 문서

**버전:** v1.0 | **작성일:** 2026-05-11 | **기반:** 4-Layer DB 스키마 설계서 v1.0

---

## 전체 데이터 흐름

```
EMR (FHIR) ──→ Layer 0 ──→ Layer 1 ──→ Phase 1~4 ──→ Layer 2 ──→ Layer 3
                (원본)      (정규화)     (처리)        (결과)      (피드백)
```

---

## Phase 1: Symptom LLM (HPO 추출)

### 입력 (READ)

```json
{
  "source_tables": ["clinical_note"],
  "layer": "Layer 1 (canonical)",
  "fields": {
    "clinical_note": {
      "note_id": "UUID — 입력 추적용",
      "patient_id": "VARCHAR(64) — 환자 식별",
      "note_type": "VARCHAR(16) — chief_complaint, hpi, pe 등",
      "note_text_ko": "TEXT — LLM에 직접 입력되는 한국어 원문",
      "note_text_en": "TEXT — (옵션) 영문 번역본"
    }
  },
  "filter": "WHERE patient_id = :target AND note_type IN ('chief_complaint', 'hpi', 'pe')",
  "join": "없음 (단일 테이블)"
}
```

### 출력 (WRITE)

```json
{
  "target_table": "phase1_hpo_extraction",
  "layer": "Layer 2 (phase_io)",
  "fields": {
    "session_id": "UUID — diagnosis_session FK",
    "phase": "1 (고정)",
    "input_note_ids": "UUID[] — 입력으로 사용한 note_id 배열",
    "positive_hpo": [
      {
        "hpo": "HP:0002094",
        "label_en": "Dyspnea",
        "label_ko": "호흡곤란",
        "source_quote": "점진적 호흡곤란",
        "confidence": 0.92
      }
    ],
    "negative_hpo": [
      {
        "hpo": "HP:0002206",
        "label_ko": "흉막삼출 없음",
        "source_quote": "흉막삼출 소견 없음"
      }
    ],
    "llm_model": "claude-sonnet-4-20250514",
    "korean_dict_ver": "v1",
    "multilang_lex_ver": "multilingual_phenotype_lexicon_v1",
    "inference_time_ms": 2340,
    "executed_at": "2026-05-11T09:23:05+09:00"
  }
}
```

---

## Phase 2: X-ray UNet + DenseNet

### 입력 (READ)

```json
{
  "source_tables": ["imaging_study"],
  "layer": "Layer 1 (canonical)",
  "fields": {
    "imaging_study": {
      "study_id": "UUID — 영상 식별 (PK, phase2 출력에 FK로 저장)",
      "patient_id": "VARCHAR(64)",
      "modality": "VARCHAR(8) — 'CXR' 필터링",
      "view_position": "VARCHAR(8) — 'PA' 우선",
      "s3_uri_png": "TEXT — 원본 PNG (UNet + DenseNet 입력)"
    }
  },
  "filter": "WHERE patient_id = :target AND modality = 'CXR'",
  "join": "없음"
}
```

### 출력 (WRITE)

```json
{
  "target_table": "phase2_xray_processing",
  "layer": "Layer 2 (phase_io)",
  "fields": {
    "session_id": "UUID",
    "phase": "2 (고정)",
    "study_id": "UUID — imaging_study FK",

    "_comment_s3_paths": "모든 이미지는 S3에 저장, DB에는 URI만",
    "s3_original_full": "s3://say2-2team-bucket/Phase_2/imaging/{patient_id}/{study_id}/original/full_res.png",
    "s3_original_512": "s3://say2-2team-bucket/Phase_2/imaging/{patient_id}/{study_id}/original/resized_512.png",
    "s3_lung_mask_512": "s3://say2-2team-bucket/Phase_2/imaging/{patient_id}/{study_id}/masks/lung_mask_512.png",
    "s3_heart_mask_512": "s3://say2-2team-bucket/Phase_2/imaging/{patient_id}/{study_id}/masks/heart_mask_512.png",
    "s3_lung_masked_512": "s3://say2-2team-bucket/Phase_2/imaging/{patient_id}/{study_id}/masked/lung_masked_512.png",
    "s3_heart_masked_512": "(옵션)",
    "s3_overlay_viz_512": "s3://say2-2team-bucket/Phase_2/imaging/{patient_id}/{study_id}/overlay/viz_512.png",

    "_comment_heatmaps": "threshold 초과 양성 판정된 heatmap들 — 환자별(이벤트별) 저장",
    "s3_heatmaps": [
      {"finding": "Lung Opacity", "s3_uri": "s3://say2-2team-bucket/Phase_2/imaging/{patient_id}/{study_id}/heatmaps/lung_opacity.png"},
      {"finding": "Pleural Effusion", "s3_uri": "s3://say2-2team-bucket/Phase_2/imaging/{patient_id}/{study_id}/heatmaps/pleural_effusion.png"}
    ],

    "unet_model_ver": "unet-jsrt-v1",
    "lung_pixel_count": 145230,
    "heart_pixel_count": 42100,
    "ctr_estimate": 0.48,
    "mask_quality_flag": "good",

    "densenet_findings": [
      {"finding": "Lung Opacity", "prob": 0.87, "severity": "moderate"},
      {"finding": "Pleural Effusion", "prob": 0.12, "severity": "normal"},
      {"finding": "Cardiomegaly", "prob": 0.05, "severity": "normal"}
    ],
    "densenet_model_ver": "densenet121-chexpert-v3",

    "_comment_hpo_mapping": "DenseNet finding → HPO 매핑 (Phase 3가 직접 읽음)",
    "xray_hpo_inferred": [
      {"hpo": "HP:0002202", "from_finding": "Pleural Effusion", "prob": 0.12, "severity": "normal"},
      {"hpo": "HP:0001944", "from_finding": "Lung Opacity", "prob": 0.87, "severity": "moderate"}
    ],

    "inference_time_ms": 4500,
    "executed_at": "2026-05-11T09:23:10+09:00"
  }
}
```

### S3 저장 경로 표준

```
s3://say2-2team-bucket/Phase_2/imaging/{patient_id}/{study_id}/
├── original/
│   ├── full_res.png              # 원본 해상도
│   └── resized_512.png           # 512×512 (모델 입력)
├── masks/
│   ├── lung_mask_512.png         # UNet 폐 마스크 (binary)
│   └── heart_mask_512.png        # UNet 심장 마스크 (binary)
├── masked/
│   ├── lung_masked_512.png       # 원본 × 폐 마스크
│   └── heart_masked_512.png      # (선택)
├── heatmaps/
│   ├── lung_opacity.png          # threshold 초과 양성 heatmap
│   ├── pleural_effusion.png      # (해당되는 finding만)
│   └── ...
└── overlay/
    └── viz_512.png               # Front 표시용 컬러 오버레이
```

---

## Phase 3: Lab + 통합 LR Ranking

### 입력 (READ)

```json
{
  "source_tables": [
    "phase1_hpo_extraction",
    "phase2_xray_processing",
    "lab_result"
  ],
  "layers": "Layer 1 (lab_result) + Layer 2 (phase1, phase2)",
  "fields": {
    "phase1_hpo_extraction": {
      "positive_hpo": "JSONB — Phase 1에서 추출한 양성 HPO 목록",
      "negative_hpo": "JSONB — Phase 1에서 추출한 음성 HPO 목록"
    },
    "phase2_xray_processing": {
      "xray_hpo_inferred": "JSONB — DenseNet finding→HPO 매핑 결과",
      "densenet_findings": "JSONB — (참고용) 원본 14-finding 확률",
      "mask_quality_flag": "VARCHAR — 영상 가중치 신뢰도 조정용"
    },
    "lab_result": {
      "loinc_code": "VARCHAR(16) — LOINC 코드로 YAML SSOT 매칭",
      "value_numeric": "NUMERIC — 수치 결과",
      "abnormal_flag": "VARCHAR(8) — H/L/HH/LL/N",
      "severity": "VARCHAR(16) — normal/mild/moderate/critical"
    }
  },
  "external_input": {
    "yaml_ssot": "s3://say2-2team-bucket/scripts/lung_disease_profiles_v3_2.yaml",
    "rare_db": "s3://say2-2team-bucket/scripts/rare_disease_db_v7.json"
  },
  "filter": "WHERE session_id = :current_session",
  "join": "phase1·phase2는 session_id로, lab_result는 patient_id로"
}
```

### 출력 (WRITE)

```json
{
  "target_table": "phase3_integrated_ranking",
  "layer": "Layer 2 (phase_io)",
  "fields": {
    "session_id": "UUID",
    "phase": "3 (고정)",

    "lab_anomalies": [
      {"loinc": "1988-5", "name_ko": "CRP", "value": 7.2,
       "severity": "moderate", "hpo_mapped": "HP:0011227"}
    ],
    "lab_ref_ver": "lab_reference_ranges_v9_4",

    "unified_positive_hpo": "Phase 1 + Phase 2 + Lab HPO 통합 (중복 제거)",
    "unified_negative_hpo": "Phase 1 negative HPO 그대로",

    "modality_weights": {"S": 0.4, "L": 0.2, "R": 0.3, "M": 0.1},
    "yaml_ssot_ver": "lung_disease_profiles_v3_2",
    "rare_db_ver": "v7",

    "stage1_filtered_count": 30,
    "stage2_full_lr_count": 30,

    "ranking": [
      {
        "rank": 1,
        "orpha": "ORPHA:538",
        "name": "Lymphangioleiomyomatosis",
        "lr_score": 12.35,
        "breakdown": {"S": 4.2, "L": 2.1, "R": 5.8, "M": 0.25},
        "matched_hpo": ["HP:0002094", "HP:0002202"],
        "missing_critical_hpo": []
      }
    ],

    "inference_time_ms": 1200,
    "executed_at": "2026-05-11T09:23:12+09:00"
  }
}
```

---

## Phase 4: 검증 LLM (Rerank)

### 입력 (READ)

```json
{
  "source_tables": [
    "phase1_hpo_extraction",
    "phase2_xray_processing",
    "phase3_integrated_ranking",
    "patient_profile"
  ],
  "layers": "Layer 1 (patient_profile) + Layer 2 (phase1~3 전체)",
  "fields": {
    "phase1_hpo_extraction": "positive_hpo, negative_hpo 전체",
    "phase2_xray_processing": "densenet_findings, xray_hpo_inferred, mask_quality_flag",
    "phase3_integrated_ranking": "ranking (top 10), modality_weights, unified_*_hpo",
    "patient_profile": "age_years, sex, smoking_status, occupation (컨텍스트)"
  },
  "filter": "WHERE session_id = :current_session",
  "note": "Phase 4는 Phase 1~3 전체 결과를 컨텍스트로 LLM에 전달"
}
```

### 출력 (WRITE)

```json
{
  "target_table": "phase4_llm_rerank",
  "layer": "Layer 2 (phase_io)",
  "fields": {
    "session_id": "UUID",
    "phase": "4 (고정)",

    "agrees_with_top1": true,
    "reranked": [
      {"rank": 1, "orpha": "ORPHA:538", "confidence": "HIGH",
       "reason": "HPO 일치도 높고 Lab 소견 부합", "evidence_used": ["positive_hpo", "lab_anomaly"]}
    ],

    "flagged_concerns": ["lab CRP 정상범위 내인데 감염성 질환 ranking됨"],
    "reasoning_summary": "Phase 3 top1 LAM과 동의. 호흡곤란+영상소견+VEGF-D 상승이 일관됨.",
    "s3_reasoning_full": "s3://say2-2team-bucket/Phase_4/reasoning/{session_id}/full_trace.json",

    "llm_model": "claude-sonnet-4-20250514",
    "prompt_ver": "v3",
    "inference_time_ms": 8500,
    "executed_at": "2026-05-11T09:23:20+09:00"
  }
}
```

---

## RAG (Final Report)

### 입력 (READ)

```json
{
  "source_tables": [
    "phase4_llm_rerank",
    "phase3_integrated_ranking",
    "patient_profile",
    "rag_api_cache"
  ],
  "layers": "Layer 1 + Layer 2",
  "fields": {
    "phase4_llm_rerank": "reranked (top 후보 목록)",
    "phase3_integrated_ranking": "ranking, unified_positive_hpo",
    "patient_profile": "age_years, sex (리포트 컨텍스트)",
    "rag_api_cache": "캐시된 외부 API 응답 (TTL 내)"
  },
  "external_apis": [
    "Orphanet (질환 상세)",
    "PubCaseFinder (유사 케이스)",
    "PubMed E-utilities (최신 논문)",
    "Monarch Initiative (유전자-표현형 연결)",
    "ClinicalTrials.gov (진행 중 임상시험)"
  ]
}
```

### 출력 (WRITE)

```json
{
  "target_tables": ["final_report", "rag_api_cache"],
  "layer": "Layer 2 (phase_io)",
  "fields_final_report": {
    "session_id": "UUID",
    "diagnosis_json": {
      "diagnosis": ["..."],
      "genetic_test": {"recommended_genes": ["TSC1", "TSC2"]},
      "treatment": {"guideline": "...", "evidence_level": "A"},
      "insight": {"...": "..."},
      "next_steps": ["..."],
      "uncertainty": ["..."],
      "self_check": {"all_pmids_in_context": true}
    },
    "markdown_report": "의사가 보는 최종 Markdown 리포트",
    "rag_citations": [
      {"pmid": "38123456", "year": 2024, "title": "...", "evidence_level": 1, "snippet": "..."}
    ],
    "rag_apis_used": ["pubmed", "orphanet", "monarch", "pubcasefinder", "clinicaltrials"],
    "self_check": {"all_pmids_in_context": true, "negative_hpo_reflected": true},
    "llm_model": "claude-sonnet-4-20250514",
    "total_inference_time_ms": 15000,
    "generated_at": "2026-05-11T09:23:35+09:00"
  },
  "fields_rag_api_cache": {
    "cache_key": "pubmed:HP:0002094:2024",
    "api_name": "pubmed",
    "query_params": {"hpo": "HP:0002094", "year_from": 2020},
    "response_json": "{...}",
    "ttl_days": 7
  }
}
```

---

## Front-end (9-screen) 읽기 전용

### 입력 (READ)

```json
{
  "source_tables": "전체 (Layer 1 + Layer 2)",
  "endpoint": "GET /api/sessions/{session_id}/full",
  "reads": {
    "patient_profile": "이름, 나이, 성별, 흡연, 직업",
    "clinical_note": "증상 원문 (요약 표시)",
    "lab_result": "비정상 lab 목록",
    "phase1_hpo_extraction": "positive/negative HPO",
    "phase2_xray_processing": "S3 URI → presigned URL 변환, findings, heatmaps",
    "phase3_integrated_ranking": "ranking top N",
    "phase4_llm_rerank": "검증 결과, reasoning_summary",
    "final_report": "diagnosis_json, markdown_report, citations"
  },
  "security": {
    "s3_uri": "절대 직접 노출 금지 → presigned URL (TTL 5~15분)로 변환",
    "phi": "name_display, mrn 등 PHI는 인증된 사용자에게만"
  }
}
```

---

## Layer 3: Outcome History (의사 피드백)

### 입력 (WRITE — 의사가 UI에서 입력)

```json
{
  "target_tables": ["physician_feedback", "final_clinical_outcome"],
  "trigger": "의사가 진단 결과 리뷰 후 UI에서 제출",
  "fields_feedback": {
    "session_id": "UUID — 어떤 진단 세션에 대한 피드백인지",
    "physician_id": "의사 ID (Cognito sub)",
    "agreed_with_top1": "boolean — 시스템 top1에 동의하는지",
    "selected_diagnosis": "ORPHA code — 의사가 최종 선택한 진단",
    "override_reason": "시스템과 다를 때 사유",
    "ui_rating": "1~5",
    "reasoning_quality": "1~5"
  },
  "fields_outcome": {
    "confirmed_diagnosis": "확진된 ORPHA code",
    "confirmation_method": "genetic_test / biopsy / imaging / clinical",
    "time_to_diagnosis_days": "진단까지 걸린 일수",
    "was_in_top3": "시스템이 맞췄는지"
  },
  "usage": "모델 재학습 데이터셋으로 활용"
}
```

---

## 테이블 요약 (총 14개)

| Layer | 테이블 | 변경 정책 | 주 사용자 |
|-------|--------|----------|----------|
| 0 | `raw_emr_bundle` | Immutable | ETL |
| 0 | `fhir_bundle_archive` (기존) | Append-only | Ingest |
| 1 | `patient_profile` | Append-only | Front, Phase 전체 |
| 1 | `clinical_note` | Append-only | Phase 1 |
| 1 | `lab_result` | Append-only | Phase 3 |
| 1 | `imaging_study` | Append-only | Phase 2 |
| 2 | `diagnosis_session` | Status 업데이트 허용 | Orchestrator |
| 2 | `phase1_hpo_extraction` | Append-only | Phase 1 |
| 2 | `phase2_xray_processing` | Append-only | Phase 2 |
| 2 | `phase3_integrated_ranking` | Append-only | Phase 3 |
| 2 | `phase4_llm_rerank` | Append-only | Phase 4 |
| 2 | `final_report` | Append-only | RAG |
| 2 | `rag_api_cache` | TTL 기반 갱신 | RAG |
| 3 | `physician_feedback` | Append-only | 의사 UI |
| 3 | `final_clinical_outcome` | Append-only | 의사 UI |

---

## Phase 간 결합 규칙

> Phase N은 (a) Layer 1의 환자 데이터와 (b) Layer 2의 직전 Phase 결과만 읽는다.
> 자신의 Phase 결과는 Layer 2에 쓴다.
> 두 Phase 사이에 직접 함수 호출 또는 in-memory 공유는 금지.
