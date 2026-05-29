# RAG 구현 보고서
## Rare-Link AI — 희귀 폐질환 진단 보조 시스템

> **최종 확정일**: 2026-04-29 (회의)
> **개정일**: 2026-05-04 (확정 문서 v1.0 반영)
> **참여자**: 권미라, 배기태, 허태웅
> **이 문서의 역할**: 확정 문서(`최종프롬프트_API시스템_확정문서_v1.docx`) 기준으로 RAG 파이프라인 전체 동작을 코드 레벨로 설명한 구현 보고서.
>
> **세부 명세는** `data/RAG_프롬프트_API_정리서.md` 참조.

---

## 1. RAG란 무엇인가

RAG(Retrieval-Augmented Generation)는 LLM이 답변을 생성할 때 외부 지식(논문·임상 DB·증례 등)을 실시간 검색해서 컨텍스트로 주입하는 방식이다.

```
[일반 LLM]
  질문 → Claude → 답변 (학습 데이터만 사용)

[RAG]
  질문 → 외부 DB 5개 검색 → 검색 결과 + 환자 데이터 + 질문 → Claude → 근거 있는 답변
```

이 프로젝트에서 RAG는 **"환자 멀티모달 데이터 → HPO 변환 → 일반/희귀 분리 랭킹 → 5개 API 병렬 검색 → 교차검증 → JSON 진단 보조 소견서 생성"**에 사용된다.

---

## 2. 전체 파이프라인 5단계 (확정)

```
환자 입력 (X-ray + 혈액검사 + 임상소견 + 환자기본정보 + Vital + Micro)
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│ ① Phase 1~3 — 멀티모달 → HPO 프로파일                   │
│   Phase 1: 증상(텍스트) → Bedrock Haiku → HPO + 근거    │
│   Phase 2: X-ray → SooNet (DenseNet-121) → HPO%         │
│   Phase 3: Lab + Vital + Micro → Rule-based → HPO       │
│   → Positive HPO + Negative HPO + LR 스코어링 입력       │
└─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│ ② 스코어링 분기 (병렬 2 트랙)                            │
│   ├─ 일반/기타 폐질환 DB → Rule-based → Top 10 랭킹      │
│   └─ 희귀 DB (Orphanet 4335) → LIRICAL LR → 희귀 리스팅  │
└─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│ ③ Phase 4 — LLM이 Phase 3 결과 정리                     │
│   Bedrock Haiku로 일반 Top 10 + 희귀 리스팅을 정리       │
│   → 최종 Top 3 도출 (일반/희귀 통합 랭킹)                │
└─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│ ④ RAG 트리거 (Top 3 기준 5개 API 병렬 호출)              │
│   조건부 수집 로직:                                      │
│   - 희귀 리스팅 있음        → Orphanet + PubCaseFinder   │
│   - 리스팅 없음 + Top3 OrphaCode 있음 → 위 + 교차검증    │
│   - Top 3 전부 일반         → PubMed + Monarch + CT만    │
│                                                          │
│   API 5개 (asyncio 병렬):                                │
│     Orphanet · PubCaseFinder · Monarch · PubMed · CT     │
│   + 내부 DB Top 3 교차검증                               │
│   → CARE JSON 컨텍스트                                   │
└─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│ ⑤ LLM 소견서 생성                                       │
│   8개 섹션 유저 프롬프트 + 시스템 프롬프트               │
│   → Bedrock Claude Sonnet 3.5                            │
│   → JSON only (recommendation + clinical_notes)          │
│      ※ Markdown 절대 금지                                │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 디렉토리 구조 (확정 후)

```
aws_say2_project_vision/
│
├── rag_pipeline.py                  ← 5단계 오케스트레이터 ✅ 확정 v1.0 구현 완료
├── soo_net.py                       ← X-ray → HPO (DenseNet-121)
│
├── data/
│   ├── en_product4.xml              ← Orphanet HPO Annotation 원본 (질환명, DisorderType)
│   ├── en_product6.xml              ← Orphanet 유전자 + association_type ★ 2026-05-04 추가
│   ├── en_product9_ages.xml         ← Orphanet 발병연령 + 유전 양식 ★ 2026-05-04 추가
│   ├── en_product9_prev.xml         ← Orphanet 유병률 (Point prevalence) ★ 2026-05-04 추가
│   ├── orphadata_weighted.csv       ← LIRICAL용 빈도 가중치 (4335 질환, 115878행)
│   ├── lung_disease_profiles_v2.yaml ← 일반/기타 폐질환 Rule-based 프로파일
│   ├── 최종프롬프트_API시스템_확정문서_v1.docx ← 절대 기준
│   ├── RAG_프롬프트_API_정리서.md   ← 명세 (이 문서의 짝)
│   └── RAG_구현_보고서.md           ← 이 파일
│
├── model/
│   └── chexnet_unet_crop_best.pth   ← SooNet 학습 가중치
│
└── rag/                             ← RAG 모듈 (5개 API + 스코어러)
    ├── __init__.py
    ├── bedrock_extractor.py         ← Phase 1: 증상 텍스트 → HPO (Haiku)
    ├── lab_rules.py                 ← Phase 3: Lab → HPO (Rule-based)
    ├── lirical_scorer.py            ← ② 희귀: LIRICAL LR 스코어링
    ├── general_disease_scorer.py    ← ② 일반: Rule-based Top 10 ★ 회의 확정
    ├── knowledge_base.py            ← Orphanet XML → CSV 파서
    │
    ├── pubcasefinder.py             ← API 1: HPO list → 후보 질환 (target=omim, enrich 포함)
    ├── orphanet_fetcher.py          ← API 2: OrphaCode → 유전자/HPO/유병률/발병연령 ✅ 완료
    ├── monarch_fetcher.py           ← API 3: 인과 유전자 + 질환명/설명 + HPO 매핑 ✅ 완료
    ├── pubmed_fetcher.py            ← API 4: 질환명 → 케이스리포트 3건
    ├── clinicaltrials_fetcher.py    ← API 5: 질환명 → 임상시험 3건
    │
    ├── ragas_eval.py                ← PMID 환각 체크
    └── valid/                       ← 검증 스크립트
        ├── fetch_mimic_patient.py
        └── run_rag_test.py
```

★ 표시는 확정 문서 기준으로 **신규 구현 필요**한 항목.

---

## 4. 단계별 코드 동작 상세

### 4-1. ① Phase 1~3 — 멀티모달 HPO 변환 (`step1_get_hpo`)

세 가지 소스에서 HPO 코드를 추출하여 통합한다.

```python
# Phase 1: 증상(텍스트) → HPO
nlp_result = self.hpo_extractor.extract_hpo(symptom_text)
pos_nlp = nlp_result["positive_hpo"]   # 환자에게 있는 증상
neg_nlp = nlp_result["negative_hpo"]   # 환자에게 없다고 명시된 증상

# Phase 2: X-ray → HPO
xray_preds = self.vision.predict(xray_path)  # 14개 라벨별 (확률, HPO)
xray_hpos = [hpo for label, (prob, hpo) in xray_preds.items()
             if prob >= 0.3 and "N/A" not in hpo]

# Phase 3: Lab + Vital + Micro → HPO
lab_hpos = lab_to_hpo(lab_results, vital_results, micro_results)

# 통합 (Negative 우선)
neg_set = set(neg_nlp)
all_positive = list(set(xray_hpos + pos_nlp + lab_hpos) - neg_set)
neg_clean    = list(neg_set)
```

**source 구분 (확정 문서 §3.2 - 3번 섹션 요구)**:
- Positive HPO는 `symptom` / `xray` 출처를 명시
- Negative HPO는 `symptom`에서만 추출 (X-ray·Lab은 Negative 추출 불가)

```python
pos_hpos_with_source = [
    {"hpo": h, "source": "symptom"} for h in pos_nlp
] + [
    {"hpo": h, "source": "xray"} for h in xray_hpos
] + [
    {"hpo": h, "source": "lab"} for h in lab_hpos
]
```

### 4-2. ② 스코어링 분기 — 일반 + 희귀 동시 진행 (`step2_dual_scoring`)

**중요 변경점**: 4월 27일까지는 LIRICAL 한 트랙이었는데, 4월 29일 회의에서 **일반/기타 DB와 희귀 DB를 완전히 분리한 듀얼 트랙**으로 확정.

```python
# 트랙 A: 일반/기타 폐질환 (Rule-based Top 10)
general_ranking = rank_general_diseases(
    positive_hpos=hpo_data["positive_hpo"],
    xray_preds=hpo_data["xray_detail"],
    lab_results=hpo_data["lab_results"],
    top_k=10,
)
# → 폐렴, COPD, 결핵 등 일반적인 폐질환 Top 10

# 트랙 B: 희귀질환 (LIRICAL LR Listing)
rare_listing = rank_diseases(
    positive_hpos=hpo_data["positive_hpo"],
    negative_hpos=hpo_data["negative_hpo"],
    disease_database=self.disease_db,  # Orphanet 4335
    top_k=10,
)
# → LR 임계치 통과한 희귀질환만 (없으면 빈 리스트)
```

**왜 분리했나**:
- 일반 질환은 빈도 높고 LIRICAL의 sparse HPO 매칭으로는 잘 안 잡힘
- 희귀질환은 Orphanet 빈도 가중치를 곱셈으로 누적해야 정확
- 두 트랙을 따로 돌려야 일반 진단 + 희귀 스크리닝을 모두 커버

### 4-3. ③ Phase 4 — LLM이 랭킹 결과 정리 (`step3_phase4_organize`)

확정 문서의 ③단계. LLM이 일반 Top 10 + 희귀 리스팅을 받아서 최종 Top 3를 정리.

```python
def step3_phase4_organize(self, general_ranking, rare_listing, hpo_data):
    """
    Bedrock Haiku 호출:
      - 입력: 일반 Top 10 + 희귀 리스팅 + HPO 프로파일
      - 출력: 정리된 Top 3 (일반·희귀 통합 우선순위)
      - 규칙: 희귀질환은 LR 점수 낮아도 Top 3 후보로 유지 (감별 진단 가치)
    """
    # 시스템 프롬프트: "두 랭킹을 임상적 우선순위에 따라 통합하여 Top 3 도출"
    # 출력: [{rank, disease_name, orpha_code|null, score, source: "general"|"rare"}]
```

**왜 LLM이 정리하나**: 일반 Top 10과 희귀 리스팅은 점수 체계가 다름 (Rule-based score vs LR). 단순 정렬로는 통합 불가. LLM이 임상 맥락(증상 + Lab + 희귀 가능성)을 보고 Top 3 우선순위를 정하는 게 회의 결론.

### 4-4. ④ RAG 트리거 — 5개 API 병렬 호출 (`step4_rag_collect`)

**조건부 수집 로직 (확정 문서 §3.3)**:

```python
def step4_rag_collect(self, top3, rare_listing):
    has_rare_listing = len(rare_listing) > 0
    top3_has_orpha   = any(d.get("orpha_code") for d in top3)

    if has_rare_listing:
        # 케이스 A: 희귀 리스팅 있음
        apis = ["orphanet", "pubcasefinder", "monarch", "pubmed", "clinicaltrials"]
        section6_text = format_rare_listing(rare_listing)

    elif top3_has_orpha:
        # 케이스 B: 리스팅은 없지만 Top 3에 OrphaCode 있음 (교차검증)
        apis = ["orphanet", "pubcasefinder", "monarch", "pubmed", "clinicaltrials"]
        section6_text = "해당 없음"

    else:
        # 케이스 C: Top 3 전부 일반 질환
        apis = ["monarch", "pubmed", "clinicaltrials"]   # Orphanet/PCF 스킵
        section6_text = "해당 없음"

    # 병렬 호출
    rag_context = await asyncio.gather(*[
        call_api(api, top3) for api in apis
    ])
    return rag_context, section6_text
```

**5개 API 역할**:

| API | 입력 | 출력 |
|-----|------|------|
| **PubCaseFinder** | HPO list | 후보 질환 disease_id/disease_name/score/pmid_list/hpo_list (target=omim, `enrich_pcf_results()`로 Monarch·PubMed 보강) |
| **Orphanet** | OrphaCode (Top 3) | 유전자(association_type), HPO 빈도, 유병률, 발병연령 (en_product6+9_ages+9_prev 로컬 XML) |
| **Monarch** | 질환명/OrphaCode (Top 3) | causal_genes, HPO 매핑 |
| **PubMed** | 질환명 (Top 3) | 케이스리포트 PMID + abstract × 3건 (eFetch retmode=xml) |
| **ClinicalTrials.gov** | 질환명 (Top 3) | RECRUITING × 3건 (NCT ID + Phase) |

**내부 DB 교차검증**: API 결과를 받은 뒤 로컬 `orphadata_weighted.csv`와 대조하여:
- 일치 → "DB·API 교차검증 일치"
- 불일치 → "DB·API 불일치 — 추가 확인 필요"

이 결과가 LLM 프롬프트의 §7번 섹션(`internal_db_context`)에 들어감.

### 4-5. ⑤ LLM 소견서 생성 (`step5_generate_report`)

**시스템 프롬프트** — `rag_pipeline.py` `SYSTEM_PROMPT` 그대로:

```
You are an elite AI diagnostician specializing in pulmonary and rare diseases.
You synthesize multimodal patient data and RAG-retrieved evidence to generate
a professional diagnostic support report.
Your role is to support physician decision-making, not to make final diagnoses.
Write your final report clearly, logically, and professionally.
Ensure your final output is written in Korean as requested by the clinical team.

[strict rules]
1. Do not make assumptions or definitive conclusions without evidence.
   (근거 없는 추측이나 단정을 포함하지 않습니다.)
2. All claims must be grounded in the provided RAG data or established medical guidelines.
   (모든 주장은 제공된 RAG 데이터 또는 공인된 의학 가이드라인에 근거해야 합니다.)
3. If a rare disease (OrphaCode) appears in Top 3, MDT referral is mandatory.
   (희귀질환이 Top 3 이내에 있으면 MDT 협진 권고는 필수입니다.)
4. Prioritize diseases cross-validated by both Local DB and Global API.
   Provide a clinical rationale for the primary diagnosis.
   (로컬 DB·글로벌 API 양쪽 교차검증 질환을 우선순위로 삼고 임상 근거를 제시합니다.)
5. Differential Diagnosis: Use the patient's negative findings to logically explain
   why certain candidate diseases should be ruled out.
   (감별 진단: Negative HPO를 활용하여 배제 근거를 논리적으로 설명합니다.)
6. Case Comparison: Compare and contrast the patient's current state with PubMed case reports.
   (사례 비교: PubMed 케이스리포트와 현재 환자 상태를 비교·대조합니다.)
7. Actionable Alternatives: Synthesize ClinicalTrials data to recommend clinical trial opportunities.
   (실행 가능한 대안: ClinicalTrials 데이터를 종합하여 임상시험 참여 기회를 권고합니다.)

[Output Format Rules]
Output must strictly follow the JSON structure below.
Do not include any text outside this JSON.

{
  "recommendation": {
    "immediate_workup": [...],
    "specialist_referral": [...],
    "treatment_guideline": ["[Disease 1] ...", "[Disease 2] ...", "[Disease 3] ..."],
    "genetic_test": [...],
    "additional_lab": [...]
  },
  "clinical_notes": {
    "summary": "...",
    "top1_reasoning": "...",
    "differential_note": "...",
    "rag_evidence": "...",
    "case_comparison": "...",
    "epidemiology_note": "...",
    "disclaimer": "AI 결과는 진단 보조이며 최종 진단은 주치의의 임상 판단과 추가 검사 결과를 종합하여 확정합니다."
  }
}

[Writing Principles]
1. summary: 나이/성별/주호소/Lab 이상치. MRN 절대 포함 금지.
2. top1_reasoning: Positive HPO + Negative HPO + 이상 Lab 수치 전부 명시.
3. rag_evidence: genes_from_orphadata, phenotypes, prevalence, age_of_onset, causal_genes 모두 인용.
   일치 → "DB·API 교차검증 일치" / 불일치 → "DB·API 불일치 — 추가 확인 필요"
4. treatment_guideline: 공식 가이드라인 기반 항목, [질환명] 접두어, 우선순위 정렬.
5. 희귀질환 Top 3 내 → specialist_referral에 MDT 권고 필수.
6. differential_note: Top 2~3 감별. 확률 낮아도 희귀 플래그 항목 반드시 포함.
7. genetic_test: association_type에 "Disease-causing" 포함 유전자 필수.
   Monarch + Orphadata 일치 시 "(복수 소스 확인)" 추가.
8. case_comparison: PubMed 케이스와 공통점/차이점/유전자 검사 시사점.
9. epidemiology_note: 일반 질환은 빈 문자열. 희귀만 Orphanet prevalence/onset/inheritance 기술.
10. disclaimer 문구 고정. 임의 변경 금지.
```

**유저 프롬프트** — `rag_pipeline.py` `_build_user_prompt()` 실제 구조:

```
You are analyzing a patient case using multimodal clinical data and
RAG-retrieved evidence from both internal DB and external APIs.
Based on the structured data below, generate a comprehensive diagnostic
support report following the output format specified in the system prompt.

[한국어 해석]
아래 다중 모달리티 임상 데이터와 내부 DB·외부 API RAG 수집 결과를 분석하여,
시스템 프롬프트에 명시된 출력 형식에 따라 진단 보조 소견서를 작성하십시오.

=========================================
=== 1. 환자 기본정보 ===
{"name": ..., "age": ..., "sex": ..., "visit_date": ...,
 "visit_type": ..., "chief_complaint": ..., "allergy": ...}
※ MRN 절대 미포함

=== 2. 증상 원문 ===
- Positive Findings (양성 증상): {symptom_text}
- Negative Findings (음성 소견): {negative_text}

=== 3. HPO 프로파일 ===
Positive HPO:
  - HP:XXXXXXX (symptom)
  - HP:XXXXXXX (xray)
  - HP:XXXXXXX (lab)
※ source 구분: symptom / xray / lab

Negative HPO (증상에서만):
  - HP:XXXXXXX

=== 4. Lab 수치 ===
  - {검사항목}: {수치}  ...

=== 5. 일반/기타 폐질환 랭킹 Top 10 (로컬 DB 기반) ===
 1. {disease_name} (score={score:.3f})
 ...

=== 6. 희귀폐질환 리스팅 (로컬 DB 기반) ===
{rare_listing 또는 "해당 없음"}

=== 7. 내부 DB 정보 — Top 3 교차검증용 ===
Top 1 교차검증: {summary}
Top 2 교차검증: {summary}
Top 3 교차검증: {summary}
※ 외부 API 결과와 대조하여 일치/불일치 여부를 rag_evidence에 반드시 명시할 것.

=== 8. RAG 검색 결과 (외부 API) ===
--- Top 1: {disease_name} (ORPHA:{orpha_code}) ---
[Orphanet]
- 유전자: {genes} ({association_type})
- Very frequent / Frequent HPO: {hpo_list}
- 유병률: {prevalence}
- 발병연령: {age_of_onset}
- 유전 양식: {inheritance}

[Monarch]
- 인과 유전자: {causal_genes}
- Orphanet 교차검증: {cross_text}

[PubMed 케이스리포트]
  - PMID:{pmid} | {title} ({pubdate})
    {abstract[:300]}...

[PubCaseFinder]  ← Case A/B Top 1에서만
  - {disease_name} (score={score:.3f}, genes={genes})

[ClinicalTrials (RECRUITING)]
  - NCT:{nct_id} | {title[:80]}
    Phase:{phase} | Status:{status}

--- Top 2: ... ---
--- Top 3: ... ---
=========================================

위 데이터를 종합하여 규정된 JSON 형식으로 출력하십시오.
```

**Bedrock 호출** (`rag_pipeline.py` line 602~612 그대로):
```python
self.bedrock.invoke_model(
    modelId="apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": 2048,    # 확정 문서 §2.2
        "temperature": 0.0,    # 확정 문서 §2.2
    }),
)
raw_text = json.loads(resp["body"].read())["content"][0]["text"]
result   = json.loads(raw_text)  # JSON only — Markdown 없음
```

**출력 파싱**: `_parse_json_response()` → ```json 코드블록 제거 후 `json.loads()`. 실패 시 `{"raw_output": ..., "error": "json_parse_failed"}` 반환.

---

## 5. 확정 출력 JSON 구조

```json
{
  "recommendation": {
    "immediate_workup": ["..."],
    "specialist_referral": ["...(MDT 권고 — 희귀 Top 3 시 필수)"],
    "treatment_guideline": [
      "[Disease 1] ...",
      "[Disease 2] ...",
      "[Disease 3] ..."
    ],
    "genetic_test": ["..."],
    "additional_lab": ["..."]
  },
  "clinical_notes": {
    "summary": "MRN 절대 미포함. 나이/성별/주호소/Lab 이상치만",
    "top1_reasoning": "Pos HPO + Neg HPO + Lab 모두 명시",
    "differential_note": "Top 2~3 (희귀 플래그 우선)",
    "rag_evidence": "DB·API 교차검증 결과 명시",
    "case_comparison": "PubMed 케이스 vs 환자 대조",
    "epidemiology_note": "희귀만 (일반은 빈 문자열)",
    "disclaimer": "AI 결과는 진단 보조이며 최종 진단은 주치의의 임상 판단과 추가 검사 결과를 종합하여 확정합니다."
  }
}
```

---

## 6. 4월 27일 → 4월 29일 변경 사항 (코드 영향)

### 변경 완료 목록 (2026-05-04)

| 모듈 | 변경 내용 | 상태 |
|------|----------|------|
| `rag/orphanet_fetcher.py` | 신규 구현: en_product6/9_ages/9_prev 파싱 — 유전자/유병률/발병연령/유전양식 | ✅ 완료 |
| `rag/monarch_fetcher.py` | 확장: `get_disease_info()` (name/description), OMIM search fallback | ✅ 완료 |
| `rag/pubcasefinder.py` | 엔드포인트 `/api/get_diseases` 수정, `phenotype=` 파라미터, `enrich_pcf_results()` | ✅ 완료 |
| `rag/pubmed_fetcher.py` | term="case reports", retmax=3, AbstractText ≤400자 | ✅ 완료 |
| `rag_pipeline.py` | 5단계 전면 개정 완료: Phase 4 분리, 조건부 RAG §3.3, JSON-only | ✅ 완료 |
| `rag_pipeline.py` SYSTEM_PROMPT | 확정 문서 §2.1~2.4 글자 그대로 적용 | ✅ 완료 |
| `rag_pipeline.py` user_prompt | 8개 섹션 구조 `_build_user_prompt()` 완성 | ✅ 완료 |
| Bedrock 파라미터 | max_tokens=2048, temperature=0.0 (확정 §2.2) | ✅ 완료 |
| Markdown 출력 제거 | JSON only, `_parse_json_response()` + `_validate_schema()` | ✅ 완료 |
| Top 3 RAG 수집 | Top 1 → Top 3 모두로 확장 | ✅ 완료 |

---

## 7. 외부 API 5개 검증 현황

| API | 상태 | 인증 | 비고 |
|-----|------|------|------|
| **AWS Bedrock (Claude Haiku)** | ✅ 정상 | AWS 자격증명 | Phase 1 (증상→HPO), Phase 4 (랭킹 정리) |
| **AWS Bedrock (Claude Sonnet 3.5)** | ✅ 정상 | AWS 자격증명 | ⑤ 최종 소견서 |
| **PubMed E-utilities** | ✅ 정상 | 무료 (NCBI API Key 권장) | rate 3 req/s |
| **ClinicalTrials.gov v2** | ✅ 정상 | 무료 | RECRUITING 필터, filter.overallStatus=RECRUITING |
| **PubCaseFinder** | ✅ 정상 | 무료 | DBCLS Japan, endpoint=get_diseases, target=omim, enrich_pcf_results()로 disease_name·pmid_list 보강 |
| **Monarch Initiative** | ✅ 정상 | 무료 | EMBL-EBI, get_disease_info()(name/description/causal_genes), OMIM 검색 fallback |
| **Orphanet (로컬 XML)** | ✅ 정상 | 무료 | en_product6/9_ages/9_prev 로컬 파싱 — 유전자/발병연령/유병률/유전양식 전부 구현 |

---

## 8. 검증 (확정 후 재실행 필요)

### 4월 27일 시점 검증 (참고)
- LIRICAL Recall@1 = 81.6%, Recall@10 = 98.3% (4293개 전수)
- 임상 시나리오 5개: Recall@3 = 100%
- PMID 유효율 = 100% (환각 0건)

### 5월 4일 이후 재검증 항목 (확정 문서 적용 후)
- [ ] 일반 Top 10 + 희귀 리스팅 듀얼 트랙 정확도
- [ ] Phase 4 LLM 정리 단계의 Top 3 변화율
- [ ] 5개 API 병렬 호출 시 wallclock latency (목표: 60초 이내)
- [ ] 조건부 RAG 분기 (케이스 A/B/C) 동작 검증
- [ ] JSON 스키마 검증 (recommendation + clinical_notes 필수 필드 누락 여부)
- [ ] MRN 누출 여부 (summary에 절대 포함 안 됨)
- [ ] DB ↔ API 교차검증 결과의 rag_evidence 반영 여부

---

## 9. 실행 방법

```bash
cd aws_say2_project_vision

# AWS 키 (환경변수로만 — CLAUDE.md 규칙)
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="ap-northeast-2"

# 1. 전체 5단계 파이프라인 (내장 샘플)
python rag_pipeline.py

# 2. MIMIC 실환자 검증
python rag/valid/fetch_mimic_patient.py
python rag/valid/run_rag_test.py

# 3. 개별 API 테스트
python rag/pubcasefinder.py
python rag/orphanet_fetcher.py
python rag/monarch_fetcher.py
python rag/pubmed_fetcher.py
python rag/clinicaltrials_fetcher.py

# 4. PMID 환각 체크
python rag/ragas_eval.py
```

---

## 10. 한계 및 추후 변경 가능 항목

확정 문서 §5 그대로:

| # | 항목 | 현재 | 검토 |
|---|------|------|------|
| ① | LLM 모델 | Sonnet 3.5 | Sonnet 4.6 |
| ② | RAG 수집 건수 | Top 3 × 3건 | Top 5~10 |
| ③ | RAG 트리거 범위 | Top 3 | Top 5/10 |
| ④ | PubCaseFinder target | ~~case~~ → **omim 확정 완료** | 추가 target 옵션 검토 |
| ⑤ | epidemiology_note | 희귀만 | 일반 포함 검토 |
| ⑥ | 내부 DB 교차검증 | Top 3 | Top 10 |
| ⑦ | 병렬 호출 | asyncio | rate limit 시 부분 순차 |
| ⑧ | 출력 랭킹 | 수집 10/출력 3 | Top 5~10 출력 |

---

## 11. 구현 현황 및 남은 To-Do

### ✅ 완료 (2026-05-04 기준)
1. ~~`rag_pipeline.py` 5단계 구조로 전면 개정~~ → **완료**
2. ~~시스템/유저 프롬프트를 확정 문서 §2~§3 그대로 교체~~ → **완료**
3. ~~JSON-only 출력으로 변경 (Markdown 제거)~~ → **완료**
4. ~~`rag/orphanet_fetcher.py` 신규 구현~~ → **완료** (en_product6/9_ages/9_prev 파싱)
5. ~~`rag/monarch_fetcher.py` 인과 유전자 추출 확장~~ → **완료** (get_disease_info 추가)
6. ~~조건부 RAG 분기 (§3.3) 구현~~ → **완료** (Case A/B/C 분기)
7. ~~Phase 4 LLM 정리 단계 구현 (Bedrock Haiku)~~ → **완료** (`step3_phase4_organize`)
8. ~~5개 API ThreadPoolExecutor 병렬 호출 + 60초 타임아웃~~ → **완료**
9. ~~DB ↔ API 교차검증 로직 구현~~ → **완료** (`cross_validate_genes`)
10. ~~JSON 스키마 검증기~~ → **완료** (`_validate_schema`)
11. ~~PubCaseFinder 엔드포인트 수정 (`/api/get_diseases`, `phenotype=`)~~ → **완료**
12. ~~Bedrock max_tokens=2048, temperature=0.0 확정~~ → **완료**

### 🟢 5/11~ (남은 검증 작업)
13. MIMIC 실환자 재검증 (3명+) — 5단계 파이프라인 기준 재실행
14. RAGAS 평가 (Faithfulness ≥ 0.8, Answer Relevancy)
15. AWS Bedrock 비용 측정 (Sonnet 3.5 vs 4.6 비교)

---

**SKKU AWS SAY 2기 2팀** | Rare-Link AI | 확정일: 2026-04-29 | 개정일: 2026-05-04 | 참여자: 권미라, 배기태, 허태웅
