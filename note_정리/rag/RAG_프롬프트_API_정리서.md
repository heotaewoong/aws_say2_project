# RAG 프롬프트 & API 호출 시스템 정리서

> **확정일**: 2026-04-29 (회의)
> **참여자**: 권미라, 배기태, 허태웅
> **기준 문서**: `최종프롬프트_API시스템_확정문서_v1.docx`
> **이 문서의 역할**: 코드 구현 시 절대 기준. 본 문서를 벗어난 임의 변경 금지.

---

## 1. 전체 파이프라인 5단계 (확정)

| 단계 | 모듈 | 처리 내용 | 출력 |
|------|------|-----------|------|
| **①** | **Phase 1~3** | 증상 / X-ray / Lab + Vital + Micro → HPO 변환 + LR 스코어링 | HPO 프로파일 |
| **②** | **스코어링 분기** | 일반/기타 DB → Top 10 랭킹 / 희귀 DB → 희귀질환 리스팅 | Top 10 + 희귀 리스팅 |
| **③** | **Phase 4** | LLM으로 Phase 3 결과 정리 | 정리된 랭킹 결과 |
| **④** | **RAG 트리거** | Top 3 기준 5개 API 병렬 호출 + 내부 DB 교차검증 | CARE JSON |
| **⑤** | **LLM 소견서 생성** | Bedrock Claude Sonnet 3.5 → JSON 소견서 | 최종 레포트 |

### 핵심 변경점 (4월 27일 → 4월 29일 확정)

| 항목 | 이전 (4-Phase) | 확정 (5단계) |
|------|-----------------|--------------|
| 스코어링 | LIRICAL 1개 | 일반/기타 DB Top10 + 희귀 DB 리스팅 **분리** |
| RAG 대상 | Top 1만 | **Top 3 모두** |
| API 수 | 3개 (PubCase, PubMed, ClinicalTrials) | **5개** (+ Monarch + Orphanet) |
| 출력 형식 | JSON + Markdown | **JSON only** (recommendation + clinical_notes) |
| RAG 수집 로직 | 무조건 수집 | **조건부 분기** (희귀 리스팅 유무 + Top 3 OrphaCode 유무) |
| Phase 4 | RAG 검색 단계 | **LLM이 Phase 3 결과 정리** (별도 단계) |

---

## 2. 시스템 프롬프트 (확정 — 글자 그대로)

### 2.1 페르소나

```
You are an elite AI diagnostician specializing in pulmonary and rare diseases.
You synthesize multimodal patient data and RAG-retrieved evidence to generate
a professional diagnostic support report.
Your role is to support physician decision-making, not to make final diagnoses.
Write your final report clearly, logically, and professionally.
Ensure your final output is written in Korean as requested by the clinical team.

[한국어 해석]
당신은 최고 수준의 폐질환 / 희귀질환 진단 전문가 AI입니다.
다중 모달리티 환자 데이터와 RAG 수집 근거를 종합하여
전문적인 진단 보조 소견서를 생성합니다.
최종 진단을 내리는 것이 아니라 의사의 의사결정을 보조하는 역할입니다.
명확하고, 논리적이며, 전문적으로 작성하십시오.
임상팀의 요청에 따라 최종 출력은 반드시 한국어로 작성하십시오.
```

### 2.2 Strict Rules (7개)

```
[strict rules]

1. Do not make assumptions or definitive conclusions without evidence.
   (근거 없는 추측이나 단정을 포함하지 않습니다.)

2. All claims must be grounded in the provided RAG data or established
   medical guidelines.
   (모든 주장은 제공된 RAG 데이터 또는 공인된 의학 가이드라인에 근거해야 합니다.)

3. If a rare disease (OrphaCode) appears in Top 3, MDT referral is mandatory.
   (희귀질환이 Top 3 이내에 있으면 MDT 협진 권고는 필수입니다.)

4. Prioritize diseases cross-validated by both Local DB and Global API.
   Provide a clinical rationale for the primary diagnosis.
   (로컬 DB·글로벌 API 양쪽 교차검증 질환을 우선순위로 삼고 임상 근거를 제시합니다.)

5. Differential Diagnosis: Use the patient's negative findings to
   logically explain why certain candidate diseases should be ruled out.
   (감별 진단: Negative HPO를 활용하여 배제 근거를 논리적으로 설명합니다.)

6. Case Comparison: Compare and contrast the patient's current state
   with the provided PubMed case reports.
   (사례 비교: PubMed 케이스리포트와 현재 환자 상태를 비교·대조합니다.)

7. Actionable Alternatives: Synthesize the Clinical Trials data to
   recommend practical clinical trial opportunities for the patient.
   (실행 가능한 대안: ClinicalTrials 데이터를 종합하여 임상시험 참여 기회를 권고합니다.)
```

### 2.3 출력 형식 (JSON only — Markdown 금지)

```json
{
  "recommendation": {
    "immediate_workup": ["Examination / procedure items"],
    "specialist_referral": ["Referral recommendation (with source and rationale)"],
    "treatment_guideline": [
      "[Disease 1] Treatment guideline (in order of priority)",
      "[Disease 2] Treatment guideline",
      "[Disease 3] Treatment guideline"
    ],
    "genetic_test": ["Genetic test recommendations (empty array if not applicable)"],
    "additional_lab": ["Additional lab recommendations"]
  },
  "clinical_notes": {
    "summary": "Comprehensive summary of chief complaint and AI analysis (include age, sex, chief complaint; exclude MRN)",
    "top1_reasoning": "Clinical rationale for Top 1 disease (use Positive HPO + Negative HPO + Lab findings)",
    "differential_note": "Top 2~3 differential diagnoses. Rare disease flags must be included regardless of probability",
    "rag_evidence": "Key clinical evidence from RAG results (cite sources). Include internal DB vs external API cross-validation results",
    "case_comparison": "Comparison of current patient with PubMed case reports (similarities, differences, implications)",
    "epidemiology_note": "Orphanet prevalence, age of onset, inheritance pattern (note DB vs API agreement). Empty string for common diseases",
    "disclaimer": "AI 결과는 진단 보조이며 최종 진단은 주치의의 임상 판단과 추가 검사 결과를 종합하여 확정합니다."
  }
}
```

**주의**: `Output must strictly follow this JSON structure. Do not include any text outside this JSON.`

### 2.4 작성 원칙 (10개)

| # | 필드 | 규칙 |
|---|------|------|
| 1 | `summary` | 나이·성별·주호소·주요 Lab 이상치 포함. **MRN 번호 절대 포함하지 말 것** |
| 2 | `top1_reasoning` | Positive HPO + Negative HPO + Lab 이상치를 **모두 구체적으로 언급** |
| 3 | `rag_evidence` | RAG JSON에 다음 데이터가 있으면 **반드시 인용** (아래 §2.4-3 상세) |
| 4 | `treatment_guideline` | Top 3 질환 각각 공식 가이드라인 기반 항목만. `[질환명]` 형식 prefix + 우선순위 정렬 |
| 5 | `specialist_referral` | 희귀질환(OrphaCode) Top 3 이내 시 **MDT 권고 반드시 포함** |
| 6 | `differential_note` | Top 2~3 감별진단. **희귀질환 플래그는 확률 낮아도 포함** |
| 7 | `genetic_test` | `association_type`이 "Disease-causing" 포함 유전자 필수. Monarch ↔ Orphadata 일치 시 "(복수 소스 확인)" 추가 |
| 8 | `case_comparison` | PubMed 케이스리포트와 환자의 공통점·차이점·유전자 검사 시사점 서술 |
| 9 | `epidemiology_note` | 일반 질환 = 빈 문자열. 희귀질환만 유병률·발병연령·유전양식 + DB ↔ API 일치 여부 |
| 10 | `disclaimer` | 고정 문구 항상 출력. **변경 불가** |

### 2.4-3 rag_evidence 인용 필수 항목 (확정 문서 §2.4-3 상세)

다음이 RAG JSON에 있으면 **반드시 인용**:

```
- genes_from_orphadata        → 유전자명 + association_type 명시
                                (예: "Disease-causing somatic mutation(s) in")

- phenotypes_from_orphadata    → frequency가 "Very frequent" 또는 "Frequent"인
                                HPO 표현형 2개 이상 인용

- epidemiology.prevalence      → 유병률 수치/범위 포함
                                (예: "1-9 / 1 000 000")

- epidemiology.age_of_onset    → 환자 연령과 비교 서술
                                (예: "Orphanet 기준 성인 발병 — 환자 34세로 전형적 범위")

- causal_genes (Monarch)       → Orphadata 유전자와 교차 확인 일치 여부 언급
                                (예: "Monarch·Orphanet 양 소스에서 TSC1/TSC2 확인")

- 일치 항목 (DB + API)         → "DB·API 교차검증 일치" 표기

- 불일치 항목                   → "DB·API 불일치 — 추가 확인 필요" 경고 표기
```

---

## 3. 유저 프롬프트 (확정 템플릿)

### 3.1 도입부

```
You are analyzing a patient case using multimodal clinical data and
RAG-retrieved evidence from both internal DB and external APIs.
Based on the structured data below, generate a comprehensive diagnostic
support report following the output format specified in the system prompt.

[한국어 해석]
아래 다중 모달리티 임상 데이터와 내부 DB·외부 API RAG 수집 결과를 분석하여,
시스템 프롬프트에 명시된 출력 형식에 따라 진단 보조 소견서를 작성하십시오.
```

### 3.2 8개 섹션 구조

```
=========================================
=== 1. 환자 기본정보 ===
{
  "name": "{name}",
  "age": {age},
  "sex": "{sex}",
  "visit_date": "{visit_date}",
  "visit_type": "{visit_type}",
  "chief_complaint": "{chief_complaint}",
  "allergy": "{allergy}"
}

=== 2. 증상 원문 ===
- Positive Findings (양성 증상): {symptoms_raw}
- Negative Findings (음성 소견): {negative_raw}

=== 3. HPO 프로파일 ===
Positive HPO:
{pos_hpos}
※ source 구분: symptom / xray

Negative HPO (증상에서만):
{neg_hpos}

=== 4. Lab 수치 ===
{lab_data}

=== 5. 일반/기타 폐질환 랭킹 Top 10 (로컬 DB 기반) ===
{ranking_general}

=== 6. 희귀폐질환 리스팅 (로컬 DB 기반) ===
{ranking_rare}

=== 7. 내부 DB 정보 — Top 3 교차검증용 ===
{internal_db_context}
※ 외부 API 결과와 대조하여 일치/불일치 여부를
  rag_evidence에 반드시 명시할 것.

=== 8. RAG 검색 결과 (외부 API) ===
※ Orphanet / PubCaseFinder는 희귀질환(OrphaCode) 해당 시에만 수집.
  일반 질환만 Top 3인 경우 해당 섹션 자동 스킵.

--- Top 1: {disease_1} ({orpha_code_1}) ---
[Orphanet] (희귀질환만, 일반 질환 스킵)
- 유전자: {genes_1} (association_type 포함)
- Very frequent / Frequent HPO: {hpo_frequent_1}
- 유병률: {prevalence_1}
- 발병연령: {age_of_onset_1}

[Monarch]
- 인과 유전자: {monarch_genes_1}
- Orphanet 교차검증: {cross_validation_1}

[PubMed 케이스리포트]
{pubmed_cases_1}

[PubCaseFinder] (희귀질환만, 일반 질환 스킵)
{pubcasefinder_1}

[ClinicalTrials (RECRUITING)]
{clinical_trials_1}

--- Top 2: {disease_2} ({orpha_code_2}) ---
[ ... Top 1과 동일 구조 ... ]

--- Top 3: {disease_3} ({orpha_code_3}) ---
[ ... Top 1과 동일 구조 ... ]
=========================================

위 데이터를 종합하여 규정된 JSON 형식으로 출력하십시오.
```

### 3.3 RAG 수집 조건부 로직 (필수)

```
[RAG 수집 로직]

희귀질환 리스팅 있음
  → Orphanet + PubCaseFinder 수집

희귀질환 리스팅 없음 + Top 3에 OrphaCode 있음
  → 6번 "해당 없음" 표기
  → Orphanet + PubCaseFinder 수집 (교차검증 목적)

희귀질환 리스팅 없음 + Top 3 전부 일반 질환
  → 6번 "해당 없음" 표기
  → Orphanet + PubCaseFinder 스킵
  → PubMed + Monarch + ClinicalTrials만 수집
```

---

## 4. API 호출 시스템 (5개 API)

### 4.1 데이터 흐름

```
HPO list
  └─→ PubCaseFinder
        └─→ 후보 질환 ID / 이름 / 점수 (Top 3 추출)
              │
              ├──── 병렬 ────┐
              │              │
              ├─→ Monarch API           (인과 유전자, HPO 매핑 메타데이터)
              ├─→ Orphanet API          (유전자, 빈도, 유병률, 발병연령)
              ├─→ PubMed API            (Top 3 질환명별 초록 3건)
              └─→ ClinicalTrials.gov    (Top 3 질환명별 RECRUITING 3건)
                          │
                          ▼
              local DB 결과 vs API 결과 비교/검증
                          │
                          ▼
              모든 데이터를 프롬프트 템플릿(8개 섹션)에 삽입
                          │
                          ▼
              Bedrock Claude Sonnet 3.5 호출
                          │
                          ▼
                    JSON 소견서 출력
```

### 4.2 API별 역할 (확정)

| API | 호출 트리거 | 입력 | 출력 데이터 | 비고 |
|-----|-------------|------|-------------|------|
| **PubCaseFinder** | 희귀 리스팅 있음 OR Top3에 OrphaCode 있음 | HPO list (`phenotype=`) | disease_id (OMIM), disease_name, score, pmid_list, HPO list | endpoint: `/api/get_diseases`, target=omim, `enrich_pcf_results()`로 Monarch·PubMed 보강 |
| **Orphanet** | 희귀 리스팅 있음 OR Top3에 OrphaCode 있음 | OrphaCode (Top 3) | genes_from_orphadata (en_product6.xml), phenotypes (빈도), prevalence (en_product9_prev.xml), age_of_onset + inheritance (en_product9_ages.xml) | 로컬 XML 4파일 파싱 — 전부 구현 완료 |
| **Monarch** | 항상 (Top 3 모두) | disease_id (URL path) | name/label, description, causal_genes | `get_disease_info()`: direct entity → search fallback (OMIM MONDO 매핑) |
| **PubMed** | 항상 (Top 3 모두) | 질환명 + "case reports" (Top 3) | PMID (ArticleTitle, AbstractText ≤400자) × 3건 | eSearch retmax=3, eFetch retmode=xml (§2.2 스펙), AbstractText XML 파싱 |
| **ClinicalTrials.gov** | 항상 (Top 3 모두) | query.cond=질환명, filter.overallStatus=RECRUITING, pageSize=3 | nctId, briefTitle, 상태&단계, 요약문, 링크 | API v2, locations 포함 |

### 4.3 병렬 호출 정책

- **방식**: `ThreadPoolExecutor`로 5개 API 병렬 실행 (현재 구현)
- **타임아웃**: 각 API 15초, 전체 wallclock 60초
- **실패 처리**: 단일 API 실패 시 해당 섹션만 "데이터 없음" 표기, 파이프라인 진행
- **Rate limit 이슈 시**: 부분 순차 전환 (PubMed → ClinicalTrials → Monarch 순)

### 4.4 Bedrock 최종 소견서 생성 파라미터 (확정 §2.2)

| 파라미터 | 확정값 | 비고 |
|----------|--------|------|
| `modelId` | `apac.anthropic.claude-3-5-sonnet-20241022-v2:0` | Claude Sonnet 3.5 (APAC) |
| `max_tokens` | `2048` | 확정 §2.2 |
| `temperature` | `0.0` | 확정 §2.2 (재현성 최우선) |
| `anthropic_version` | `bedrock-2023-05-31` | 고정 |
| `system` | §2.1~2.4 System Prompt 그대로 | 임의 변경 금지 |
| `messages[0].content` | §3.2 User Prompt 8개 섹션 | 임의 변경 금지 |

---

## 5. 추후 변경 가능 항목 (Future Work)

| # | 항목 | 현재값 | 변경 검토 |
|---|------|--------|-----------|
| ① | LLM 모델 | Claude Sonnet 3.5 | Sonnet 4.6 (성능·토큰수·비용 검토 후) |
| ② | RAG 수집 건수 | Top 3 × PubMed 3건 + ClinicalTrials 3건 | Top 5~10 확장 또는 건수 조정 |
| ③ | RAG 트리거 범위 | Top 3 기준 API 호출 | Top 5 또는 Top 10으로 확장 |
| ④ | PubCaseFinder target | ~~`target=case`~~ → **`target=omim` 확정 완료** | 추가 target 옵션(rare_disease) 검토 |
| ⑤ | epidemiology_note 활용 | 희귀질환만 (검증용) | 일반 질환도 역학 데이터 포함 검토 |
| ⑥ | 내부 DB 교차검증 범위 | Top 3만 | Top 10 전체 확장 |
| ⑦ | 병렬 호출 | asyncio 병렬 | rate limit 이슈 시 부분 순차 전환 |
| ⑧ | 출력 랭킹 범위 | 수집 Top 10, 출력 Top 3 | 임상 피드백 반영 시 Top 5~10 확장 |

---

## 6. 구현 시 절대 규칙 (Quick Reference)

### Do
- ✅ 시스템 프롬프트는 §2.1~2.4 글자 그대로 사용
- ✅ 유저 프롬프트는 §3.2의 8개 섹션 순서/제목 그대로
- ✅ JSON 출력만 — Markdown 절대 출력 금지
- ✅ RAG 조건부 분기 로직 (§3.3) 필수 구현
- ✅ Top 3 모두에 대해 5개 API 호출 (Top 1만 X)
- ✅ Monarch + Orphanet 결과를 항상 교차검증해서 rag_evidence에 반영
- ✅ MRN은 절대 LLM에 노출 금지 (summary 작성 원칙)
- ✅ 희귀질환(OrphaCode) Top 3 시 MDT 권고 무조건 포함

### Don't
- ❌ Phase 1A/1B/1C 옛 명명 사용 금지 (확정: Phase 1, 2, 3, 4)
- ❌ Top 1만 RAG 수집 금지 — Top 3 모두
- ❌ Markdown 보고서 따로 생성 금지 (JSON only)
- ❌ 일반/기타 랭킹과 희귀 리스팅을 같은 표에 합치지 말 것 (§3.2 5번/6번 분리)
- ❌ disclaimer 문구 임의 변경 금지
- ❌ 희귀질환 Top 3 진입했는데 일반 가이드라인만 권고하는 것 금지

---

**SKKU AWS SAY 2기 2팀** | Rare-Link AI | 확정일: 2026-04-29 | 참여자: 권미라, 배기태, 허태웅
