# Rare-Link AI — RAG 파이프라인 구현 보고서

> 작성일: 2026-05-06 (코드 재검증)
> 대상 독자: RAG·AI 비전공 팀원 (초보자용)
> 코드 기준: `rag_pipeline.py` + `rag/` 패키지 (확정 v1.0)

---

## 목차

1. [RAG가 뭔가요? — 30초 이해](#1-rag가-뭔가요--30초-이해)
2. [우리 시스템의 전체 그림](#2-우리-시스템의-전체-그림)
3. [핵심 개념 용어집 — 한 페이지로 보기](#3-핵심-개념-용어집--한-페이지로-보기)
4. [5단계 파이프라인 상세](#4-5단계-파이프라인-상세)
5. [API 호출 시스템 — 5개 외부 데이터베이스](#5-api-호출-시스템--5개-외부-데이터베이스)
6. [HPO란? — 증상을 코드로 바꾸는 이유](#6-hpo란--증상을-코드로-바꾸는-이유)
7. [스코어링 시스템 — 어떻게 질환 순위를 매기나](#7-스코어링-시스템--어떻게-질환-순위를-매기나)
8. [프롬프트 시스템 — AI에게 어떻게 말하나](#8-프롬프트-시스템--ai에게-어떻게-말하나)
9. [출력 JSON — AI가 내놓는 최종 결과물](#9-출력-json--ai가-내놓는-최종-결과물)
10. [데이터 흐름 요약표](#10-데이터-흐름-요약표)
11. [실패 대비 설계](#11-실패-대비-설계)
12. [실행 방법 & 파일별 역할 (부록)](#12-실행-방법--파일별-역할-부록)

---

## 1. RAG가 뭔가요? — 30초 이해

**RAG = Retrieval-Augmented Generation**
→ "검색해서 찾은 정보를 AI에게 먹여서 더 나은 답변을 만드는 방식"

일반 AI(GPT 등)의 문제점:
- 학습 데이터에 없는 최신 논문, 새 임상시험 정보를 모름
- "내가 아는 지식"만으로 대답하기 때문에 희귀질환처럼 전문적인 분야에선 틀릴 수 있음

RAG를 쓰면:
1. 먼저 실시간으로 PubMed, ClinicalTrials 등에서 **관련 정보를 검색**
2. 그 정보를 AI 프롬프트에 **통째로 붙여서** 질문
3. AI가 외부 데이터에 근거해서 답변

쉽게 말해 **"오픈북 시험"** 처럼 AI가 참고자료를 보고 답변하는 방식입니다.

---

## 2. 우리 시스템의 전체 그림

```
환자 데이터 입력
  ├─ 흉부 X-ray 이미지
  ├─ 증상 텍스트 (의사 메모)
  ├─ 혈액검사·폐기능 수치
  └─ 기본 정보 (나이/성별 등)
          │
          ▼
  ┌──────────────────────┐
  │   HPO 변환 (Phase 1~3) │  ← 증상을 국제 표준 코드로 변환
  └──────────┬───────────┘
             │ HPO 코드 목록
             ▼
  ┌──────────────────────┐
  │   질환 스코어링       │  ← 어떤 질환일 가능성이 높은지 점수 계산
  │  [일반 Top10 / 희귀 리스팅]  │
  └──────────┬───────────┘
             │ Top 3 후보 질환
             ▼
  ┌──────────────────────┐
  │   5개 API 병렬 호출   │  ← 외부 의료 DB에서 근거 수집
  │  (RAG 핵심 단계)      │
  └──────────┬───────────┘
             │ 논문, 유전자, 임상시험 데이터
             ▼
  ┌──────────────────────┐
  │  AWS Bedrock AI 최종 │  ← 모든 데이터를 AI에게 전달 → 소견서 생성
  │  소견서 생성          │
  └──────────────────────┘
          │
          ▼
  JSON 진단 보조 소견서
  (권고사항 + 임상 노트)
```

---

## 3. 핵심 개념 용어집 — 한 페이지로 보기

| 용어 | 쉬운 설명 |
|------|-----------|
| **HPO** | "증상을 숫자코드로 바꾼 것". 예: `HP:0002094`=호흡곤란. 영어·한국어 상관없이 같은 코드를 씀. |
| **OrphaCode** | Orphanet 희귀질환 DB의 고유 ID. 예: `ORPHA:723`=LAM(림프관근육종증). |
| **OMIM ID** | 유전성 질환 전용 DB의 ID. 예: `OMIM:617300`. |
| **Positive Findings** | 환자에게 "있는" 증상·소견 (예: 호흡곤란, 흉통). |
| **Negative Findings** | 환자에게 "없는" 증상·소견 (예: 기침 없음, 발열 없음). 감별진단에 필수. |
| **LR (Likelihood Ratio)** | "이 증상 조합이 이 질환일 가능성을 얼마나 높이는가"의 비율. 높을수록 확실. |
| **MDT** | Multi-Disciplinary Team. 여러 과 전문의가 모여서 하는 협진 회의. |
| **temperature=0** | LLM이 항상 같은 답변을 내도록 하는 설정. 창의적 답변 방지 → 의료용. |
| **ThreadPoolExecutor** | 여러 API를 **동시에** 호출해서 총 대기시간을 줄이는 파이썬 도구. |
| **Fallback(폴백)** | 원래 API가 실패했을 때 쓰는 대체 수단. 우리는 로컬 CSV를 폴백으로 사용. |
| **Caching(캐시)** | 한번 받은 결과를 저장해서 같은 요청이 오면 DB 호출 없이 바로 반환. |

---

## 4. 5단계 파이프라인 상세

`rag_pipeline.py`의 `RareLinkPipeline` 클래스가 5단계를 순서대로 실행합니다.

### ① 단계 1 — 멀티모달 → HPO 변환 (`step1_phase123_get_hpo`)

**목적**: 서로 다른 형태의 환자 데이터를 하나의 표준 코드(HPO)로 통일

세 가지 소스에서 동시에 HPO를 추출합니다:

| Phase | 입력 | 처리 방법 | 출력 |
|-------|------|-----------|------|
| Phase 1 | 증상 텍스트 ("호흡곤란, 흉통") | Bedrock Claude Haiku AI가 읽고 코드 추출 | `HP:0002094`, `HP:0100749` |
| Phase 2 | X-ray 이미지 | SooNet 딥러닝 모델이 판독 | `HP:0002202` (흉막삼출) |
| Phase 3 | 혈액검사 수치 (WBC=15.2) | 임계값 기반 Rule (WBC>11 → 백혈구 증가) | `HP:0011897` |

**중요한 디테일 — Negative 우선 통합**:
같은 HPO 코드가 Positive와 Negative 양쪽에 나오면 Negative를 우선합니다. 예를 들어 의사가 "기침이 있었는데 지금은 없음"이라고 썼다면 `HP:0012735`(기침)는 Negative로 처리.

```python
# 코드 예시 (rag_pipeline.py:245~316)
nlp_pos   = self.hpo_extractor.extract_hpo(symptom_text)   # Phase 1: Positive 추출
neg_from_pos = nlp_pos.get("negative_hpo", [])              # symptom_text에 있던 부정문
# negative_text (예: "기침 없음, 발열 없음")이 별도로 들어오면 그것도 추출
nlp_neg   = self.hpo_extractor.extract_hpo(negative_text)
neg_explicit = nlp_neg.get("positive_hpo", [])              # negative 텍스트의 양성 = 음성 소견
neg_nlp   = list(set(neg_from_pos + neg_explicit))

xray_hpos = [hpo for label, (prob, hpo) in xray_preds.items()
             if prob >= 0.3]                               # Phase 2: threshold 0.3
lab_hpos  = lab_to_hpo(lab_results)                        # Phase 3: Rule 적용

# Negative 우선 제거
all_positive = list(set(xray_hpos + pos_nlp + lab_hpos) - set(neg_nlp))
```

**출력물**: 각 HPO에 `source`(symptom/xray/lab) 태그가 붙어서 LLM이 "이 증상은 X-ray에서 나온 거구나"처럼 출처를 구분할 수 있게 됨.

---

### ② 단계 2 — 질환 스코어링 (`step2_dual_scoring`)

**목적**: 수천 개의 질환 중에서 환자 HPO와 가장 잘 맞는 질환을 추려냄

두 트랙이 **독립적으로** 동작합니다:

**트랙 A — 일반 폐질환 (Rule-based)**
`general_disease_scorer.py`가 담당. `lung_disease_profiles_v2.yaml` 파일에 정의된 폐렴/COPD/폐암 등 일반 질환 프로파일과 매칭.
점수 = `HPO 매칭(25%) + X-ray 소견(50%) + Lab 수치(25%)` 가중치 합산 → **Top 10** 반환

**트랙 B — 희귀질환 (LIRICAL 방식)**
`lirical_scorer.py`가 담당. Orphanet DB의 약 4,335개 희귀질환과 **LR(우도비) 계산**.

LIRICAL 공식 (있는 증상 × 없는 증상 양쪽 반영):
```
LR = ∏(있는 증상 i: sensitivity_i / bg_freq_i)
   × ∏(없는 증상 j: (1 - sensitivity_j) / (1 - bg_freq_j))
```
- `sensitivity`: 해당 질환 환자에서 그 증상이 나타날 확률 (Orphanet 빈도)
- `bg_freq`: 일반 인구에서 그 증상 빈도 (기본값 0.05 = 5%)

LR > 1.0 인 희귀질환만 리스팅 → **희귀 후보 목록** 반환

---

### ③ 단계 3 — Top 3 통합 (`step3_phase4_organize`)

**목적**: 일반 Top10과 희귀 리스팅을 합쳐서 최종 Top 3 확정

**Bedrock Haiku(`claude-3-5-haiku-20241022`)** 를 호출해서 두 랭킹을 임상 우선순위로 통합합니다.

동작 방식:
1. 일반 Top 10 + 희귀 리스팅을 Haiku에게 전달
2. Haiku가 임상적 우선순위를 판단해서 Top 3 JSON 배열 반환
3. Haiku 호출 실패 시 → **폴백**: 희귀 우선 배치 → 일반으로 채우는 휴리스틱 자동 실행

희귀질환이 Top 3 안에 들면 → 최종 보고서에 **MDT 협진 필수** 플래그

> 💡 **초보자 가이드** — MDT 협진이란?
> Multi-Disciplinary Team. 희귀질환은 한 명의 의사가 판단하기 어려워서 호흡기내과·유전학과·영상의학과가 함께 모여 회의하는 걸 말합니다. 우리 시스템은 희귀질환이 Top 3 안에 들면 "MDT 권고"를 자동으로 붙입니다.

---

### ④ 단계 4 — RAG 트리거: 5개 API 병렬 호출 (`step4_rag_collect`)

**목적**: Top 3 질환에 대한 외부 의료 DB 근거 수집

이 단계가 RAG의 핵심입니다. 이미지에서 설명한 흐름과 동일합니다.
자세한 내용은 [5. API 호출 시스템](#5-api-호출-시스템--5개-외부-데이터베이스)에서 설명합니다.

---

### ⑤ 단계 5 — AI 소견서 생성 (`step5_generate_report`)

**목적**: 수집된 모든 데이터를 AI에게 전달 → JSON 진단 보조 소견서

- 모델: AWS Bedrock Claude Sonnet 3.5 (`apac.anthropic.claude-3-5-sonnet-20241022-v2:0`)
- 온도: 0.0 (재현성 최대화, 창의적 답변 방지)
- 최대 토큰: 4096
- 출력: JSON만 (마크다운 절대 금지)

> 💡 **초보자 가이드** — temperature=0 이 왜 중요한가?
> LLM은 기본적으로 매번 다른 답을 만듭니다. temperature가 높을수록 "창의적"이 되지만 의료에서는 **같은 환자에겐 같은 소견서**가 나와야 하므로 0으로 고정합니다.

**출력 검증 단계** (`_validate_schema`):
AI 출력 JSON이 정해진 필드(`recommendation.immediate_workup`, `clinical_notes.summary` 등)를 모두 포함하는지 자동 체크. 필드가 빠지면 경고 로그.

---

## 5. API 호출 시스템 — 5개 외부 데이터베이스

> 2026-04-29 회의 확정 흐름 기준. `rag_pipeline.py step4_rag_collect()` 반영 완료.

### 확정 데이터 흐름 (이미지 기준)

```
HPO list (Positive HPO 코드들)
    │
    ▼
[1단계] PubCaseFinder ← 1번 관문
    HPO 코드 목록 → 후보 질환 ID / 이름 / 점수 반환
    (희귀질환 케이스에만 호출. 실패 시 로컬 폴백)
    │
    ▼ 후보 질환 ID/OrphaCode 확보
    │
    ├─────────────────────────────────────────────┐
    │ [2단계] 병렬 메타데이터 보강                  │
    ├──▶ Monarch API                              │
    │      OrphaCode → 인과 유전자 목록            │
    │      예: ["TSC1", "TSC2"]                   │
    │                                             │
    └──▶ Orphanet (로컬 XML)                      │
           OrphaCode → 유전자 + 역학 + 발현형      │
           (유병률, 발병연령, 유전양식, HPO 빈도)   │
    │                                             │
    ▼ (보강된 질환명으로)                           │
    │                                             │
    ├─────────────────────────────────────────────┘
    │ [3단계] 병렬 근거 수집 (Top 3 질환 각각)
    ├──▶ PubMed
    │      질환명 → 케이스리포트 초록 수집
    │      반환: PMID, 제목, 출판일, abstract
    │
    └──▶ ClinicalTrials.gov
           질환명 → 현재 모집 중인 임상시험
           반환: NCT ID, 제목, Phase, 참여기관
    │
    ▼
local DB 결과 vs API 결과 비교·검증
    (Orphanet 유전자 ↔ Monarch 유전자 교차검증)
    │
    ▼
모든 데이터를 8개 섹션 프롬프트 템플릿에 삽입
    │
    ▼
AWS Bedrock Claude Sonnet 3.5 → JSON 소견서
```

### 이전 구현과의 차이

| 항목 | 이전 구현 | 현재 구현 (이미지 확정) |
|------|-----------|------------------------|
| PubCaseFinder 위치 | 조건부 보조 (Top1에 한 번) | **1번 관문** (HPO → 후보 질환 확보) |
| API 실행 구조 | 모든 API 한 번에 병렬 | **3단계 순차** (PCF → 보강 → 근거수집) |
| 희귀질환 판단 기준 | 단순 OR 판단 | **케이스 A/B/C 분기** (§3.3 확정 복원) |

### 케이스 분기 로직 (확정 §3.3)

| 케이스 | 조건 | 수집 API |
|--------|------|---------|
| **A** | 희귀 리스팅 존재 | PubCaseFinder + Orphanet + Monarch + PubMed + ClinicalTrials (전부) |
| **B** | 리스팅 없지만 Top3에 OrphaCode 있음 | A와 동일 (교차검증 목적) |
| **C** | Top3 전부 일반 질환 | Orphanet + PubCaseFinder **스킵**, 나머지 3개만 |

케이스 C로 가면 희귀질환 전용 API 호출을 건너뛰어 비용·시간 절약.

> 💡 **초보자 가이드** — 왜 병렬 호출인가?
> API 5개를 **순서대로** 호출하면 각 API가 3초씩만 걸려도 총 15초 이상. `ThreadPoolExecutor(max_workers=6)`를 쓰면 6개를 **동시에** 호출해서 가장 느린 API 시간만 기다리면 됨. 총 20~30초 내 완료.

### 각 API 상세

#### [1] PubCaseFinder (`rag/pubcasefinder.py`) — 1번 관문

- **운영**: DBCLS (일본 생명과학통합데이터베이스)
- **엔드포인트**: `https://pubcasefinder.dbcls.jp/api/get_diseases`
- **기본 target**: `"omim"` (기본값) — 반환 `disease_id`가 `OMIM:NNNNNN` 형식
- **입력**: HPO 코드 목록 (쉼표 구분)
- **반환 정보**:
  ```json
  {
    "disease_id": "OMIM:617300",
    "score": 0.95,
    "matched_hpo_id": "HP:0002202,HP:0002094",
    "gene_id": "GENEID:2050"
  }
  ```
- **보강 단계**: `enrich_pcf_results(pcf_results, fetch_pmids=False)` 호출
  - disease_name이 없으면 Monarch로 채움
  - `fetch_pmids=False`인 이유: PubMed 호출은 3단계(근거수집)에서 별도로 하기 때문에 중복 방지
- **역할**: HPO 코드만 넣으면 "이 증상 조합과 가장 유사한 희귀질환이 뭔지" 후보 목록 반환. 이 후보 ID를 가지고 2단계 API들이 세부 정보를 채움
- **LIRICAL과의 차이**: LIRICAL은 Orphanet 빈도 통계 기반(로컬), PCF는 케이스리포트 유사도 기반(API) → 두 결과를 비교해서 교차검증 가능
- **실패 시**: 로컬 `orphadata_weighted.csv`에서 HPO 매칭으로 대체 (폴백)
- **캐시**: 동일 HPO 조합은 MD5 해시로 `data/pcf_cache/{hash}.json`에 저장 → 재요청 없음

#### [2] Orphanet (`rag/orphanet_fetcher.py`) — 메타데이터 보강

- **운영**: INSERM (프랑스)
- **구현**: 로컬 XML 파일 파싱 (`data/` 폴더의 en_product*.xml) — **인터넷 안 탐**
- **입력**: OrphaCode (예: "538")
- **반환 정보**:

| 필드 | 소스 파일 | 내용 |
|------|-----------|------|
| `disease_name`, `disorder_type` | `en_product4.xml` | 질환명 + 질환 분류 |
| `genes_from_orphadata` | `en_product6.xml` | 유전자명 + association_type (예: "Disease-causing germline mutation") |
| `phenotypes_from_orphadata` | `orphadata_weighted.csv` | Very frequent / Frequent HPO |
| `epidemiology.prevalence` | `en_product9_prev.xml` | 유병률 (예: "1-9/100000") |
| `epidemiology.age_of_onset` | `en_product9_ages.xml` | 발병연령 (예: "Adult") |
| `epidemiology.inheritance` | `en_product9_ages.xml` | 유전 양식 (예: "Autosomal dominant") |

- **호출 조건**: 희귀질환 OrphaCode 있을 때만 (일반질환 스킵)
- **`orphadata_weighted.csv`는 어디서 왔나**: `rag/knowledge_base.py`가 `en_product4.xml`을 파싱해서 HPO 빈도 라벨을 가중치(Always=1.0, Very frequent=0.9, Frequent=0.5, Occasional=0.1, Unknown=0.3)로 변환한 CSV

#### [3] Monarch Initiative (`rag/monarch_fetcher.py`) — 유전자 교차검증

- **운영**: Monarch Initiative Consortium (국제 연구 컨소시엄)
- **엔드포인트**: `https://api.monarchinitiative.org/v3/api/entity`
- **두 가지 역할**:
  1. **HPO 코드 → 영어 이름 변환**: `HP:0002094` → `"Dyspnea"` (`get_hpo_name`)
  2. **OrphaCode → 인과 유전자 목록**: `"538"` → `["TSC1", "TSC2"]` (`get_causal_genes`)
- **교차검증 방식** (`cross_validate_genes`):
  - 두 세트의 교집합 / 대칭차 계산
  - 완전 일치 → `"DB·API 교차검증 일치"`
  - 부분 일치 → `"부분 일치 — 일치: X / 불일치: Y"`
  - 완전 불일치 → `"DB·API 불일치 — 추가 확인 필요"`
- **폴백**: 자주 쓰는 HPO 25개는 로컬 딕셔너리(`_HPO_FALLBACK`)에 미리 정의 → API 실패해도 동작

#### [4] PubMed (`rag/pubmed_fetcher.py`) — 케이스리포트 근거

- **운영**: NIH/NLM (미국 국립의학도서관)
- **API**: E-utilities (**3단계 호출**)
  1. **esearch**: 질환명으로 PMID 목록 검색 (`sort=date`, `retmax=3`)
  2. **esummary**: PMID 목록 → 제목·저자·출판일 (JSON)
  3. **efetch**: PMID 목록 → abstract 본문 (XML 파싱)
- **검색 쿼리**: `"질환명"[Title/Abstract] AND case reports[Title/Abstract]`
- **Rate limit**: API key 없이 초당 3회. 호출 사이 `0.4초 대기` 설정
- **반환 정보**:
  ```json
  {
    "pmid": "38765432",
    "title": "A case of...",
    "pubdate": "2024 Mar",
    "url": "https://pubmed.ncbi.nlm.nih.gov/38765432/",
    "abstract": "최대 400자 초록"
  }
  ```
- **구조적 abstract 처리**: "Background: ... / Methods: ... / Results: ..." 형태를 라벨 보존하면서 파싱
- **호출 방식**: Top 1/2/3 각각 독립 호출 → 질환별 최대 3편 수집

#### [5] ClinicalTrials.gov (`rag/clinicaltrials_fetcher.py`) — 임상시험 정보

- **운영**: NIH (미국 국립보건원)
- **API**: v2 (`https://clinicaltrials.gov/api/v2/studies`)
- **필터**: `overallStatus=RECRUITING` (현재 모집 중인 것만)
- **반환 정보**:
  ```json
  {
    "nct_id": "NCT05123456",
    "title": "A Phase 2 Study of...",
    "phase": "PHASE2",
    "status": "RECRUITING",
    "locations": ["서울대병원", "Mayo Clinic"],
    "summary": "임상시험 요약 (300자)"
  }
  ```
- **목적**: "지금 이 질환으로 참여 가능한 임상시험이 있습니다" 정보 제공
- **호출 방식**: Top 1/2/3 각각 독립 호출

### 3단계 실행 구조 (코드 기준)

```python
# [1단계] PubCaseFinder — 순차 실행 (1번 관문)
pcf_results = get_ranked_diseases(hpo_data["positive_hpo"], top_k=3)
pcf_results = enrich_pcf_results(pcf_results, fetch_pmids=False)
# → 후보 질환 ID 확보 (disease_name 보강)

# [2단계] Monarch + Orphanet — 병렬 실행 (메타데이터 보강)
with ThreadPoolExecutor(max_workers=6) as executor:
    executor.submit(get_causal_genes, orpha_code)   # Monarch
    executor.submit(get_orphanet_data, orpha_code)  # Orphanet (희귀만)

# [3단계] PubMed + ClinicalTrials — 병렬 실행 (근거 수집)
with ThreadPoolExecutor(max_workers=6) as executor:
    executor.submit(pubmed.get_top_papers, disease_name)  # Top1/2/3 각각
    executor.submit(get_clinical_trials, disease_name)    # Top1/2/3 각각

# 마지막: local DB vs API 교차검증
cross_validate_genes(orpha_genes, monarch_genes)
```

전체 API 호출 수: 희귀질환 케이스 기준 최대 **1(PCF) + 6(Monarch×3) + 3(Orphanet×3) + 6(PubMed×3) + 6(CT×3) = 22회** → 3단계 병렬로 약 20~30초 내 완료.

---

## 6. HPO란? — 증상을 코드로 바꾸는 이유

**HPO (Human Phenotype Ontology)** = 의학적 증상을 국제 표준 코드로 표현하는 체계

왜 필요한가?
- "숨이 차다", "호흡곤란", "dyspnea" → 모두 같은 증상이지만 표현이 다름
- HPO 코드를 쓰면 언어/표현에 관계없이 동일하게 처리 가능
- Orphanet, Monarch 등 모든 글로벌 의료 DB가 HPO로 데이터를 저장 → **API와 통신 가능**

예시:
```
"숨이 차다" → HP:0002094 (Dyspnea, 호흡곤란)
"흉막에 물이 찼다" → HP:0002202 (Pleural effusion, 흉막삼출)
"기침이 없다" → Negative HPO: HP:0012735
```

**Phase 1** (`bedrock_extractor.py`): Bedrock Haiku AI가 자연어 텍스트에서 HPO 추출
- 모델 ID: `anthropic.claude-3-haiku-20240307-v1:0`
- max_tokens=512, temperature=0.0
```
시스템 프롬프트: "HPO 코드를 추출하시오. HP:XXXXXXX 형식만. 확신 없으면 생략."
유저 입력: "40세 여성. 호흡곤란과 흉통. 기침은 없음."
AI 출력: {"positive_hpo": ["HP:0002094", "HP:0100749"], "negative_hpo": ["HP:0012735"]}
```

**Phase 3** (`lab_rules.py`): 혈액검사 수치는 Rule로 직접 변환
```python
LAB_HPO_RULES = {
    "WBC": {
        "high": {"threshold": 11.0, "hpo": "HP:0011897"},  # WBC > 11 → 백혈구 증가
        "low":  {"threshold": 4.0,  "hpo": "HP:0001882"},  # WBC < 4  → 백혈구 감소
    },
    "SpO2": {
        "low":  {"threshold": 95.0, "hpo": "HP:0012418"},  # SpO2 < 95 → 저산소증
    },
    "FEV1": {
        "low":  {"threshold": 80.0, "hpo": "HP:0002093"},  # FEV1 < 80 → 기도 폐쇄
    },
    # ... 15+ 종의 Lab 항목 정의
}
```

**키 정규화** (`_normalize_key`): "d-dimer", "D_DIMER", "hemoglobin" 같이 다른 표기도 표준 키로 매핑.

---

## 7. 스코어링 시스템 — 어떻게 질환 순위를 매기나

### 일반 질환 스코어링 (`general_disease_scorer.py`)

세 가지 신호를 가중치 합산:

```
종합 점수 = 증상 HPO 매칭(25%) + X-ray 소견 매칭(50%) + Lab 패턴(25%)
```

X-ray 가중치가 50%인 이유: 우리 시스템이 흉부 X-ray를 기반으로 하는 폐질환 진단이라서

각 점수 계산법:
- **HPO 매칭**: (환자 HPO ∩ 질환 프로파일 HPO) / 질환 프로파일 HPO 수
- **X-ray 소견**: SooNet 예측 라벨 → 키워드 매핑(`_XRAY_LABEL_MAP`) → 질환 `radiology_findings` 키워드와 매칭
- **Lab 패턴**: `lab_patterns` (예: "leukocytosis", "elevated crp") → 실제 수치 Rule로 검증

예시 계산 (폐렴 후보):
- HPO 매칭: 환자의 3개 HPO 중 폐렴 프로파일과 2개 일치 → 0.67
- X-ray: Consolidation(0.82), Opacity(0.65) → 폐렴 키워드 매칭 → 0.65
- Lab: WBC=14.5 (leukocytosis), CRP=45 (elevated) → 2/3 패턴 일치 → 0.67
- 최종: 0.25×0.67 + 0.50×0.65 + 0.25×0.67 = **0.659**

### 희귀질환 스코어링 — LIRICAL (`lirical_scorer.py`)

**LIRICAL**(Likelihood Ratio Interpretation of Clinical AbnormaLities)은 논문(Jacobsen et al. 2020)에서 제안된 알고리즘입니다.

핵심 공식:
```
LR 점수 = 있는 증상들의 LR 곱 × 없는 증상들의 LR 곱

있는 증상 LR = P(이 증상이 이 질환에서 나타날 확률) / P(일반인에서 이 증상 확률)
없는 증상 LR = P(이 증상이 이 질환에서 없을 확률) / P(일반인에서 이 증상 없을 확률)
```

Orphanet 빈도 가중치 (`knowledge_base.py`의 `freq_weight_map`):

| Orphanet 빈도 라벨 | 가중치(sensitivity) |
|-------------------|---------------------|
| Always (100%) | 1.0 |
| Very frequent (99-80%) | 0.9 |
| Frequent (79-30%) | 0.5 |
| Occasional (29-5%) | 0.1 |
| Unknown | 0.3 |

LR 점수가 높을수록 "이 질환일 가능성이 높다"는 의미.
LR = 1.0 이면 "일반인과 다르지 않다" → 의미 없음.
LR > 10 → "이 증상 조합은 이 질환 환자에서 일반인보다 10배 더 자주 나타남"

> 💡 **초보자 가이드** — 왜 "있는 증상"과 "없는 증상"을 둘 다 쓰나?
> "기침이 없다"는 사실도 중요한 진단 정보입니다. 예를 들어 기관지염 환자는 거의 다 기침이 있는데 환자에게 기침이 없다면 → 기관지염일 확률이 크게 떨어집니다. LIRICAL은 이걸 수치로 반영합니다.

---

## 8. 프롬프트 시스템 — AI에게 어떻게 말하나

### 시스템 프롬프트 (`rag_pipeline.py:79~167`)

AI의 역할과 제약을 정의하는 "직업기술서"입니다. 매번 고정으로 전달됩니다.

**핵심 규칙 (10가지)**:
1. 근거 없는 추측 금지
2. 모든 주장은 RAG 데이터 또는 공인 가이드라인 기반
3. 희귀질환이 Top 3 안에 있으면 MDT 협진 권고 필수
4. 로컬 DB + 글로벌 API 교차검증 결과 우선
5. Negative HPO로 배제 진단 논리 설명
6. PubMed 케이스리포트와 현재 환자 비교·대조
7. ClinicalTrials 데이터로 임상시험 참여 권고
8~10. 출력 필드별 세부 작성 기준 (MRN 포함 금지, 유전자명 명시 등)

> 💡 **초보자 가이드** — MRN은 왜 막는가?
> MRN(Medical Record Number)은 환자 고유 식별번호입니다. 개인정보보호법(HIPAA, 한국 개보법)에 따라 AI 프롬프트·출력에 포함되면 안 됩니다. `_build_user_prompt()`는 환자 정보를 조립할 때 `safe_info` 딕셔너리를 만들면서 MRN 필드를 의도적으로 제외합니다.

**출력 포맷** (JSON 고정):
```json
{
  "recommendation": {
    "immediate_workup": ["즉시 시행할 검사 목록"],
    "specialist_referral": ["협진 권고"],
    "treatment_guideline": ["[질환명] 치료 가이드라인"],
    "genetic_test": ["유전자 검사 권고"],
    "additional_lab": ["추가 혈액검사 권고"]
  },
  "clinical_notes": {
    "summary": "환자 종합 요약",
    "top1_reasoning": "1순위 진단 근거",
    "differential_note": "2~3순위 감별진단",
    "rag_evidence": "RAG에서 찾은 주요 근거",
    "case_comparison": "PubMed 사례와 비교",
    "epidemiology_note": "역학 정보 (희귀질환만)",
    "disclaimer": "AI 결과는 진단 보조이며..."
  }
}
```

### 유저 프롬프트 — 8개 섹션 구조 (`rag_pipeline.py:642~746`)

매 환자마다 동적으로 생성됩니다. 8개 섹션으로 구성됩니다:

```
=== 1. 환자 기본정보 ===
{"name": "익명", "age": 40, "sex": "F", ...}
※ MRN(환자번호)은 절대 포함 안 됨

=== 2. 증상 원문 ===
- Positive Findings: "40세 여성. 3주째 지속되는 호흡곤란..."
- Negative Findings: "기침은 없으며 발열도 없습니다."

=== 2-1. X-ray 예측 결과 (SooNet 모델, 확률값 Top 10) ===
  - Pleural Effusion: 0.823 → HPO: HP:0002202
  - Consolidation: 0.654 → HPO: HP:0032177
  ...
※ 임계값(0.3) 이상인 항목이 HPO로 변환되어 섹션 3에 반영됨

=== 3. HPO 프로파일 ===
Positive HPO:
  - HP:0002094 (symptom)   ← 어디서 추출됐는지 source 명시
  - HP:0002202 (xray)
  - HP:0012418 (lab)
Negative HPO:
  - HP:0012735

=== 4. Lab 수치 ===
  - WBC: 12.5
  - HGB: 9.8
  - LDH: 310

=== 5. 일반/기타 폐질환 랭킹 Top 10 (로컬 DB 기반) ===
 1. 지역사회획득 폐렴 (score=0.659)
 2. 폐결핵 (score=0.521)
 ...

=== 6. 희귀폐질환 리스팅 (로컬 DB 기반) ===
 1. Lymphangioleiomyomatosis (ORPHA:723, LR=45.2)
 2. ...

=== 7. 내부 DB 정보 — Top 3 교차검증용 ===
Top 1 교차검증: DB·API 교차검증 일치 (TSC1, TSC2)

=== 8. RAG 검색 결과 (외부 API) ===
--- Top 1: Lymphangioleiomyomatosis (ORPHA:723) ---
[Orphanet]
- 유전자: TSC1 (Disease-causing germline mutation), TSC2 (...)
- Very frequent / Frequent HPO: Dyspnea [Very frequent], Cough [Frequent]
- 유병률: 1-9/100000
- 발병연령: Adult
- 유전 양식: Autosomal dominant

[Monarch]
- 인과 유전자: TSC1, TSC2
- Orphanet 교차검증: DB·API 교차검증 일치

[PubMed 케이스리포트]
  - PMID:38765432 | A case of LAM presenting with... (2024)
    Background: Lymphangioleiomyomatosis (LAM) is a rare...

[PubCaseFinder]
  - Lymphangioleiomyomatosis (score=0.95, genes=TSC1,TSC2)

[ClinicalTrials (RECRUITING)]
  - NCT05123456 | Sirolimus for LAM...
    Phase:PHASE2 | Status:RECRUITING
```

이 모든 내용이 합쳐져서 AI에게 전달되면, AI는 참고자료를 토대로 JSON 소견서를 작성합니다.

---

## 9. 출력 JSON — AI가 내놓는 최종 결과물

Bedrock Sonnet 3.5가 생성하는 결과 예시 (`_parse_json_response` → `_validate_schema` 검증 통과):

```json
{
  "recommendation": {
    "immediate_workup": [
      "흉부 CT (고해상도)",
      "동맥혈 가스 분석",
      "흉막액 세포학 검사"
    ],
    "specialist_referral": [
      "호흡기내과 및 유전학과 MDT 협진 권고 (희귀질환 후보)",
      "영상의학과 판독 자문"
    ],
    "treatment_guideline": [
      "[Lymphangioleiomyomatosis] Sirolimus(라파마이신) 2mg/day 시작 권고 — ATS/JRS 2016 가이드라인",
      "[지역사회획득 폐렴] 아목시실린/클라불라네이트 경험적 항생제 — IDSA 2019",
      "..."
    ],
    "genetic_test": [
      "TSC1/TSC2 시퀀싱 권고 (복수 소스 확인)"
    ],
    "additional_lab": [
      "VEGF-D 혈청 농도 측정"
    ]
  },
  "clinical_notes": {
    "summary": "40세 여성 환자가 3주째 지속되는 호흡곤란과 우측 흉통을 주소로 내원. WBC 12.5, SpO2 92%, FEV1 68%로 경도의 저산소증과 제한성 환기 장애 소견...",
    "top1_reasoning": "Positive HPO: 호흡곤란(symptom), 흉막삼출(xray), 저산소증(lab). Negative HPO: 기침, 발열. LAM의 Very frequent HPO와 일치하며 중년 여성에서의 급성 호흡곤란은 LAM 초기 증상으로 보고된 바 있음...",
    "differential_note": "Top 2: IPF (폐섬유화 패턴 부족으로 가능성 낮음). Top 3: CAP (발열·기침 부재로 가능성 낮지만 배제 불가).",
    "rag_evidence": "Orphanet 유전자(TSC1, TSC2)와 Monarch causal_genes가 일치 → DB·API 교차검증 일치. Orphanet 유병률 1-9/100000로 희귀질환. PubMed PMID:38765432에 유사 사례 보고...",
    "case_comparison": "PubMed 케이스 사례와 환자 나이·성별·증상 양상 유사. 차이점은 본 환자가 체중감소를 동반...",
    "epidemiology_note": "LAM: 유병률 1-9/100000, 발병연령 Adult, 유전양식 Autosomal dominant (DB·API 일치)",
    "disclaimer": "AI 결과는 진단 보조이며 최종 진단은 주치의의 임상 판단과 추가 검사 결과를 종합하여 확정합니다."
  }
}
```

---

## 10. 데이터 흐름 요약표

| 단계 | 입력 | 처리 모듈 | 출력 |
|------|------|-----------|------|
| Phase 1 | 증상 텍스트 | `bedrock_extractor.py` (Bedrock Haiku) | Positive/Negative HPO |
| Phase 2 | X-ray 이미지 | `soo_net.py` (SooNet 모델) | X-ray HPO + 확률 |
| Phase 3 | 혈액검사 수치 | `lab_rules.py` (Rule-based) | 이상 HPO 코드 |
| 스코어링-일반 | HPO + X-ray + Lab | `general_disease_scorer.py` | 일반 질환 Top 10 |
| 스코어링-희귀 | HPO | `lirical_scorer.py` | 희귀질환 LR 리스팅 |
| Top 3 통합 | 일반 Top10 + 희귀 리스팅 | `rag_pipeline.py` (Bedrock Haiku, 실패 시 휴리스틱 폴백) | Top 3 후보 |
| API 호출 | Top 3 disease_name / OrphaCode | 5개 fetcher (병렬) | 논문/유전자/임상시험 |
| 교차검증 | Orphanet 유전자 + Monarch 유전자 | `orphanet_fetcher.cross_validate_genes` | 일치/불일치 판정 |
| 소견서 생성 | 8개 섹션 통합 프롬프트 | Bedrock Sonnet 3.5 | JSON 소견서 |
| 스키마 검증 | 생성된 JSON | `_validate_schema` | 필드 누락 경고 |

---

## 11. 실패 대비 설계

각 컴포넌트는 실패해도 **파이프라인이 중단되지 않도록** 설계되어 있습니다.

| 컴포넌트 | 실패 시 처리 |
|----------|-------------|
| PubCaseFinder API | 로컬 orphadata_weighted.csv에서 HPO 매칭으로 대체 |
| ClinicalTrials.gov | 빈 리스트 반환, 파이프라인 계속 |
| Monarch HPO 이름 조회 | 로컬 딕셔너리 25개 HPO 폴백, 없으면 코드 그대로 |
| Monarch 유전자 조회 | 빈 리스트 반환, 교차검증 결과 "데이터 없음" |
| Bedrock API | ClientError 포착 → `{"error": ...}` 반환 |
| 전체 API 병렬 호출 | `_safe_call()` wrapper로 감싸서 None 반환 |
| Orphanet XML 없음 | 빈 dict 반환, 해당 섹션 "데이터 없음" 표시 |
| 일반질환 YAML 없음 | 빈 리스트 반환, 섹션 건너뜀 |
| LLM 출력 JSON 파싱 실패 | `_parse_json_response` 실패 → raw_output 그대로 반환 |
| JSON 스키마 필드 누락 | `_validate_schema`가 경고 로그 (파이프라인은 계속) |

```python
# rag_pipeline.py:551~558 — 모든 API 호출을 감싸는 안전 wrapper
@staticmethod
def _safe_call(api_name: str, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"     ⚠️  {api_name} 호출 실패: {e}")
        return None  # None 반환으로 파이프라인 계속
```

**조건부 분기** (케이스 A/B/C):
- **케이스 A**: 희귀 리스팅 있음 → Orphanet + PubCaseFinder 수집
- **케이스 B**: Top 3에 OrphaCode 있지만 별도 리스팅 없음 → 동일하게 수집
- **케이스 C**: Top 3 모두 일반 질환 → Orphanet + PubCaseFinder 스킵, Monarch/PubMed/CT만 수집

이렇게 하면 일반 질환 환자에겐 불필요한 API 호출을 줄이고, 희귀질환 의심 환자에겐 최대한 많은 데이터를 수집합니다.

---

## 12. 실행 방법 & 파일별 역할 (부록)

### 실행 방법

```bash
cd aws_say2_project_vision
python rag_pipeline.py
```

`rag_pipeline.py` 하단의 `if __name__ == "__main__":` 블록에서 내장 샘플 환자로 전체 5단계를 실행합니다.

외부에서 사용할 때:
```python
from rag_pipeline import RareLinkPipeline

pipeline = RareLinkPipeline(
    vision_model_path="model/chexnet_unet_crop_best.pth",
)

report = pipeline.run(
    patient_info  = {"name": "익명", "age": 40, "sex": "F", ...},
    xray_path     = "path/to/xray.jpg",
    symptom_text  = "호흡곤란, 흉통...",
    negative_text = "기침·발열 없음",
    lab_results   = {"WBC": 12.5, "SpO2": 92.0, ...},
)
```

### 환경 변수 & 사전 준비

- `.env` 파일 (자동 로드)에 AWS 인증 정보 필요 (Bedrock 호출용)
- AWS Bedrock Model access에서 다음 3개 모델 활성화:
  - `claude-3-haiku-20240307` (Phase 1: HPO 추출)
  - `claude-3-5-haiku-20241022` (Phase 4: Top 3 통합)
  - `claude-3-5-sonnet-20241022-v2` (Phase 5: 최종 소견서)
- 데이터 파일 준비:
  - `data/orphadata_weighted.csv` (없으면 `knowledge_base.py`로 생성)
  - `data/en_product*.xml` (Orphanet 공개 XML)
  - `data/lung_disease_profiles_v2.yaml` (일반 질환 프로파일)
  - `model/chexnet_unet_crop_best.pth` (SooNet 가중치)

### 파일별 역할

```
aws_say2_project_vision/
├── rag_pipeline.py          ← 전체 5단계 오케스트레이터 (메인 진입점)
├── soo_net.py               ← Phase 2: X-ray → 라벨 확률 + HPO (SooNet 딥러닝)
│
└── rag/
    ├── bedrock_extractor.py     ← Phase 1: 증상 텍스트 → HPO (Bedrock Haiku)
    ├── lab_rules.py             ← Phase 3: 혈액검사 수치 → HPO (Rule-based)
    ├── knowledge_base.py        ← Orphanet XML → orphadata_weighted.csv 빌더
    ├── lirical_scorer.py        ← 희귀질환 LR 스코어링 (LIRICAL 방식)
    ├── general_disease_scorer.py ← 일반 폐질환 가중치 스코어링
    ├── pubcasefinder.py         ← API ①: HPO → 희귀질환 랭킹 (DBCLS 일본)
    ├── orphanet_fetcher.py      ← API ②: OrphaCode → 유전자/역학 (로컬 XML)
    ├── monarch_fetcher.py       ← API ③: HPO 이름 변환 + 인과 유전자 (Monarch)
    ├── pubmed_fetcher.py        ← API ④: 질환명 → 최신 논문 (NIH PubMed)
    ├── clinicaltrials_fetcher.py ← API ⑤: 질환명 → 모집 중 임상시험
    └── ragas_eval.py            ← RAG 품질 자동 평가 (선택 사용)
```

---

*이 보고서는 `rag_pipeline.py` 및 `rag/` 패키지 전체 코드를 2026-05-06 기준으로 재검증 후 작성되었습니다.*
*코드 변경 시 해당 섹션을 업데이트하세요.*
