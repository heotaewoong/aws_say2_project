# RAG 구현 보고서
## Rare-Link AI — 희귀 폐질환 진단 보조 시스템

작성일: 2026-04-29 (최종 업데이트)

---

## 1. RAG란 무엇인가

RAG(Retrieval-Augmented Generation)는 LLM(Claude)이 답변을 생성할 때 외부 지식을 실시간으로 검색해서 주입하는 방식이다.

```
[일반 LLM]
  질문 → Claude → 답변 (학습 데이터만 사용, 2024년 이전 지식)

[RAG]
  질문 → 외부 DB 검색 → 검색 결과 + 질문 → Claude → 근거 있는 답변
```

이 프로젝트에서 RAG는 "환자 증상 → 희귀질환 매칭 → 최신 논문 + 임상시험 기반 치료 리포트 생성"에 사용된다.

---

## 2. 전체 RAG 파이프라인 구조

```
환자 입력 (X-ray + 혈액검사 + 임상소견)
        ↓
① INPUT — 3가지 소스에서 HPO 코드 추출
  ├─ X-ray → SooNet (DenseNet-121) → 14개 소견 → HPO 코드
  ├─ 임상소견 텍스트 → Bedrock Claude Haiku → HPO 코드
  └─ 혈액검사 수치 → Rule-based → HPO 코드
        ↓
② SCORE — LIRICAL 알고리즘으로 희귀질환 랭킹
  └─ Orphanet DB (4335개 질환) × HPO 빈도 가중치 → LR 점수
        ↓
③ TRIG — RAG 실행 여부 판단
  └─ 희귀질환이거나 1/2위 점수 차이 불확실 → RAG 실행
        ↓
④ GEN — 외부 소스 검색 + Claude 리포트 생성
  ├─ PubCaseFinder API → 유사 케이스 검색 [현재 서버 장애 → 로컬 폴백]
  ├─ PubMed E-utilities API → 최신 논문 검색
  ├─ ClinicalTrials.gov API → 현재 모집 중인 임상시험 [신규]
  └─ Claude Bedrock Sonnet → 진단 보조 리포트 생성
        ↓
⑤ OUT — 최종 진단 보조 리포트 (JSON + Markdown)
```

---

## 3. 파일 구조 상세 설명 (초보자용)

```
aws_say2_project_vision/
│
├── rag_pipeline.py          ← 전체 파이프라인 오케스트레이터 (핵심 파일)
├── soo_net.py               ← X-ray 분류 모델 (DenseNet-121 기반)
│
├── data/
│   ├── en_product4.xml      ← Orphanet 원본 XML (희귀질환-HPO 매핑 데이터)
│   ├── orphadata_weighted.csv ← 파싱된 질환-HPO DB (4335개 질환, 115,878행)
│   └── RAG_구현_보고서.md   ← 이 파일
│
├── model/
│   └── chexnet_unet_crop_best.pth ← SooNet 학습된 가중치 파일
│
└── rag/                     ← RAG 관련 모듈 모음
    ├── __init__.py
    ├── bedrock_extractor.py     ← 임상 텍스트 → HPO 코드 변환 (Bedrock Haiku)
    ├── lab_rules.py             ← 혈액검사 수치 → HPO 코드 변환 (규칙 기반)
    ├── lirical_scorer.py        ← LIRICAL 알고리즘 (질환 랭킹 계산)
    ├── knowledge_base.py        ← Orphanet XML → CSV 파서
    ├── pubcasefinder.py         ← PubCaseFinder API + 로컬 폴백 + 캐시
    ├── pubmed_fetcher.py        ← PubMed 최신 논문 검색
    ├── clinicaltrials_fetcher.py ← ClinicalTrials.gov 임상시험 검색 [신규]
    ├── ragas_eval.py            ← PMID 환각 체크 + 품질 평가
    └── valid/                   ← 검증 스크립트 모음
        ├── fetch_mimic_patient.py  ← MIMIC S3에서 환자 데이터 준비
        ├── run_rag_test.py         ← 다중 환자 파이프라인 검증
        └── RAG_통합_검증보고서.md  ← 상세 검증 결과
```

### 각 파일이 하는 일 (초보자용 설명)

**`rag_pipeline.py`** — 전체 흐름을 지휘하는 파일
- 의사가 X-ray, 소견서, 혈액검사를 입력하면 이 파일이 모든 단계를 순서대로 실행
- `RareLinkPipeline` 클래스 하나에 5단계가 다 들어있음
- `pipeline.run(xray_path, symptom_text, lab_results)` 한 줄로 전체 실행

**`soo_net.py`** — X-ray 이미지를 읽는 AI 모델
- DenseNet-121 기반, 흉부 X-ray에서 14가지 소견(폐렴, 흉막삼출 등)을 감지
- 각 소견의 확률값을 HPO 코드로 변환해서 반환

**`rag/bedrock_extractor.py`** — 의사 소견서를 HPO 코드로 변환
- "호흡곤란과 흉통이 있습니다" 같은 자유 텍스트를 Claude Haiku에게 보내서
- "HP:0002094 (호흡곤란), HP:0002202 (흉막삼출)" 같은 표준 코드로 변환

**`rag/lab_rules.py`** — 혈액검사 수치를 HPO 코드로 변환
- WBC > 11.0 → 백혈구 증가 (HP:0011897)
- SpO2 < 95% → 저산소증 (HP:0012418)
- 규칙 기반이라 빠르고 안정적

**`rag/lirical_scorer.py`** — 질환 랭킹을 계산하는 알고리즘
- 환자의 HPO 코드와 Orphanet DB의 4335개 질환을 비교
- "이 증상 조합이 이 질환에서 얼마나 자주 나타나는가"를 수치화
- LR(Likelihood Ratio) 점수가 높을수록 해당 질환일 가능성 높음

**`rag/pubmed_fetcher.py`** — 최신 논문을 검색
- 1위 질환명으로 PubMed에서 최신 논문 3편을 가져옴
- PMID, 제목, abstract, URL을 반환

**`rag/clinicaltrials_fetcher.py`** — 현재 모집 중인 임상시험 검색 (신규)
- 1위 질환명으로 ClinicalTrials.gov에서 현재 모집 중인 임상시험을 검색
- 환자에게 "이 질환으로 현재 임상시험이 있습니다"라는 정보 제공

**`rag/ragas_eval.py`** — 리포트 품질 검증
- 생성된 리포트에서 PMID를 추출해서 실제 PubMed에 존재하는지 확인
- 가짜 논문 인용(환각) 여부를 자동으로 탐지

---

## 4. rag_pipeline.py 동작 방식 상세 (초보자용)

### 4-1. 클래스 초기화 (`__init__`)

```python
pipeline = RareLinkPipeline(
    vision_model_path='model/chexnet_unet_crop_best.pth',
    orphanet_csv_path='data/orphadata_weighted.csv',
)
```

이 한 줄이 실행되면:
1. SooNet 모델 가중치 로드 (X-ray 분석 준비)
2. Bedrock Haiku 클라이언트 초기화 (소견서 HPO 추출 준비)
3. Orphanet CSV 4335개 질환 로드 (LIRICAL 스코어링 준비)
4. Bedrock Sonnet 클라이언트 초기화 (리포트 생성 준비)
5. PubMed 검색 엔진 초기화

### 4-2. Step 1 — HPO 통합 (`step1_get_hpo`)

세 가지 소스에서 HPO 코드를 뽑아서 합친다.

```
X-ray 이미지 → SooNet → [HP:0002095, HP:0002202, ...]
소견서 텍스트 → Bedrock Haiku → Positive: [HP:0002094], Negative: [HP:0002013]
혈액검사 수치 → 규칙 → [HP:0001903, HP:0012418, ...]
                    ↓
          중복 제거 + Negative 우선 처리
                    ↓
     통합 Positive HPO 7개, Negative HPO 3개
```

Negative HPO가 중요한 이유: "기침이 없다"는 정보가 기침이 필수 증상인 질환을 걸러내는 데 사용됨

### 4-3. Step 2 — LIRICAL 스코어링 (`step2_score`)

```
각 질환에 대해:
  LR = ∏(있는 증상의 민감도/배경빈도) × ∏(없는 증상의 1-민감도/1-배경빈도)

예시:
  LAM 질환에서 흉막삼출 민감도 = 0.9, 배경빈도 = 0.05
  → 이 증상이 있으면 LR에 0.9/0.05 = 18배 기여
```

4335개 질환 전부 계산해서 LR 점수 내림차순으로 Top 10 반환

### 4-4. Step 3 — RAG 트리거 판단 (`step3_rag_trigger`)

두 조건 중 하나라도 해당하면 RAG 실행:
- 1위 질환이 희귀질환인 경우 (Orphanet 수록 = 희귀질환)
- 1위/2위 LR 점수 비율이 3.0 미만인 경우 (불확실)

### 4-5. Step 4 — RAG 컨텍스트 수집 + 리포트 생성 (`step4_rag_generate`)

**컨텍스트 수집 순서:**

```
1. PubCaseFinder API 호출 → 실패 시 로컬 Orphanet CSV 폴백
2. PubMed API 호출 → 1위 질환 최신 논문 3편
3. ClinicalTrials.gov API 호출 → 현재 모집 중인 임상시험 3건
```

**프롬프트 구성:**

Claude에게 보내는 프롬프트는 크게 4부분으로 구성된다:

```
[절대 규칙]
- Context 밖 정보 사용 금지
- PMID 없는 정보는 "불확실" 표시
- 환각 금지, 의사 보조 시스템임을 명시

[INPUT DATA]
- 환자 HPO (Positive/Negative)
- LIRICAL 질환 랭킹 Top 5
- Top 3 질환 상세 (ORPHA 코드, LR 점수, 유전자)
- RAG 컨텍스트 (PubCaseFinder + PubMed + ClinicalTrials)

[출력 형식]
- JSON 구조화 출력 먼저 (diagnosis, genetic_test, treatment, insight, next_steps, uncertainty)
- 그 다음 Markdown 리포트 (5개 섹션)

[Self-Check]
- 모든 근거가 context에 있는가?
- PMID가 실제 context에 존재하는가?
- 추측이 포함되어 있는가?
```

**왜 JSON을 먼저 출력하는가?**
- 프론트엔드에서 구조화된 데이터를 파싱해서 UI에 표시하기 위함
- confidence 레벨(HIGH/MEDIUM/LOW)로 신뢰도를 명시
- uncertainty 목록으로 한계를 투명하게 공개

**Bedrock 호출 파라미터:**
- `temperature=0.2` — 낮은 값으로 일관성 있는 출력 (창의성 최소화)
- `top_p=0.9` — 상위 90% 확률 토큰만 사용
- `max_tokens=2000` — 리포트 최대 길이

---

## 5. RAG 구성 요소별 구현 현황

### 5-1. Hard data — Orphanet 희귀질환 DB

| 항목 | 내용 |
|------|------|
| 소스 | Orphanet en_product4.xml |
| 변환 | knowledge_base.py → orphadata_weighted.csv |
| 데이터 | 115,878행 (질환-HPO 매핑), 4335개 질환 |
| 활용 | LIRICAL 스코어링의 빈도 가중치 |
| 상태 | ✅ 완료 · 검증됨 |

HPO 빈도 가중치 매핑:
- Always(100%) → 1.0
- Very frequent(99-80%) → 0.9
- Frequent(79-30%) → 0.5
- Occasional(29-5%) → 0.1

### 5-2. Soft data — PubCaseFinder API

| 항목 | 내용 |
|------|------|
| 소스 | DBCLS PubCaseFinder API |
| URL | https://pubcasefinder.dbcls.jp/api/pcf_get_ranked_list |
| 입력 | HPO 코드 목록 |
| 출력 | 희귀질환 랭킹 + 관련 유전자 |
| 상태 | ⚠️ 서버 장애 (404) — 로컬 Orphanet CSV 폴백 자동 동작 |

> 2026-04-29 확인: 메인 사이트(200)는 살아있으나 API 전체 404 반환 중. 서버 측 문제로 우리가 수정 불가. 로컬 폴백으로 파이프라인 중단 없음.

### 5-3. 최신 논문 — PubMed E-utilities API

| 항목 | 내용 |
|------|------|
| 소스 | NIH/NLM PubMed E-utilities |
| 의존성 | requests만 필요 |
| 입력 | 질환명 |
| 출력 | 최신 논문 Top K (PMID + 제목 + abstract + URL) |
| 상태 | ✅ 완료 · 검증됨 |

### 5-4. 임상시험 — ClinicalTrials.gov API (신규)

| 항목 | 내용 |
|------|------|
| 소스 | NIH ClinicalTrials.gov API v2 |
| URL | https://clinicaltrials.gov/api/v2/studies |
| 입력 | 질환명 |
| 출력 | 현재 모집 중인 임상시험 (NCT ID, 단계, 기관, URL) |
| 비용 | 무료, 등록 불필요 |
| 상태 | ✅ 완료 · 검증됨 |

### 5-5. HPO 추출 — Bedrock Claude Haiku

| 항목 | 내용 |
|------|------|
| 모델 | anthropic.claude-3-haiku-20240307-v1:0 (APAC) |
| 입력 | 임상 소견 자유 텍스트 (한국어/영어) |
| 출력 | Positive HPO + Negative HPO 코드 목록 |
| 상태 | ✅ 완료 · 검증됨 |

### 5-6. Lab → HPO 변환 — Rule-based

| 항목 | 내용 |
|------|------|
| 방식 | 정상범위 기반 규칙 |
| 커버리지 | 17개 검사항목 (CBC, 폐기능, 산소화 등) |
| 상태 | ✅ 완료 · 검증됨 |

### 5-7. LIRICAL 스코어링

| 항목 | 내용 |
|------|------|
| 알고리즘 | Likelihood Ratio 곱 (Robinson et al. 2020) |
| 입력 | Positive HPO + Negative HPO + Orphanet DB |
| 출력 | 희귀질환 Top 10 랭킹 (LR 점수) |
| 정확도 | Recall@1 = 81.6%, Recall@10 = 98.3% (4293개 전수 테스트) |
| 상태 | ✅ 완료 · 검증됨 |

### 5-8. 리포트 생성 — Bedrock Claude Sonnet

| 항목 | 내용 |
|------|------|
| 모델 | apac.anthropic.claude-3-5-sonnet-20241022-v2:0 |
| 출력 | JSON 구조화 + Markdown 리포트 |
| 내용 | ①질환 평가 ②유전자 검사 권고 ③치료 가이드라인 ④최신 동향 ⑤다음 단계 |
| Safety | confidence 레벨 명시, 불확실성 목록, HITL 배너 |
| 상태 | ✅ 완료 · 검증됨 |

### 5-9. PMID 환각 체크 — ragas_eval.py

| 항목 | 내용 |
|------|------|
| 방식 | PubMed API로 PMID 실존 여부 자동 확인 |
| 버그 수정 | `"error" not in entry` 조건 추가 (가짜 PMID 오판정 방지) |
| 상태 | ✅ 완료 · 검증됨 (유효율 100%) |

---

## 6. 프롬프트 설계 원칙

### 왜 이렇게 프롬프트를 설계했는가?

**1. Evidence-bound (근거 기반 제약)**

의료 AI에서 가장 위험한 것은 없는 논문을 인용하거나 근거 없는 치료를 권고하는 것이다. 이를 막기 위해:
- "Context 밖 정보 사용 금지"를 절대 규칙으로 명시
- PMID 없는 정보는 반드시 "불확실"로 표시하도록 강제
- Self-Check 단계를 프롬프트 끝에 추가해서 Claude가 스스로 검토하도록 유도

**2. JSON + Markdown 이중 출력**

```json
{
  "diagnosis": [...],
  "confidence": "MEDIUM",
  "uncertainty": ["치료 가이드라인 부족", ...]
}
```

JSON은 프론트엔드에서 파싱해서 UI에 표시하고, Markdown은 의사가 읽는 용도다. 두 형식을 동시에 출력해서 개발자와 의사 모두를 지원한다.

**3. Negative HPO 반영 강제**

"기침이 없다"는 정보를 감별진단에 반드시 반영하도록 프롬프트에 명시했다. 의사가 차팅할 때는 있는 증상만 기록하지만, 없는 증상도 진단에 중요하기 때문이다.

**4. temperature=0.2 (낮은 창의성)**

의료 리포트는 창의적일 필요가 없다. 일관성 있고 예측 가능한 출력이 중요하므로 temperature를 낮게 설정했다.

---

## 7. 검증 결과 (2026-04-29 최종)

### 7-1. 다중 환자 end-to-end 테스트

MIMIC 실환자 3명으로 전체 파이프라인 실행:

| 환자 | X-ray HPO | LIRICAL 1위 | PubMed | ClinicalTrials | PMID 유효율 |
|---|---|---|---|---|---|
| 10000032 | Atelectasis 1개 | Mucopolysaccharidosis-like (LR=124) | 0편 | 0건 | 인용없음 |
| 10000764 | Atelectasis + Lung Opacity + Pleural Effusion 3개 | Kaposiform lymphangiomatosis (LR=3888) | 3편 | 3건 | **3/3 = 100%** |
| 10000898 | Atelectasis + Pleural Effusion 2개 | Acute interstitial pneumonia (LR=6480) | 3편 | 2건 | **2/2 = 100%** |

**결과: 3/3 환자 PASS, PMID 환각 0건**

### 7-2. LIRICAL 알고리즘 정확도 (4293개 전수)

| 지표 | 값 |
|---|---|
| Recall@1 | 81.6% |
| Recall@3 | 94.2% |
| Recall@5 | 96.4% |
| Recall@10 | 98.3% |
| MRR | 0.8825 |

### 7-3. 임상 시나리오 5개

| 지표 | 결과 |
|---|---|
| Recall@1 | 4/5 (80%) |
| Recall@3 | 5/5 (100%) |
| Faithfulness | 0.62 / 1.0 |
| Answer Relevancy | 0.86 / 1.0 |
| PMID 유효율 | 100% |

---

## 8. 현재 한계 및 개선 방향

### 한계
1. PubCaseFinder API 서버 장애 — 유전자 정보 누락 (로컬 폴백으로 파이프라인은 동작)
2. Faithfulness 0.62 — 감별진단 설명 시 컨텍스트 이탈 발생
3. X-ray HPO 추출 — MIMIC 이미지에서 threshold 0.3 기준 1~3개만 검출

### 개선 방향
1. Orphanet en_product6.xml 파싱으로 유전자 정보 직접 확보 (PubCaseFinder 대체)
2. 상위 3개 질환 모두에 대해 PubMed 검색 확장 → Faithfulness 0.8+ 달성
3. HPO API (Monarch Initiative) 추가 → 리포트 가독성 향상
4. SooNet 모델 재학습 후 X-ray HPO 검출률 개선

---

## 9. 실행 방법

```bash
cd aws_say2_project_vision

# AWS 키 설정 (환경변수로)
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="ap-northeast-2"

# 1. 전체 파이프라인 실행 (내장 샘플)
python rag_pipeline.py

# 2. MIMIC 실환자 데이터 준비 + 검증
python rag/valid/fetch_mimic_patient.py
python rag/valid/run_rag_test.py

# 3. 개별 컴포넌트 테스트
python rag/pubmed_fetcher.py              # PubMed 검색
python rag/clinicaltrials_fetcher.py      # 임상시험 검색
python rag/ragas_eval.py                  # PMID 환각 체크
```

---

## 10. 사용 중인 외부 API 요약

| API | 역할 | 상태 | 비용 |
|-----|------|------|------|
| AWS Bedrock (Claude Haiku) | 임상 텍스트 → HPO 추출 | ✅ 정상 | AWS 크레딧 |
| AWS Bedrock (Claude Sonnet) | 진단 리포트 생성 | ✅ 정상 | AWS 크레딧 |
| PubMed E-utilities | 최신 논문 검색 | ✅ 정상 | 무료 |
| ClinicalTrials.gov v2 | 모집 중 임상시험 | ✅ 정상 | 무료 |
| PubCaseFinder | 케이스 기반 교차검증 | ⚠️ 서버 장애 | 무료 |
| Orphanet (로컬 CSV) | 질환-HPO 매핑 | ✅ 정상 | 로컬 |
