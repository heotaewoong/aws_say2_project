# Rare-Link AI — RAG 파이프라인 통합 검증 보고서

**작성일:** 2026-04-29  
**작성자:** 허태웅  
**모델:** apac.anthropic.claude-3-5-sonnet-20241022-v2:0 (ap-northeast-2 Bedrock)

---

## 결론 먼저

**RAG 파이프라인은 정상 작동합니다.**

- 5단계 전체 오류 없이 완주
- LIRICAL 알고리즘 Recall@10 = 98.3% (4293개 전수 테스트)
- 임상 시나리오 5개 중 4개 1위, 5개 전부 3위 안
- 생성된 소견서 PMID 환각 없음 (유효율 100%)

---

## 1. 검증 구조

| 검증 레벨 | 방법 | 핵심 질문 |
|---|---|---|
| 컴포넌트 단위 | 4개 모듈 개별 실행 | 각 부품이 혼자서 작동하는가 |
| 파이프라인 통합 | MIMIC 실환자 + 내장 샘플 | 5단계 전체가 연결되어 돌아가는가 |
| 알고리즘 정확도 | 4293개 질환 전수 자동 테스트 | LIRICAL이 정답을 몇 위로 찾는가 |
| 임상 시나리오 | 희귀 폐질환 5개 케이스 | 실제 의료 상황에서 성능이 나오는가 |
| 소견서 품질 | PMID 환각 체크 + Bedrock 평가 | 생성된 리포트를 신뢰할 수 있는가 |

---

## 2. 컴포넌트 단위 테스트

### LIRICAL 스코어링

입력: Positive HPO 3개 (흉막삼출·호흡곤란·저산소증), Negative HPO 1개 (폐섬유화 없음)

| 순위 | 질환 | LR 점수 |
|---|---|---|
| 1 | Lymphangioleiomyomatosis (LAM) | **1657.89** |
| 2 | Idiopathic Pulmonary Fibrosis (IPF) | 19.59 |

**PASS** — LR 비율 84.6배. Negative HPO가 IPF 점수를 낮추는 방향으로 정확히 반영됨.

---

### PubMed 논문 검색

입력: `"Lymphangioleiomyomatosis"`, top_k=3

| PMID | 제목 | 연도 |
|---|---|---|
| 42021993 | Recurrent catamenial pneumothorax in an adolescent... | 2026 |
| 41964900 | PEComas: Current Concepts in Diagnosis... | 2026 Apr |
| 41953042 | Bilateral Lung Transplantation for LAM... | 2026 |

**PASS** — PubMed E-utilities 공개 API 정상 응답, 최신 논문 3편 수집.

---

### PMID 환각 체크

| PMID | 상태 |
|---|---|
| 32386464 | ✅ 실존 |
| 99999999 | ❌ 가짜 정확히 탐지 |

**PASS** — 버그 수정 완료 및 실제 동작 확인됨

> 수정 내역: PubMed API가 없는 PMID에도 `uid` 필드를 반환하고 `error` 키를 추가함. 기존 코드는 `uid == pmid`만 체크해 가짜 PMID를 유효로 오판정. `"error" not in entry` 조건 추가로 수정 완료. (`ragas_eval.py:175`)  
> **2026-04-29 실행 결과**: 32386464 ✅ 실존 / 99999999 ❌ 가짜 탐지 — 정상 동작 확인

---

### PubCaseFinder

**미작동 (외부 서버 장애)** — 로컬 Orphanet CSV 폴백 자동 동작

> 2026-04-29 직접 확인: 메인 사이트(200)는 살아있으나 API 전체 404 반환 중.  
> 공식 API spec 기준 올바른 엔드포인트(`/api/pcf_get_ranked_list`)로 요청해도 동일하게 404.  
> nginx 서버는 응답하지만 API 라우팅 자체가 비활성화된 상태 — 서버 측 문제로 우리가 수정 불가.  
> 로컬 Orphanet CSV 폴백 정상 동작 → 파이프라인 중단 없음. 유전자 정보만 누락.  
> **향후 대응**: Orphanet `en_product6.xml` 파싱으로 유전자 정보 직접 확보 예정.

---

## 3. 파이프라인 통합 테스트

### 테스트 A — MIMIC 실환자 (subject_id=10000032)

| 입력 | 값 |
|---|---|
| X-ray | MIMIC-CXR s50414267 (1174KB), S3 `say1-pre-project-5` |
| 소견서 | MIMIC-IV discharge.csv 스트리밍 추출 |
| Lab | WBC=10.5, HGB=11.2, SpO2=94.0, CRP=8.5 |

| 단계 | 결과 |
|---|---|
| ① X-ray HPO | Atelectasis (HP:0100750) 1개 |
| ① NLP HPO | Positive 6개 · Negative 6개 |
| ① Lab HPO | Anemia, Hypoxemia, Elevated CRP 3개 |
| ② LIRICAL 1위 | Mucopolysaccharidosis-like syndrome (LR=123.96) |
| ③ RAG 트리거 | True (희귀질환) |
| ④ Claude 소견서 | 생성 완료 — confidence=LOW, 불확실성 3개 명시 |
| PMID 환각 | 인용 없음 → 환각 없음 |

**PASS** — 5단계 오류 없이 완주. Safety Layer 정상 동작.

> 참고: 이 환자의 실제 진단은 HCV 간경화 + 복수 (복부 팽만 호소). 희귀 폐질환 DB 범위 밖이므로 랭킹 불일치는 예상된 결과. 검증용 환자는 ICD-10 폐질환 코드로 필터링 필요.

---

### 테스트 B — 내장 샘플 (40세 여성 호흡곤란)

| 입력 | 값 |
|---|---|
| 증상 | 40세 여성, 3주 호흡곤란·우측 흉통, 체중 감소, 기침·발열 없음 |
| Lab | WBC=12.5, HGB=9.8, LDH=310, CRP=7.2, SpO2=92.0, FEV1=68.0 |

| 단계 | 결과 |
|---|---|
| ② LIRICAL 1위 | Cryptogenic organizing pneumonia (LR=2991.69) |
| ② LIRICAL 2위 | Anti-GBM disease (LR=2907.92) — 1/2위 비율 1.03 |
| ③ RAG 트리거 | True (희귀질환 + 불확실) |
| ④ PubMed | **3편 수집** (COP 역학, 기관지경 진단, 암 관련 폐합병증) |
| ④ Claude 소견서 | 생성 완료 — confidence=MEDIUM |

**PASS**

---

## 4. 알고리즘 정확도 — LIRICAL 전수 테스트 (4293개 질환)

각 질환의 대표 HPO 상위 3개를 입력하고 해당 질환이 몇 위로 랭킹되는지 전수 측정.

| 지표 | 값 | 해석 |
|---|---|---|
| **Recall@1** | **81.6%** | 대표 HPO 3개로 10회 중 8회 정답 1위 |
| **Recall@3** | **94.2%** | 10회 중 9회 이상 3위 안에 정답 포함 |
| **Recall@5** | **96.4%** | 임상 감별진단 Top-5에 거의 항상 포함 |
| **Recall@10** | **98.3%** | 4293개 중 단 35개만 Top-10 밖 |
| **MRR** | **0.8825** | 평균 정답 순위 1.13위 |

Recall@1이 100%가 아닌 이유: 여러 질환이 동일 HPO를 공유할 때 LR이 비슷해 순위가 뒤집힘. 알고리즘 버그가 아닌 HPO 특이도의 구조적 한계. Top-10 밖 35개는 "발달 지연 + 지적 장애"처럼 수백 개 질환이 동일 HPO를 가진 경우.

---

## 5. 임상 시나리오 5개 케이스

의학 교과서 기반 희귀 폐질환 5개. 실제 DB HPO 상위값 + 임상 텍스트 입력.

| # | 질환 | ORPHA | 정답 랭킹 | @1 | @3 | Faithfulness | Relevancy |
|---|---|---|---|---|---|---|---|
| 1 | Lymphangioleiomyomatosis (LAM) | 538 | **1위** | ✅ | ✅ | 0.80 | 0.80 |
| 2 | Idiopathic Pulmonary Fibrosis (IPF) | 2032 | **1위** | ✅ | ✅ | 0.80 | 0.90 |
| 3 | Sarcoidosis | 797 | **1위** | ✅ | ✅ | 0.50 | 0.90 |
| 4 | Pulmonary Arterial Hypertension (PAH) | 422 | **1위** | ✅ | ✅ | 0.50 | 0.80 |
| 5 | Cryptogenic Organizing Pneumonia (COP) | 1302 | **2위** | ❌ | ✅ | 0.50 | 0.90 |

| 지표 | 결과 |
|---|---|
| Recall@1 | **4/5 (80%)** |
| Recall@3 | **5/5 (100%)** |
| Faithfulness 평균 | **0.62 / 1.0** |
| Answer Relevancy 평균 | **0.86 / 1.0** |

COP 2위 이유: COP의 핵심 HPO(기침, 폐경화, ESR 상승)를 Staphylococcal necrotizing pneumonia(ORPHA:36238)가 더 높은 가중치로 보유. "항생제 무반응 + 스테로이드 반응"은 HPO로 표현 불가 → LIRICAL 구조적 한계.

Faithfulness 0.5 케이스: Claude가 2·3위 감별진단 설명 시 PubMed 컨텍스트 밖 내용 포함. 상위 3개 질환 모두에 대해 PubMed 검색을 확장하면 0.8+ 달성 가능.

---

## 6. 소견서 품질 검증

### PMID 환각 체크 (테스트 B 소견서)

| PMID | 상태 | 논문 제목 |
|---|---|---|
| 41812990 | ✅ 실존 | The Epidemiology of Cryptogenic Organizing Pneumonia |
| 41664670 | ✅ 실존 | When PET Is Misleading: Ion™ Robotic Bronchoscopy |
| 41901659 | ✅ 실존 | Pulmonary Complications of Cancer Therapy |

**PMID 유효율: 3/3 = 100%** — 환각 없음

### Bedrock 자동 품질 평가

> RAGAS 라이브러리 Python 3.14 비호환으로 Claude Haiku(Bedrock)로 동일 지표 직접 구현.

| 지표 | 점수 | 평가 내용 |
|---|---|---|
| **Faithfulness** | **0.80 / 1.0** | 주요 주장 대부분 PubMed 컨텍스트에 근거 |
| **Answer Relevancy** | **0.80 / 1.0** | 희귀 폐질환 감별진단 질문에 직접 답변 |

---

## 7. 코드 수정 이력

| 파일 | 수정 내용 | 이유 |
|---|---|---|
| `rag/ragas_eval.py:175` | PMID 검증 버그 수정 (`"error" not in entry` 추가) | PubMed API가 없는 PMID에도 uid 반환 |
| `rag/ragas_eval.py` | `evaluate_with_bedrock()` 함수 추가 | RAGAS Python 3.14 비호환 대체 |
| `rag/pubcasefinder.py:23` | API URL 수정 (`get_ranked_list` → `pcf_get_ranked_list`) | 공식 spec 기준 엔드포인트 변경 |
| `rag_pipeline.py:29` | X-ray 임계값 수정 (`XRAY_THRESHOLD = 0.4 → 0.3`) | 0.4 기준 시 모든 케이스 HPO 0개 |

---

## 8. 잔여 이슈

| 이슈 | 심각도 | 현재 상태 |
|---|---|---|
| PubCaseFinder API 서버 장애 | 중 | 로컬 폴백 동작 중 (유전자 정보 누락) |
| Faithfulness 0.62 (임상 시나리오 평균) | 중 | 감별진단 Top-3 모두 PubMed 검색 확장 시 해결 |
| MIMIC 실환자 폐질환 환자 선택 필요 | 낮 | ICD-10 필터링으로 향후 보완 |

---

## 9. 종합

```
┌──────────────────────────────────────────────────────────────┐
│  컴포넌트 단위 (4개)                                         │
│    LIRICAL ✅  PubMed ✅  PMID체크 ✅  PubCaseFinder ⚠️     │
├──────────────────────────────────────────────────────────────┤
│  파이프라인 통합 (2회)                                       │
│    MIMIC 실환자 ✅  내장 샘플 ✅                             │
├──────────────────────────────────────────────────────────────┤
│  알고리즘 정확도 (4293개 전수)                               │
│    Recall@1 = 81.6%  Recall@10 = 98.3%  MRR = 0.88         │
├──────────────────────────────────────────────────────────────┤
│  임상 시나리오 (5개)                                         │
│    Recall@1 = 4/5 (80%)  Recall@3 = 5/5 (100%)             │
├──────────────────────────────────────────────────────────────┤
│  소견서 품질                                                 │
│    PMID 유효율 100%  환각 없음                               │
│    Faithfulness 0.62  Answer Relevancy 0.86                 │
└──────────────────────────────────────────────────────────────┘
```

**강점**
- LIRICAL 알고리즘 신뢰 가능 — 4293개 전수 Recall@10 = 98.3%
- 임상 5개 케이스 중 4개 1위, 5개 전부 3위 안
- 소견서 PMID 환각 없음 — 의료 AI 핵심 요건 충족
- 파이프라인 안정성 — 5단계 전체 오류 없이 완주, Safety Layer 정상 동작

**한계**
- PubCaseFinder API 불가 — 유전자 정보 누락 (API 복구 시 즉시 개선)
- Faithfulness 0.62 — 1위 외 감별진단 설명에서 컨텍스트 이탈 발생
- 실환자 데이터 범위 — 희귀 폐질환 환자 데이터 확보 후 추가 검증 필요
