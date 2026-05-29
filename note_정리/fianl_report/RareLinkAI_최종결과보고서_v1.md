# Rare-Link AI

## 흉부 X-ray · 혈액검사 · 임상소견 멀티모달 통합 기반 희귀 폐질환 임상 의사결정 지원 시스템

### 최종 결과보고서 (Final Project Report)

---

| 항목 | 내용 |
|---|---|
| 프로젝트명 | AI 기반 폐 관련 질환 통합 진단 보조 시스템 (시스템명: RareLinkAI) |
| 소속 | AWS 바이오헬스케어 부트캠프 · 성균관대학교(SKKU) AWS SAY 2기 2팀 |
| 수행 기간 | 2026. 04. 17 ~ 2026. 05. 28 (Final Project) |
| 보고서 작성일 | 2026. 05. 22 |
| 배포 환경 | AWS `ap-northeast-2` (서울 리전) · CloudFront `https://d300v14l8u0wx7.cloudfront.net` |
| 코드/데이터 저장소 | Amazon S3 `say2-2team-bucket` |
| 팀 구성 | 박성수(PM·기획·아키텍처) · 허태웅(RAG·CloudFront·아키텍처) · 권미라(RAG·희귀질환 DB·라이선스) · 양희인(Phase 3·4·아키텍처) · 배기태(Phase 1·2·Symptom LLM) |

---

## 국문 초록 (Abstract)

희귀질환 환자는 정확한 진단까지 평균 5~7년이 소요되며, 그 과정에서 평균 4.8개 이상의 의료기관을 전전하는 이른바 '진단 방랑(Diagnostic Odyssey)'을 겪는다. 동시에 CT·MRI 같은 고가 정밀장비가 없는 1차 의료기관에서는 흉부 X-ray 단독 판독의 약 37.8%가 불확실 구간에 놓여, 전원(refer) 결정이 의사 개인의 경험에 크게 의존한다. 본 프로젝트는 이 두 가지 임상 공백을 동시에 겨냥하여, 흉부 X-ray 영상 · 혈액검사 수치 · 임상소견 텍스트의 3축 멀티모달 데이터를 단일 파이프라인으로 통합하는 **근거 기반 AI 임상 의사결정 지원 시스템(Clinical Decision Support System, CDSS)** 인 **Rare-Link AI**를 설계·구현하였다.

시스템은 흉부 X-ray를 14개 CheXpert 소견으로 분류하는 자체 DenseNet-121 기반 모델 **SooNet**, 임상소견을 표준 표현형 온톨로지(Human Phenotype Ontology, HPO)로 변환하는 LLM 에이전트, 혈액검사 89항목의 이상치를 임상 점수로 환산하는 Lab 분석기, 그리고 LIRICAL 우도비(Likelihood Ratio) 알고리즘으로 희귀질환을 순위화하는 추론 엔진, 마지막으로 Orphanet·PubMed·Monarch를 실시간 인용하는 검색증강생성(Retrieval-Augmented Generation, RAG) 보고서 생성기로 구성된다. 전체 흐름은 AWS Step Functions가 Phase 1~5 + RAG의 6개 Lambda 함수를 오케스트레이션하며, 모든 산출물은 Aurora PostgreSQL의 4계층 데이터베이스에 적재된다.

SooNet 모델은 CheXpert 검증셋에서 평균 AUROC 0.8094, MIMIC-CXR 교차도메인에서 0.7384를 달성했으며, Pleural Effusion(0.9179)·Edema(0.9501) 등 임상적으로 긴급한 소견에서 특히 높은 성능을 보였다. CheXpert의 불확실 라벨(uncertain label) 문제에 대해서는 U-Ones / U-Zeros / U-Ignore 정책을 병변별로 차등 적용하는 per-class 정책을 실험적으로 도출하였다. 지식베이스는 일반·기타 폐질환 105개와 희귀 폐질환 322개를 포괄하며, 희귀질환 측은 5,135건의 HPO 표현형 매핑을 보유한다.

본 보고서는 시스템의 문제 정의, 이론적 배경, 아키텍처 설계, Phase별 구현, CheXpert 14 라벨 의학 참고 체계, 불확실 라벨 정책 실험 결과, 지식베이스 구축, 규제 정합성, 적용 시나리오, 한계와 향후 과제를 박사 논문 수준의 깊이로 종합 정리한다.

**핵심어**: 희귀 폐질환, 임상 의사결정 지원 시스템(CDSS), 멀티모달 AI, CheXpert, DenseNet-121, Human Phenotype Ontology(HPO), LIRICAL Likelihood Ratio, Retrieval-Augmented Generation(RAG), AWS Bedrock, 진단 방랑

---

## English Abstract

Patients with rare diseases endure an average diagnostic delay of 5–7 years, consulting more than 4.8 institutions during this "diagnostic odyssey." In parallel, at primary-care facilities lacking CT or MRI, roughly 37.8% of chest X-ray readings fall into an uncertain zone where referral decisions depend heavily on individual clinician experience. **Rare-Link AI** is an evidence-based clinical decision support system (CDSS) that integrates three input modalities — chest radiographs, blood-test values, and free-text clinical notes — into a single diagnostic pipeline targeting both gaps simultaneously.

The system comprises **SooNet**, an in-house DenseNet-121 chest X-ray classifier producing 14 CheXpert observations; an LLM agent mapping clinical notes to the Human Phenotype Ontology (HPO); a laboratory analyzer scoring abnormalities across 89 blood-test items; a LIRICAL likelihood-ratio engine ranking rare diseases; and a retrieval-augmented generation (RAG) report generator citing Orphanet, PubMed, and Monarch in real time. AWS Step Functions orchestrates six Lambda functions (Phase 1–5 + RAG), persisting all artifacts into a four-layer Aurora PostgreSQL database.

SooNet achieved a mean AUROC of 0.8094 on the CheXpert validation set and 0.7384 under MIMIC-CXR cross-domain evaluation, with especially strong performance on clinically urgent findings such as pleural effusion (0.9179) and edema (0.9501). A per-class uncertain-label policy (U-Ones / U-Zeros / U-Ignore) was derived experimentally. The knowledge base covers 105 common and other lung-related diseases and 322 rare pulmonary diseases with 5,135 HPO phenotype mappings. This report documents the problem definition, theoretical background, architecture, phase-by-phase implementation, the CheXpert 14-label medical reference, the uncertain-label policy experiments, knowledge-base construction, regulatory alignment, application scenarios, and future work at the depth of a doctoral dissertation.

---

## 목차

1. 서론
2. 관련 연구 및 이론적 배경
3. 시스템 설계
4. Phase별 구현 상세
5. CheXpert 14 라벨 의학 참고 체계
6. 실험 및 결과
7. 지식베이스 구축
8. 규제 정합성·윤리·데이터 거버넌스
9. 적용 시나리오 및 사업화 전략
10. 한계 및 향후 과제
11. 결론
- 참고문헌
- 부록 A. AWS 리소스 인벤토리
- 부록 B. 데이터·모델 파일 버전 관리 대장
- 부록 C. CheXpert 14 라벨–희귀질환 연결 매핑표

---

# 1. 서론

## 1.1 연구 배경

폐 질환은 전 세계 사망 원인의 상위를 점유하는 질환군이며, 흉부 X-ray는 가장 보편적이고 저렴한 1차 영상검사 수단이다. 그러나 흉부 X-ray는 2차원 투영 영상이라는 본질적 한계로 인해, 폐 실질의 미세한 변화나 초기 병변을 단독으로 확정하기 어렵다. 더욱이 폐를 침범하는 질환의 스펙트럼은 지역사회획득폐렴 같은 흔한 질환에서부터, 림프관평활근종증(LAM)·폐포단백증(PAP)·폐정맥폐색증(PVOD)과 같이 전 세계 환자 수가 극히 적은 희귀질환에 이르기까지 매우 넓다.

문제의 핵심은 '흔한 질환의 불확실 판독'과 '희귀질환의 진단 지연'이 서로 다른 임상 현장에서, 서로 다른 양상으로 발생한다는 점이다. 전자는 1차 의료기관에서 전원 결정의 객관적 기준 부재로 나타나고, 후자는 상급 전문센터에서조차 의사 한 명이 1만 종이 넘는 희귀질환의 표현형을 모두 숙지할 수 없다는 인지적 한계로 나타난다. 본 프로젝트는 이 두 공백을 하나의 멀티모달 AI 파이프라인으로 동시에 메우는 것을 목표로 한다.

## 1.2 문제 정의

### 1.2.1 1차 의료기관의 진단 공백

CT·PET·MRI 등 고가 정밀장비를 갖추지 못한 지방 의원과 도서·산간 보건소에서는 흉부 X-ray와 혈액검사만으로 환자 상태를 판단해야 한다. 이때 환자를 상급 병원으로 전원할 것인지를 결정하는 객관적 기준이 없어, 경험이 부족한 의사일수록 판단 편차가 커진다. CheXpert 데이터셋 72,202건의 분석에 따르면 흉부 X-ray 단독 판독의 **약 37.8%가 불확실 구간**에 해당한다. 2024년 정부가 권역책임의료기관 AI 투자 120억 원을 발표하면서, 지역 의료 격차 해소는 국가적 과제로 격상되었다.

### 1.2.2 희귀질환의 진단 방랑(Diagnostic Odyssey)

Orphanet 통계에 따르면 희귀질환 환자는 정확한 진단까지 평균 5~7년이 걸리며, 그 과정에서 평균 4.8개 이상의 병원을 거친다. 잘못된 진단으로 누적되는 의료비는 환자 1인당 평균 3,000~5,000만 원에 달한다(NORD, 2022). 국내 희귀질환 산정특례 등록자는 2023년 기준 약 55만 명이며 매년 증가 추세다. Orphanet에 등재된 희귀질환은 11,456종에 이르러, 개별 의사가 이를 모두 숙지하는 것은 원천적으로 불가능하다.

### 1.2.3 의료 AI 시장의 가속

유럽 방사선사의 AI 도구 사용률은 2018년 20%에서 2024년 48%로 급증하였다. 국내에서는 흉부 X선 AI-CAD가 2024년 건강보험에 임시등재되어 수가 보상이 시작되었으며, 미국 ARPA-H의 RAPID 프로그램은 희귀질환 AI 진단을 국가 사업으로 2026년 착수하였다. 글로벌 의료 AI 영상 분석 시장은 2023년 20.9억 달러에서 2030년 81.5억 달러(CAGR 21.7%)로, 국내 의료 AI 시장은 2023년 3,680억 원에서 2028년 1조 2,000억 원(CAGR 26.7%)으로 성장이 전망된다.

## 1.3 연구 목적 및 기여

본 프로젝트의 목적은 흉부 X-ray·혈액검사·임상소견의 3축 입력을 통합하여, 일반 폐질환과 희귀 폐질환을 단일 파이프라인에서 감별하고 의학적 근거와 함께 진단 보조 리포트를 자동 생성하는 CDSS를 구현하는 것이다. 주요 기여는 다음과 같다.

첫째, 영상·검체·텍스트라는 이종(heterogeneous) 모달리티를 표준 표현형 온톨로지(HPO)와 우도비(LR) 통계 위에서 정합적으로 결합하는 아키텍처를 제시하였다. 둘째, CheXpert 14 라벨 분류기(SooNet)의 불확실 라벨 처리 정책을 병변별로 차등 적용하는 per-class 정책을 실험적으로 도출하였다. 셋째, 일반·기타 폐질환 105개와 희귀 폐질환 322개를 포괄하는 가중치 기반 지식베이스를 구축하였다. 넷째, LLM의 환각(hallucination)을 억제하기 위해 Orphanet·PubMed 등 1차 출처를 실시간 인용하는 RAG 파이프라인을 구현하였다. 다섯째, 전 과정을 AWS 서버리스 아키텍처(Lambda·Step Functions·Aurora·Bedrock·CloudFront) 위에 배포하여 재현 가능한 클라우드 시스템으로 완성하였다.

## 1.4 기존 솔루션 대비 차별성

VUNO Med-CXR, 루닛 INSIGHT, Aidoc, Siemens AI-Rad 등 상용 AI 방사선 솔루션은 흉부 X-ray의 소견 분류에 머무른다. 이들은 혈액검사 이상치 결합, 희귀질환 스크리닝, HPO 온톨로지 연동, RAG 기반 치료 근거 제시 기능을 제공하지 않는다.

| 기능 | VUNO Med-CXR | 루닛 INSIGHT | Aidoc | Siemens AI-Rad | **RareLinkAI** |
|---|---|---|---|---|---|
| X-ray 소견 분류 | 약 10개 | 약 10개 | 일부 | 다수 | **14개 (SooNet)** |
| 혈액검사 연동 | 미지원 | 미지원 | 미지원 | 미지원 | **89항목 + 임상점수** |
| 희귀질환 스크리닝 | 미지원 | 미지원 | 미지원 | 미지원 | **322개 HPO 매칭** |
| HPO 온톨로지 연동 | 미지원 | 미지원 | 미지원 | 미지원 | **5,135건 매핑** |
| 유전자 검사 권고 | 미지원 | 미지원 | 미지원 | 미지원 | **WES/WGS 선별** |
| RAG 기반 치료 근거 | 미지원 | 미지원 | 미지원 | 미지원 | **구현 완료** |
| 한국어 임상 리포트 | 지원 | 미지원 | 미지원 | 미지원 | **다중 섹션 자동 생성** |
| 오픈 데이터 기반 | 미지원 | 미지원 | 미지원 | 미지원 | **Orphadata·MIMIC-IV** |

X-ray·혈액검사·HPO 온톨로지·희귀질환 스크리닝·RAG 치료 근거를 하나의 파이프라인으로 통합한 서비스는 조사 시점 기준 국내외에 상용 사례가 확인되지 않았으며, 이는 본 프로젝트가 겨냥한 시장 공백(market gap)이다.

---

# 2. 관련 연구 및 이론적 배경

## 2.1 흉부 X-ray 딥러닝의 계보

흉부 X-ray 자동 판독 연구는 Rajpurkar 등이 2017년 발표한 **CheXNet**에서 본격화되었다. CheXNet은 121층 DenseNet을 ChestX-ray14 데이터셋으로 학습하여 폐렴 검출에서 방사선 전문의에 준하는 성능을 보고하였다. 이후 Stanford ML Group이 2019년 공개한 **CheXpert**(Irvin 등, AAAI 2019)는 224,316장 규모의 대형 흉부 X-ray 데이터셋으로, 방사선 판독문에서 자동 라벨러로 추출한 14개 관찰 소견과 그에 수반되는 '불확실(uncertain)' 라벨을 함께 제공한다는 점이 특징이다. 본 프로젝트의 X-ray 모델 SooNet은 이 계보 위에서 DenseNet-121을 기반 아키텍처로 채택하였다.

## 2.2 CheXpert 14 라벨 체계와 불확실 라벨 문제

CheXpert의 14개 라벨은 판독문에서의 출현 빈도와 임상적 중요도를 기준으로 선정되었으며, Fleischner Society 흉부 영상 용어집(Hansell 등, Radiology 2008)의 표준 정의를 따른다. 14 라벨은 단순한 평면적 목록이 아니라 의학적 포함 관계(parent-child hierarchy)를 가진다. Lung Opacity는 Atelectasis·Consolidation·Pneumonia·Edema·Lung Lesion을 포괄하는 상위 우산(umbrella) 개념이며, Pneumonia는 Consolidation의 하위, Cardiomegaly는 Enlarged Cardiomediastinum의 하위에 위치한다.

CheXpert의 핵심적 난점은 자동 라벨러가 판독문에서 확신할 수 없는 소견에 부여하는 **불확실 라벨(-1)**의 처리 방법이다. 원 논문은 이를 무시(U-Ignore), 음성 취급(U-Zeros), 양성 취급(U-Ones), 자가학습(U-SelfTrained), 다중클래스(U-MultiClass)의 다섯 가지 전략으로 비교하였다. 본 프로젝트는 이 정책 선택이 병변별로 달라야 한다는 가설을 세우고 실험적으로 검증하였다(6.3절).

## 2.3 Human Phenotype Ontology(HPO)

HPO는 인간 질병의 표현형 이상(phenotypic abnormality)을 표준화된 코드 체계로 기술하는 온톨로지다. 본 프로젝트는 HPO를 모달리티 간 공통 언어(common interlingua)로 사용한다. 즉, 임상소견 텍스트(Phase 1)와 X-ray 소견(Phase 2)을 각각 HPO 코드로 정규화함으로써, 본질적으로 이종인 데이터를 동일한 표현형 공간에서 비교·결합할 수 있게 한다. 전체 HPO 온톨로지(`hpo_official.json`, 약 22 MB)는 11,514개 텀과 282,728건의 어노테이션을 포함한다.

HPO 활용에서 특히 중요한 개념은 **음성 표현형(Negative HPO)**이다. 환자에게 어떤 증상이 '있다'는 정보뿐 아니라 '없다'고 확인된 정보 역시 감별진단의 강력한 근거가 된다. 어떤 질환의 전형 증상이 환자에게 부재하면 해당 질환의 신뢰도를 하향해야 하기 때문이다.

## 2.4 LIRICAL 우도비(Likelihood Ratio) 기반 진단

LIRICAL(Likelihood Ratio Interpretation of Clinical Abnormalities, Robinson 등, Am J Hum Genet 2020, PMID:32755546)은 환자의 표현형 조합이 특정 희귀질환에서 관찰될 가능성을 우도비로 계산하는 알고리즘이다. 각 HPO 증상에 대해 '해당 질환에서 그 증상이 나타날 빈도'를 '일반 인구(전체 질환)에서 그 증상이 나타나는 배경 빈도'로 나눈 값이 우도비이며, 모든 증상의 우도비를 곱(로그 공간에서는 합)하여 질환 점수를 산출한다. LIRICAL의 가장 큰 장점은 **설명 가능성(explainability)**이다. 어떤 질환이 상위에 오른 이유를 매칭된 HPO와 각 우도비 항으로 직접 추적할 수 있어, 의사가 AI 결과를 검토·납득할 수 있다. 본 프로젝트의 Phase 5는 이 알고리즘을 자체 구현하였다(4.6절).

## 2.5 RAG 기반 의학 LLM

대형 언어모델(LLM)을 의학 진단에 단독 사용할 때는 세 가지 치명적 한계가 있다. 첫째 **최신성** — 학습 컷오프 이후의 치료 가이드라인을 반영하지 못한다. 둘째 **희귀질환 지식 공백** — 학습 데이터에 희귀질환 텍스트가 극히 적어 환각 위험이 높다. 셋째 **근거 투명성** — 의사는 "AI가 그렇게 말했다"를 처방 근거로 삼을 수 없으며 반드시 출처 인용이 필요하다.

검색증강생성(RAG)은 LLM이 답을 생성하기 전에 신뢰할 수 있는 외부 지식베이스에서 관련 문서를 검색하여 프롬프트에 주입함으로써 이 세 한계를 동시에 완화한다. 본 프로젝트의 RAG는 Orphanet·PubMed Case Reports·진료 가이드라인을 3계층(Tier)으로 구성하고, PubCaseFinder·Monarch·ClinicalTrials.gov를 병렬 조회하는 Hybrid Dual RAG 구조를 채택하였다(4.7절). 관련 선행 연구로는 RDguru(PMC, 2025), DeepRare(2026), PhenoBrain(Nature npj, 2025; EHR HPO 추출 AI의 Top-10 recall 0.813) 등이 있다.

---

# 3. 시스템 설계

## 3.1 전체 아키텍처 개요

RareLinkAI는 환자 데이터를 입력받아 희귀 폐질환 후보를 자동 도출하고, 의학 근거 기반 진단 보조 리포트를 생성하는 멀티모달 AI 파이프라인이다. 시스템은 기능적으로 5개의 에이전트로, 실행 단위로는 Phase 0~5 + RAG의 7개 단계로 기술할 수 있다.

### 3.1.1 5 에이전트 구조

| 에이전트 | 역할 | 입력 | 출력 | 핵심 기술 |
|---|---|---|---|---|
| Agent A (Vision) | X-ray 소견 분류 | 흉부 X-ray | 14소견 확률 + Grad-CAM 히트맵 | SooNet (DenseNet-121, 448px) |
| Agent B (NLP) | 임상소견서 → HPO | 의사 소견서 텍스트 | HPO 코드 목록 | AWS Bedrock Claude |
| Agent C (Lab) | 혈액검사 이상치 분석 | Lab JSON / PDF | 이상치 + 임상점수(NEWS2 등) | 89항목 YAML + PDF 파서 |
| Agent D (Reporter) | 임상 리포트 생성 | 전체 HPO + Phase 결과 | 한국어 소견서 + 논문 인용 | Bedrock + RAG 컨텍스트 |
| Agent E (Inference) | 희귀질환 HPO 매칭 | HPO 프로파일 | Top-N 희귀질환 + 유전자 권고 | Orphanet IC 가중 매칭 |

### 3.1.2 실행 파이프라인 (Phase 0 ~ RAG)

```
환자 입력 (EMR FHIR Bundle)
   │
   ▼
Phase 0  데이터 수집·정규화 ── FHIR → 4개 canonical 테이블
   │
   ▼
Phase 1  임상소견 → HPO 변환 ────────┐
Phase 2  X-ray 분석 (SooNet)  ───────┤ 병렬
   │                                  │
   ▼                                  │
Phase 3  Lab·미생물 멀티모달 스코어링 ◄┘
   │
   ▼
Phase 4  LLM 검증·리랭킹 ────────────┐
Phase 5  LIRICAL LR 희귀질환 랭킹 ───┤ 병렬
   │                                  │
   ▼                                  │
RAG  Hybrid Dual RAG 최종 보고서 생성 ◄┘
   │
   ▼
CloudFront 프론트엔드 결과 화면
```

전체 흐름은 AWS Step Functions가 오케스트레이션하며, Phase 1·2는 병렬, Phase 4·5도 병렬로 실행된다. Phase 4(일반 질환 리랭킹)의 최고 점수가 0.5 미만이면 — 즉 흔한 질환으로 환자 상태가 설명되지 않으면 — Phase 5(희귀질환 LIRICAL 스코어링)가 자동으로 발동하는 '희귀질환 탐정 모드' 설계가 본 시스템의 임상적 핵심 동작이다.

## 3.2 데이터 입력 3축과 우선순위

시스템은 세 가지 원시 입력(Raw Input)을 받는다. (A1) 의사가 기록한 임상소견 원문, (A2) 흉부 X-ray, (A3) 혈액·폐기능 검사 수치다. 이들은 가공 단계를 거치며 다음과 같은 근거 우선순위로 재배열된다.

| 순위 | 데이터 | 성격 |
|---|---|---|
| 1순위 | RAG Context | 인용 가능한 유일한 근거 데이터 |
| 2순위 | LIRICAL Ranking | 통계 기반 질환 랭킹 |
| 3순위 | HPO 변환 결과 | 표준 표현형 코드 |
| 4순위 | Raw Input | 원시 환자 데이터 |

이 우선순위는 LLM이 최종 리포트를 작성할 때 충돌을 해소하는 규칙으로도 작동한다. 예컨대 통계 랭킹과 RAG 근거가 충돌하면 RAG를 우선하고, 근거 간 충돌은 Evidence Level이 높은 쪽을 우선한다.

## 3.3 S/L/R/M 임상 가중치 체계

본 시스템은 진단 점수를 4개 축으로 분해한다. **S**(Symptom, 증상)·**L**(Lab, 혈액검사)·**R**(Radiology, 영상)·**M**(Microbiology, 미생물)이며, 네 가중치의 합은 항상 1.0이다. 가중치는 질환마다 다르게 설정되는데, 이는 "어떤 질환은 X-ray가 결정적이고 어떤 질환은 폐활량 검사가 결정적"이라는 임상 현실을 수치화한 것이다.

```
최종 점수 = (S × 증상 매칭률) + (L × 혈액 매칭률)
          + (R × 영상 매칭률) + (M × 미생물 매칭률)
```

| 점수 구간 | 판정 | 임상적 의미 |
|---|---|---|
| 0.70 이상 | STRONG | 확진 검사 권고 |
| 0.40 ~ 0.70 | MODERATE | 추가 검사 필요 |
| 0.40 미만 | WEAK | 가능성 낮음 — 희귀질환 경로 검토 |

질환별 기본 가중치는 표준 가이드라인을 근거로 한다. 예를 들어 ARDS는 영상이 최우선이므로 R=0.45, COPD(J44 계열)는 GOLD 2026 보고서가 "흉부 X-ray가 진단에 비유용"하다고 명시함에 따라 R=0.10으로 낮추고 폐활량 검사(L=0.40)를 높였다. 일반·희귀 질환 전반의 기본값은 일반질환 S0.25·L0.20·R0.35·M0.20(폐질환에서 영상이 핵심), 희귀질환 S0.45·L0.20·R0.20·M0.15(HPO 표현형 매칭이 1차 수단)로 설정된다.

| 질환 | S | L | R | M | 설정 근거 |
|---|---|---|---|---|---|
| 지역사회획득폐렴(CAP) | 0.20 | 0.25 | 0.35 | 0.20 | X-ray 확진 → R 높음 |
| 결핵 | 0.20 | 0.15 | 0.30 | 0.35 | 균 배양 확진 → M 높음 |
| COPD (J44) | 0.30 | 0.40 | 0.10 | 0.20 | GOLD 2026: CXR 비유용 |
| 폐기종(J43) | 0.45 | 0.40 | 0.10 | 0.05 | 폐활량 기반 → L·S 높음 |
| 만성기관지염(J41) | 0.45 | 0.30 | 0.10 | 0.15 | 임상 증상 우위 → S 최고 |
| ARDS | 0.15 | 0.20 | 0.45 | 0.20 | 영상이 최우선 → R 최고 |

모든 가중치는 단일 진실 원천(Single Source of Truth, SSOT) 파일인 `lung_disease_profiles_v3_7.yaml`(2.3 MB, 105개 질환)에 집약되어, 이 파일을 수정하면 시스템 전체에 자동 반영된다.

## 3.4 AWS 클라우드 아키텍처

시스템은 AWS 서울 리전(`ap-northeast-2`, 계정 666803869796)에 서버리스 중심으로 배포되었다.

```
React Frontend ──HTTPS──► FastAPI (api/app)
                              │ boto3.start_execution
                              ▼
                    AWS Step Functions
                  (rare-link-pipeline-dev)
                              │ Invoke
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
       Lambda × 6      Aurora PostgreSQL   AWS Bedrock
   (Phase 1~5 + RAG)   (rarelinkai 스키마)  (Claude 계열)
              │
              ▼
        CloudWatch Logs / Alarms / X-Ray Tracing
```

| 구성 요소 | 사양·식별자 |
|---|---|
| 컴퓨팅 | AWS Lambda 6종 (Python 3.11), Phase 3는 메모리 2,048 MB·Timeout 300초 |
| 오케스트레이션 | AWS Step Functions State Machine `rare-link-pipeline-dev` (ASL v4) |
| 데이터베이스 | Aurora PostgreSQL 16.4, 클러스터 `patient-db-cluster`, 스키마 `rarelinkai` |
| 생성형 AI | AWS Bedrock — Claude 3.5 Sonnet(RAG), Claude Haiku(Phase 1 HPO 추출), Claude Sonnet(Phase 4 검증) |
| 네트워크 | VPC `vpc-06dd0ad1f2335ea74`, private subnet 2개(multi-AZ), NAT Gateway, S3·SageMaker·Bedrock VPC Endpoint |
| 정적 호스팅·CDN | Amazon S3 `say2-2team-bucket` + CloudFront `d300v14l8u0wx7.cloudfront.net` |
| 비밀 관리 | AWS Secrets Manager (`rare-link-ai/aurora/app-user` 등) |
| API 게이트웨이 | Amazon API Gateway (Phase별 REST 엔드포인트), Throttling Burst 100 / Rate 50 |
| 보안·관측 | GuardDuty 위협 탐지, CloudWatch 알람, AWS X-Ray 분산 트레이싱 |
| 배포 도구 | AWS SAM (CloudFormation 기반 IaC), 스택 6개 |

Lambda를 VPC 내부에 배치한 이유는 Aurora가 사설 서브넷에 위치하여 RDS 5432 포트 접근이 보안그룹(`sg-03b9bc5d95699b797`, RDS ingress가 허용된 유일한 SG)으로 제한되기 때문이다. 외부 Secrets Manager 호출을 위해 NAT Gateway를 경유시키며, 멀티 AZ 서브넷 구성으로 ENI 부족 시에도 가용성을 확보한다.

## 3.5 Aurora 4-Layer 데이터베이스

데이터베이스는 원본 추적성(provenance)과 감사(audit) 가능성을 위해 4개 계층으로 설계되었다.

| Layer | 이름 | 대표 테이블 | 쓰기 정책 |
|---|---|---|---|
| Layer 0 | Raw EMR | raw_emr_bundle, fhir_bundle_archive | 불변(Immutable) — INSERT만, UPDATE/DELETE는 트리거로 차단 |
| Layer 1 | Canonical | patient_profile, clinical_note, lab_result, imaging_study, cxr_image_registry | Append-only, bundle_id FK로 Layer 0 추적 |
| Layer 2 | Phase IO | diagnosis_session, phase1~5 결과, final_report, rag_api_cache | Append-only, (session_id, phase, executed_at) 키 |
| Layer 3 | Outcome | physician_feedback, final_clinical_outcome | Append-only |

이와 별도로, 모든 Phase의 실행 이력을 단일 테이블에 통합 기록하는 운영 로그 `phase_execution_log`가 존재한다. 이 테이블은 log_id·session_id·phase_name·status·duration_ms·error_code·error_stacktrace·model_versions 등을 담아, 장애 분석과 성능 모니터링을 위한 단일 SQL 진입점을 제공한다. Layer 0를 불변으로 설계한 것은 의료 데이터의 법적 증거능력과 재현성을 보장하기 위한 핵심 결정이다.

---

# 4. Phase별 구현 상세

## 4.1 Phase 0 — 데이터 수집(Ingest)

Phase 0는 MIMIC-IV 데이터셋 및 외부 API에서 환자 데이터를 수집하여 데이터베이스에 적재한다. FHIR Bundle을 patient_profile·clinical_note·lab_result·imaging_study 4개 canonical 테이블로 정규화하며, 모든 Layer 1 레코드에 bundle_id를 부여하여 Layer 0 원본으로의 추적성을 유지한다. 국내 온프레미스 환경 적용 시에는 EMR 연동 방식을 별도 검토하도록 설계 단계에서 명시하였다.

## 4.2 Phase 1 — 임상소견 → HPO 변환

Phase 1은 한국어로 작성된 임상소견 텍스트를 HPO 표준 코드로 변환한다. 환자가 "기침하고 열이 나요"라고 기술하면 LLM이 이를 표준 표현형 코드로 자동 매핑하는 방식이다.

- **Lambda**: `phase1-symptom-dev` (Python 3.11)
- **핵심 코드**: `Phase_1/symptom_llm_4.py`의 `BedrockHPOExtractor` 클래스
- **온톨로지**: `hpo_official.json`(HPO OBO Graph JSON)을 로드하여 11,514개 HPO 텀에 매핑
- **추출 전략**: Discovery → Reference → Extraction의 2단계 방식
- **출력**: `phase1_hpo_extraction` 테이블의 `positive_hpo`·`negative_hpo` (JSONB)

Phase 1의 산출물은 두 종류로 구분된다. **Positive HPO**는 환자에게 있는 증상으로 진단을 지지하는 근거이며, **Negative HPO**는 환자에게 없다고 확인된 증상으로 감별진단의 핵심 근거가 된다. Negative HPO가 어떤 질환의 전형 증상을 부정하면, 그 질환의 신뢰도가 하향 적용된다.

## 4.3 Phase 2 — X-ray 분석 (SooNet)

Phase 2는 흉부 X-ray 영상을 입력받아 14개 CheXpert 소견의 확률값을 산출한다.

- **Lambda**: `say2-2team-phase2-vision`
- **모델 가중치**: `anatomy_soonet_v5_best.pth` (43.4 MB, 현재 배포 Best 모델)
- **보조 모델**: `unet_lung_heart_ep5.pth` (57.9 MB) — 폐·심장 영역 분할
- **핵심 코드**: `Phase_2/soo_net_5.py`의 `AnatomySooNetV5` 클래스
- **전처리**: `anatomy_preprocessor.py` — 448px 입력 정규화, CLAHE 대비 보정

### 4.3.1 SooNet 아키텍처

SooNet(AnatomySooNetV5)은 DenseNet-121을 기반 아키텍처로 채택한다. DenseNet-121은 각 레이어가 이전 모든 레이어와 직접 연결되는 Dense Connection 구조로, 파라미터 효율성이 높고 기울기 소실(gradient vanishing) 문제에 강하여 의료영상 분석에서 검증된 백본이다. SooNet v5는 여기에 세 가지 의학적 사전지식 모듈을 추가하였다.

첫째, **xrv 사전학습 백본** — 대규모 흉부 X-ray로 사전학습된 가중치로 초기화하여 의료 도메인 적응을 가속한다. 둘째, **Anatomy Soft Attention** — UNet이 분할한 폐·심장 해부 영역 마스크를 부드러운 주의(soft attention) 신호로 활용하여, 모델이 해부학적으로 유의미한 영역에 집중하도록 유도한다. 셋째, **A³(Anatomy-Aware Aggregation)** — 해부 영역별 특징을 집계하여 최종 14 라벨 로짓을 산출한다.

### 4.3.2 출력 활용과 설계 변경

모델 출력은 14개 CheXpert 라벨과 확률값(내림차순 정렬)이며, 동시에 Grad-CAM 히트맵을 생성하여 모델이 어느 영역을 근거로 판단했는지 시각적으로 제시한다. 

설계상 중요한 결정으로, 2026년 5월 15일 팀은 **Phase 2의 CheXpert 라벨을 HPO로 직접 변환하던 방식을 폐기**하였다. CheXpert 라벨을 HPO로 변환하는 것은 의학적 타당성이 부족하다고 판단하였기 때문이다. 대신 Phase 3·4 스코어링 단계에서 CheXpert 라벨과 확률값을 그대로 활용하는 방향으로 수정하였다. 이는 영상 소견의 의미를 인위적으로 변환하지 않고 원본 신호로 보존하려는 결정이다.

## 4.4 Phase 3 — Lab 멀티모달 스코어링

Phase 3는 Phase 1의 HPO, Phase 2의 X-ray 결과, 그리고 혈액검사 원시 수치를 통합하여 질환 레지스트리와 매칭하고 순위를 산출한다.

- **Lambda**: `phase3-scorer-dev` (메모리 2,048 MB, Timeout 300초)
- **기준 파일**: `lab_reference_ranges_v9_5.yaml` (345.9 KB) — 혈액검사 정상 기준치
- **출력**: `phase3_integrated_ranking` 테이블의 `unified_positive_hpo`·`ranking` (JSONB)

혈액검사 수치는 정상 / 이상 / 위급의 3단계로 판별되며, 위급 판정 시 진단 점수에 +0.05의 보너스가 부여된다. Phase 3는 NEWS2·qSOFA·CURB-65·PESI 등 4종의 표준 임상 스코어링 시스템과 37개 활력징후(VRH) 파라미터를 포함하여, 단순 이상치 탐지를 넘어 환자 중증도를 정량화한다. 또한 실물 혈액검사 결과지 PDF를 자동 파싱하여 89항목으로 매핑하는 Lab 파서를 갖추었다.

Phase 3의 계산 효율을 위해 **2단계 필터링(two-stage filtering)**을 적용한다. 먼저 HPO 교집합으로 전체 질환 후보를 약 30개로 축소한 뒤, 이 후보군에 대해서만 전체 우도비 계산을 수행하여 최종 top-K를 산출한다. 이 전략으로 계산 비용을 약 17배 절감하였다.

## 4.5 Phase 4 — LLM 검증·리랭킹

Phase 4는 Phase 3의 순위 결과를 AWS Bedrock의 LLM으로 검증하고 재순위화하며, 진단 보조 리포트의 초안을 생성한다.

- **Lambda**: `phase4-verifier-dev`
- **모델**: AWS Bedrock Claude 계열 (리전 `ap-northeast-2` 고정)
- **출력**: `phase4_llm_rerank` 테이블

### 4.5.1 Evidence Level과 Confidence 체계

Phase 4는 각 근거를 출처 신뢰도에 따라 4단계 Evidence Level로 분류한다.

| 레벨 | 근거 유형 |
|---|---|
| Level 1 | PubMed 무작위대조시험(RCT) |
| Level 2 | PubMed 관찰 연구 |
| Level 3 | 증례 보고(Case report) |
| Level 4 | Orphanet 등재 정보 |

이를 바탕으로 각 후보 질환의 최종 신뢰도(Confidence)를 결정한다. **HIGH**는 근거 2개 이상이며 Level 1 또는 2를 포함할 때, **MEDIUM**은 근거 1개 이상일 때, **LOW**는 근거가 부족하거나 충돌이 존재할 때 부여된다.

### 4.5.2 6단계 필수 처리 순서와 금지 사항

Phase 4의 LLM은 다음 6단계를 반드시 순서대로 수행한다. ① HPO 기반 후보 질환 필터링 → ② Negative HPO 기반 제외 → ③ 랭킹 기반 우선순위 결정 → ④ RAG Context에서 근거 매칭 → ⑤ Lab·X-ray 임상 해석 → ⑥ 최종 Confidence 결정.

또한 환각을 원천 차단하기 위해 네 가지 금지 사항을 강제한다. Context 외 정보 사용 금지, 존재하지 않는 PMID 생성 금지, 일반 의학 지식 사용 금지, Negative HPO 무시 금지가 그것이다. 충돌 해소 규칙으로는 랭킹과 RAG가 충돌하면 RAG 우선, 근거 간 충돌은 Evidence Level이 높은 것 우선, Negative HPO 충돌 시 해당 질환 Confidence 하향을 적용한다.

## 4.6 Phase 5 — LIRICAL LR 희귀질환 랭킹

Phase 5는 Phase 4의 최고 점수가 0.5 미만일 때 병렬로 발동하여, 322개 희귀 폐질환을 LIRICAL 우도비로 순위화한다. 2026년 5월 18일부터 RAG 코드에 혼재되어 있던 LR 로직이 독립 Lambda(`phase5-lr-dev`)로 완전히 분리되었다.

- **Lambda**: `phase5-lr-dev` (메모리 1,024 MB, Timeout 300초, VPC 내부 배치)
- **알고리즘 기준**: LR_pipeline_v2 명세(권미라 v3.1), Robinson 등 2020(PMID:32755546)
- **출력**: `phase5_rare_disease_listing` 테이블

### 4.6.1 LR 계산 공식

```
단일 HPO 텀의 우도비:
  lr(HP | D)  = frequency_p(HP | D) / background_freq(HP)
  log_lr(HP)  = log10( lr )

모달리티 가중 합산:
  weighted_log_lr = Σ_m weights_D[m] × Σ_{HP∈m} log_lr(HP)

사전확률 보정 및 최종 점수:
  log_prior   = log10( prevalence_numeric_D )
  final_score = weighted_log_lr + log_prior
  lr_value    = exp( final_score )

피스팅(listing) 조건:  lr_value > 5.0
```

분자인 `frequency_p`는 해당 질환에서 그 증상이 나타나는 빈도(0~1)이며, 분모인 `background_freq`는 12,974개 전체 질환에서 그 HPO가 나타나는 평균 빈도다. HPOA의 범주형 빈도는 Obligate(1.00)·Very frequent(0.90)·Frequent(0.55)·Occasional(0.17)·Very rare(0.025)·Excluded(0.00)으로, 정보가 없으면 LIRICAL 기본값 0.50으로 매핑된다. 0 빈도로 인한 로그 발산을 막기 위해 Laplace floor 1/(2×12,974) ≈ 3.85×10⁻⁵를 적용한다.

### 4.6.2 모달리티 가중치와 lr_category

각 희귀질환은 영상(radiology)·임상소견(symptoms)·혈액(lab)·미생물(micro) 4개 모달리티에 대한 가중치 `lr_weights`와, 진단 신뢰도에 따른 분류 `lr_category`(A~G)를 가진다. 322개 질환의 카테고리 분포는 다음과 같다.

| 카테고리 | 질환 수 | 특징 |
|---|---|---|
| A | 105 | 높은 신뢰도 — HPO 증거 충분 |
| B | 72 | 중간 신뢰도 |
| G | 65 | 유전자 기반 분류 |
| C | 25 | 임상 특이 소견 중심 |
| E | 25 | 환경·외인성 원인 |
| D | 21 | 진단 난도 높음 |
| F | 9 | 극히 드문 질환 |

### 4.6.3 Negative HPO 처리와 설명 가능성

Negative HPO가 어떤 질환의 전형 증상 목록에 포함되어 있으면, 해당 텀은 LR 계산에서 완전히 제외되고 `contradicted_hpo` 리스트에 기록된다. 이는 "질환의 전형 증상이 환자에게 없다"는 사실을 RAG 보고서로 전달하여, LLM이 감별진단 서술에 활용하도록 하기 위함이다.

LIRICAL 방식의 결정적 장점은 설명 가능성이다. 출력 레코드는 `matched_hpo_phase1`·`matched_hpo_phase2`·`contradicted_hpo`·`weights_applied`·`evidence`(각 모달리티별 log_lr, weighted_log_lr, log_prior, final_score) 필드를 모두 보존하므로, 의사는 특정 희귀질환이 왜 상위에 올랐는지를 증상 단위로 추적·검증할 수 있다.

### 4.6.4 실행 사례 — 림프관평활근종증(LAM)

다음은 LAM(ORPHA:538) 환자에 대한 LR 계산 예시다.

```
증상 ① 호흡곤란 (Phase 1)
   LR = 0.90 / 0.03 = 30      → log10 = +1.48
증상 ② 흉막 삼출 (Phase 2, X-ray 확인)
   LR = 0.55 / 0.018 = 30.6   → log10 = +1.49
✗ 폐 섬유증 없음 (Negative HPO)
   LR 계산 제외 + contradicted_hpo 기록

가중 합산 = (radiology 0.4 × 1.49) + (symptoms 0.4 × 1.48)
         = 0.596 + 0.592 = +1.188
유병률 보정 log10(1/200,000) = -5.30
final_score = 1.188 - 5.30 = -4.112
lr_value = exp(-4.112) ≈ 16.3   →  5.0 초과, 후보 포함 ✅
```

이 계산을 KB의 12,974개 희귀질환 전체에 반복하고, lr_value 내림차순으로 정렬한 것이 최종 희귀질환 후보 목록이다. LR 데이터 소스로는 Orphanet `phenotype.hpoa`(35.3 MB, LR 분자), `hpo_background_freq.json`(328 KB, LR 분모), `rare_disease_profiles_v3_1.yaml`(1.1 MB, 322개 질환 임상 프로필 SSOT)을 사용한다. KB와 배경빈도 파일은 Lambda Layer(`/opt/data/`)에 번들되어 S3 다운로드 없이 즉시 로드되며, 엔진 객체를 싱글턴으로 유지하여 cold start를 최소화한다.

## 4.7 RAG — Hybrid Dual RAG 최종 보고서

RAG 단계(`phase5-rag-dev`)는 모든 Phase 결과를 종합하여 의학 근거가 인용된 최종 진단 보조 리포트를 생성한다.

### 4.7.1 3계층 지식베이스

| Tier | 데이터 소스 | 채택 이유 |
|---|---|---|
| Tier 1 (최우선) | Orphanet 질환 페이지, GARD, NORD | 증상→진단검사→치료 단계가 구조화·6개월 주기 갱신 |
| Tier 2 (핵심) | PubMed Case Reports | 환자 내원→검사→진단→치료 내러티브 포함 |
| Tier 3 (보완) | ATS·ERS·GOLD·GINA 가이드라인 | 표준 진단 알고리즘·치료 프로토콜 |

### 4.7.2 RAG 파이프라인

진단명 추출 → Tier 1 Orphanet JSON 직접 파싱 → Tier 2 PubMed API 검색(질환명 + Case Reports 필터 + 최근 2년) → 500토큰 단위 청크 분할(PMID·DOI 메타데이터 태깅) → MedCPT(PubMed 특화 BERT)로 768차원 임베딩 → 벡터 유사도 기반 Top-5 청크 추출 → Bedrock 프롬프트 주입 → PMID·DOI 클릭 가능 인용 삽입의 8단계로 동작한다. 외부 API는 PubMed E-utilities, Monarch Initiative, PubCaseFinder, ClinicalTrials.gov를 병렬 호출한다. 비용 최적화를 위해 `rag_api_cache` 테이블에 TTL 7일 캐싱을 적용한다.

최종 출력은 JSON과 Markdown 두 형식으로 생성되며, JSON에는 진단 후보(rank·disease·orpha_code·lr_score·evidence·confidence), 유전자 검사 권고, 치료 가이드라인, 최신 동향, 다음 단계, 그리고 모든 PMID가 컨텍스트 내에 존재하는지를 스스로 점검하는 `self_check` 필드가 포함된다.

## 4.8 Step Functions 오케스트레이션

전체 파이프라인은 AWS Step Functions State Machine `rare-link-pipeline-dev`(ASL v4)가 조율한다.

```
[Start] {session_id, patient_fhir_id, symptom_text, cxr_s3_key}
   │
   ▼ ParallelPhase1And2 (Phase1 ‖ Phase2)
   ▼ Phase3  (Sequential — 동시 cold start 회피)
   ▼ ParallelPhase4AndPhase5 (Phase4 ‖ Phase5)
   ▼ RAG  (Catch+Pass 우회 — 일부 Phase 실패해도 전체는 진행)
   ▼ [End]
```

설계 과정에서 Phase 4·5를 VPC 내에서 동시 cold start할 때 ENI 및 RDS 커넥션 경쟁으로 timeout이 발생하는 이슈가 관찰되었다. 이를 ASL v4에서 Phase 3를 순차 실행으로 분리하고, RAG 단계에 Catch+Pass 우회를 적용하여 일부 Phase가 실패해도 전체 실행은 SUCCEEDED로 종결되도록 보강하였다. 통합 실행 검증 결과는 6.6절에 기술한다.

---

# 5. CheXpert 14 라벨 의학 참고 체계

본 장은 SooNet이 출력하는 14개 CheXpert 라벨 각각에 대한 의학적 정의, 영상학적 특징, 그리고 RareLinkAI 파이프라인에서의 활용 방식을 정리한다. 흉부 X-ray는 X선 투과 영상이므로 공기가 찬 폐는 검게, 뼈·액체·조직은 희게 보인다는 기본 원리 위에서 모든 라벨이 해석된다. 14 라벨의 의학적 정의는 CheXpert 원 논문(Irvin 등, AAAI 2019)과 Fleischner Society 흉부 영상 용어집(Hansell 등, Radiology 2008)을 표준으로 삼는다.

## 5.1 14 라벨 개요와 계층 구조

| # | 라벨 (영문) | 한국어 | 위치 | 희귀질환 관련성 |
|---|---|---|---|---|
| 01 | No Finding | 이상 없음 | 전체 | 기준선 |
| 02 | Enlarged Cardiomediastinum | 심장종격 비대 | 중앙 | 중간 |
| 03 | Cardiomegaly | 심비대 | 심장 | 중간 |
| 04 | Lung Opacity | 폐 혼탁 | 폐 실질 | 높음 |
| 05 | Lung Lesion | 폐 병변 | 폐 국소 | 높음 |
| 06 | Edema | 폐부종 | 폐 전체 | 중간 |
| 07 | Consolidation | 경화 | 폐 실질 | 높음 |
| 08 | Pneumonia | 폐렴 | 폐 실질 | 중간 |
| 09 | Atelectasis | 무기폐 | 폐 부분 | 중간 |
| 10 | Pneumothorax | 기흉 | 흉막강 | 매우 높음 |
| 11 | Pleural Effusion | 흉막 삼출 | 흉막강 | 높음 |
| 12 | Pleural Other | 기타 흉막 이상 | 흉막 | 중간 |
| 13 | Fracture | 골절 | 뼈 | 낮음 |
| 14 | Support Devices | 지지 장치 | 전체 | 보정 변수 |

14 라벨은 평면적 목록이 아니라 의학적 포함 관계를 가진다. **Lung Opacity**는 Atelectasis·Edema·Lung Lesion·Consolidation을 포괄하는 상위 우산 개념이며, **Consolidation**은 다시 그 하위에 Pneumonia를 포함한다. **Enlarged Cardiomediastinum**은 Cardiomegaly를 하위로 가진다. 이 계층은 모델 학습에서 결정적으로 중요하다. Pneumonia가 양성이면 Consolidation도 양성이어야 하고, Consolidation이 양성이면 Lung Opacity도 양성이어야 한다는 논리적 제약이 성립하기 때문이다. 이 계층 관계는 6.3절의 불확실 라벨 정책 설계의 핵심 근거가 된다.

특수 규칙으로, CheXpert 라벨러는 보조기구만 있는 정상 폐를 `No Finding = positive`로도 라벨링하도록 설계되어 있어, `No Finding`과 `Support Devices`가 동시에 양성인 경우가 논리적으로 허용된다.

## 5.2 라벨별 의학적 정의와 희귀질환 연결

**Label 01 — No Finding (이상 없음).** 13개 병리 소견이 모두 음성인 정상 상태다. 양쪽 폐가 균일하고, 심장이 흉곽 너비의 50% 미만이며, 횡격막 경계가 선명한 것이 핵심이다. 다만 흉부 X-ray가 정상이라고 해서 질환이 없는 것은 아니다. 초기 폐섬유증, 초기 폐암 결절, 폐고혈압, 폐색전증, 초기 결핵 등은 정상 X-ray로 나타날 수 있어, 본 시스템은 No Finding 확률이 높아도 HPO와 혈액검사를 반드시 종합 판단한다.

**Label 02 — Enlarged Cardiomediastinum (심장종격 비대).** 흉부 중앙부(심장·대혈관·기관지가 모인 종격)가 비정상적으로 넓어진 상태다. 진단 기준은 상종격 폭 ≥8 cm(PA view), mediastinal-to-thoracic ratio >0.25이다. 대동맥류·대동맥박리·림프종·흉선종·종격 종양이 주요 원인이며, 희귀질환으로는 Castleman disease(ORPHA:160), 섬유성 종격동염이 연결된다. Swan-Ganz 카테터와 함께 관찰되면 폐동맥고혈압(PAH)을 강력히 시사한다.

**Label 03 — Cardiomegaly (심비대).** 심장 크기가 흉곽 너비의 50% 이상(심흉비 CTR >0.5, PA 기립영상)으로 커진 상태다. 폐동맥고혈압이 우심실 비대를 거쳐 심비대로 진행하거나, 특발성 폐섬유증(IPF) 말기에 이차적으로 발생한다. 젊은 환자의 원인 불명 심비대는 비후성 심근병증(HCM, ORPHA:99739)·Fabry병·아밀로이드증 같은 유전·대사 희귀질환을 적극 감별해야 한다.

**Label 04 — Lung Opacity (폐 혼탁).** 폐에 비정상적인 흰 음영이 나타난 상태로, CheXpert에서 가장 광범위한 우산 라벨이다. 간유리음영(Ground-Glass Opacity, 혈관이 비치는 흐릿한 음영)과 경화(Consolidation, 혈관이 보이지 않는 고음영)의 두 패턴으로 나뉜다. 분포 패턴(대엽성·분절성·반점상·미만성·주변부·중심부)이 원인 추정의 핵심이며, 간질성 폐질환(IPF ORPHA:2032, NSIP, 과민성 폐렴), 폐포단백증(PAP, ORPHA:747, crazy-paving 패턴), 폐포출혈(Goodpasture syndrome ORPHA:375)과 연결된다.

**Label 05 — Lung Lesion (폐 병변).** 폐 안의 국소적 비정상 구조물로, 결절(≤3 cm)과 종괴(>3 cm)로 구분된다. 분엽화 경계·스피큘레이션·상엽 위치·공동화는 악성을, 완전 중심성 석회화·지방 성분·2년 이상 불변은 양성을 시사한다. 다발성 낭종은 LAM·BHD증후군·LCH, 공동성 병변은 GPA(베게너 육아종증, ORPHA:900)·진균 감염, 다발성 결절은 사르코이도시스와 연결된다.

**Label 06 — Edema (폐부종).** 폐포 안팎에 과도한 액체가 고인 상태다. 1단계 폐문부 혈관 확장, 2단계 Kerley B선, 3단계 양측 나비형(butterfly) 음영으로 진행한다. 심비대를 동반하고 양측 대칭이며 Kerley B선이 보이면 심인성, 심장이 정상이고 양측 비대칭이면 비심인성(ARDS)이다. Edema가 단독으로 나타나고 심장이 정상이면 폐정맥폐색증(PVOD, ORPHA:31837)·폐모세혈관혈관종증(PCH, ORPHA:199241) 같은 초희귀 폐혈관 질환을 탐색한다.

**Label 07 — Consolidation (경화).** 폐포 안의 공기가 액체·세포·조직으로 완전히 대체된 상태다. 가장 특징적 소견은 경화된 폐 속에서 기관지만 검게 보이는 공기 기관지조영(air bronchogram)이며, 폐 부피가 유지된다는 점에서 무기폐와 결정적으로 구분된다. 분포 패턴이 감별의 단서로, 이동성 경화는 기질화 폐렴(COP, ORPHA:171876), 중심부 나비형은 폐포단백증(PAP), 주변부 분포는 호산구성 폐렴을 시사한다.

**Label 08 — Pneumonia (폐렴).** 병원체가 폐포를 감염시켜 염증이 생긴 상태다. CheXpert 원 논문은 폐렴이 본질적으로 임상 진단임에도 "감염을 시사하는 영상"을 대표하기 위해 라벨로 포함했다고 명시한다. 대엽성·기관지성·간질성의 세 패턴이 있으며, 반복성 폐렴은 원발성 섬모운동이상증(PCD)·만성 육아종병·CVID(면역결핍)를, 진균성 패턴은 Aspergillus 관련 희귀질환을 시사한다. 백혈구·CRP·프로칼시토닌 등 혈액 지표와 결합 시 진단 정확도가 크게 상승한다.

**Label 09 — Atelectasis (무기폐).** 폐의 일부 또는 전체에서 공기가 빠져 쪼그라든 상태로, 폐 부피 감소가 핵심이다(경화와의 결정적 차이). 흡수성·압박성·반흔성·유착성·수동성으로 분류되며, 반복적 중엽 무기폐는 원발성 섬모운동이상증(PCD, ORPHA:244)을 시사한다.

**Label 10 — Pneumothorax (기흉).** 흉막강에 공기가 새어 들어가 폐가 압박·허탈된 상태다. 장측 흉막선(visceral pleural line) 바깥쪽의 폐 혈관음영 소실이 진단의 핵심이며, 종격이 반대쪽으로 밀리는 긴장성 기흉은 응급이다. **기흉은 RareLinkAI 관점에서 가장 풍부한 희귀질환 연결점을 가진 라벨이다.** 젊은 여성의 반복·양측 기흉에 낭종을 동반하면 림프관평활근종증(LAM, ORPHA:538), 피부 섬유낭종을 동반하면 BHD증후군(ORPHA:122), 흡연 청년의 상엽 결절·낭종을 동반하면 PLCH(ORPHA:79127), 그 밖에 Marfan증후군(ORPHA:558)·알파-1 항트립신 결핍(AATD, ORPHA:60)·낭성섬유증(ORPHA:586)이 감별 대상이다.

**Label 11 — Pleural Effusion (흉막 삼출).** 흉막강에 액체가 고인 상태다. 늑골횡격막각 둔화(가장 초기 징후), 초승달 모양 메니스커스 징후, 대량 시 한쪽 폐 전체의 백화가 단계적으로 나타난다. 삼출액 성상이 중요한데, 우유빛의 유미흉(chylothorax)은 LAM·림프관종·황색손발톱증후군과 직결된다. 유미흉 HPO(HP:0010310)와 흉막 삼출이 젊은 여성에서 함께 관찰되면 LAM을 강력히 시사한다.

**Label 12 — Pleural Other (기타 흉막 이상).** 흉막 삼출 외의 흉막 비후·석회화·섬유판(plaque)·종양을 포괄하는 라벨로, CheXpert에서 불확실성이 가장 높은 라벨 중 하나다. 석면 노출의 특이적 표지인 흉막 섬유판, 그리고 악성 중피종(ORPHA:50251)·고립성 섬유종양(ORPHA:1335)과 연결된다.

**Label 13 — Fracture (골절).** 흉부 X-ray에 함께 찍히는 갈비뼈·쇄골·흉골·척추의 골절 소견이다. 단독으로는 희귀 폐질환 관련성이 낮으나, 다발성·병적 골절은 골형성부전증(OI, ORPHA:666)을, Lung Lesion과 동시에 나타나면 뼈와 폐를 동시에 침범하는 LCH를 시사한다. HPO 연결은 HP:0002659(골절 감수성 증가)이다.

**Label 14 — Support Devices (지지 장치).** 흉부 X-ray에 찍힌 기관내튜브(ETT)·중심정맥관(CVC)·흉관·심박동기·Swan-Ganz 카테터 등 의료기구다. 각 장치의 정상 위치 확인은 방사선 판독의 핵심 업무다. 본 시스템에서 Support Devices 확률이 높으면 다른 라벨의 신뢰도를 하향 보정하며(중환자 영상의 잡음 보정), 동시에 환자 중증도의 프록시(proxy)로 활용한다.

## 5.3 라벨 조합–희귀질환 매핑

개별 라벨보다 라벨의 조합이 희귀질환을 더 강력하게 시사한다. RareLinkAI는 Phase 3 스코어링에서 다음 조합 규칙을 활용한다.

| 라벨 조합 | 강력 시사 희귀질환 |
|---|---|
| Pneumothorax + Lung Lesion(낭종) + 여성 | 림프관평활근종증(LAM) |
| Lung Lesion(낭종) + Fracture | 랑게르한스세포 조직구증(LCH) |
| Consolidation(이동성) | 기질화 폐렴(COP) |
| Consolidation(중심부 나비형) | 폐포단백증(PAP) |
| Pleural Effusion(유미흉) + 여성 | LAM, 림프관종 |
| Edema(비심인성) | PVOD, PCH |
| Enlarged Cardiomediastinum + Swan-Ganz | 폐동맥고혈압(PAH) |
| Lung Opacity(양측 간유리음영) | 간질성 폐질환군(IPF, NSIP 등) |

라벨별 주 연관 희귀질환 범주를 정리하면, Pneumothorax는 구조적·유전/대사 범주(LAM·BHD·PLCH·AATD·Marfan·CF), Lung Lesion·공동성 Consolidation은 감염/면역 범주(GPA·EGPA·결핵·아스페르길루스종), 비심인성 Edema는 혈관/순환 범주(PVOD·PCH·만성혈전색전성 폐고혈압), 미만성 Lung Opacity는 섬유화/간질성 범주(IPF·NSIP·과민성 폐렴·PAP)와 주로 맞물린다. 부록 C에 전체 매핑표를 수록한다.

---

# 6. 실험 및 결과

## 6.1 데이터셋

| 데이터셋 | 규모 | 역할 |
|---|---|---|
| CheXpert (Stanford) | 224,316장 흉부 X-ray | SooNet 기본 학습 |
| MIMIC-CXR | 377,110장 흉부 X-ray | 교차도메인(도메인 시프트) 검증 |
| MIMIC-IV (전체) | 360,000+ 환자 Lab·Vital | 멀티모달 스코어링 검증 |
| Orphadata (2025-12-09) | 11,456종 희귀질환 | Phase 5 지식베이스 + RAG Tier 1 |
| HPO Ontology | 11,514 텀 / 282,728 어노테이션 | 증상-질환 매핑 |

CheXpert와 MIMIC-CXR을 함께 사용한 이유는 도메인 시프트(domain shift) 검증을 위함이다. CheXpert는 미국 단일 기관 데이터로, 이 데이터로만 학습한 모델은 촬영 장비·환자 분포가 다른 환경에서 성능이 저하될 수 있다. 본 프로젝트는 MIMIC-CXR을 교차도메인 검증셋으로 사용하여 일반화 성능을 정량화하였다.

## 6.2 SooNet X-ray 모델 성능

| 평가 항목 | 지표 | 결과 |
|---|---|---|
| CheXpert 검증셋 | 평균 AUROC | **0.8094** |
| MIMIC-CXR 교차도메인 | 평균 AUROC | **0.7384** |
| 2클래스(PE vs PTX) | AUROC / F1 | **0.8863 / 0.8101** |
| Pleural Effusion (단일 라벨) | AUROC | **0.9179** |
| Edema (단일 라벨) | AUROC | **0.9501** |

SooNet은 CheXpert 검증셋에서 평균 AUROC 0.8094를 달성하여 목표치(0.80+)를 충족하였다. 교차도메인 성능은 0.7384로, CheXpert 대비 약 0.07의 하락이 관찰되었다. 이는 미국 단일기관 학습 데이터의 도메인 편향이 정량적으로 드러난 것으로, 한국 데이터 추가 시 도메인 적응 재검증이 필요함을 의미한다.

주목할 점은 임상적으로 긴급한 소견에서 성능이 특히 높다는 것이다. Pleural Effusion(0.9179)과 Edema(0.9501)는 모두 즉각적 처치가 필요할 수 있는 소견으로, 이들에 대한 높은 AUROC는 9.3절의 '워크리스트 트리아지' 시나리오 — 응급 소견을 자동 플래그하여 판독 우선순위를 조정하는 — 의 실현 가능성을 뒷받침한다.

## 6.3 불확실 라벨(Uncertainty Label) 정책 실험

### 6.3.1 실험 배경

CheXpert 자동 라벨러는 판독문에서 확신할 수 없는 소견에 불확실 라벨(-1)을 부여한다. 이 불확실 샘플을 학습에 어떻게 반영할지에 따라 모델 성능이 달라진다. 본 프로젝트는 2026년 4월 15일 이 문제를 집중 분석하여, 단일 정책을 14 라벨 전체에 일괄 적용하는 대신 **병변별 차등 정책(per-class policy)**을 채택하는 것이 타당함을 검증하였다.

### 6.3.2 정책별 AUROC 비교

CheXpert 원 논문(Irvin 등, AAAI 2019, Table 3)의 검증셋(200 studies) 결과는 다음과 같다.

| 병변 | U-Ignore | U-Zeros | U-Ones | U-SelfTrained | U-MultiClass |
|---|---|---|---|---|---|
| Atelectasis | 0.818 | 0.811 | **0.858** | 0.833 | 0.821 |
| Cardiomegaly | 0.828 | 0.840 | 0.832 | 0.831 | **0.854** |
| Consolidation | 0.938 | 0.932 | 0.899 | **0.939** | 0.937 |
| Edema | 0.934 | 0.929 | **0.941** | 0.935 | 0.928 |
| Pleural Effusion | 0.928 | 0.931 | 0.934 | 0.932 | **0.936** |

이 표의 핵심 발견은 **최적 정책이 병변마다 다르다**는 것이다. Atelectasis와 Edema는 U-Ones가, Consolidation은 U-Ignore(또는 U-SelfTrained)가, Cardiomegaly와 Pleural Effusion은 U-MultiClass가 최고 성능을 보인다. 특히 Consolidation에 U-Ones를 적용하면 AUROC가 약 0.04 하락하는데, 이는 불확실 라벨을 무비판적으로 양성 처리하면 오히려 성능이 악화될 수 있음을 보여준다.

### 6.3.3 불확실 비율과 정책 결정

왜 원 논문이 14 라벨 중 5개만 비교했는지를 분석한 결과, 그 답은 각 라벨의 불확실 라벨 비율에 있었다. 불확실 비율이 낮은 라벨은 어떤 정책을 써도 AUROC 차이가 거의 없어 분석 가치가 적기 때문이다.

| 병변 | 불확실 비율 | 논문 근거 | 본 프로젝트 채택 정책 | 근거 |
|---|---|---|---|---|
| Atelectasis | 15.7% | 강함 | **U-Ones** | "possible atelectasis"는 임상적으로 양성에 가까움 |
| Consolidation | 12.8% | 강함 | **U-Ignore** | atelectasis와 혼동 → U-Ones 시 -0.04 손실 |
| Pneumonia | 8.3% | 없음 | **U-Ignore** | Consolidation과 유사 구조로 추정 |
| Edema | 6.2% | 강함 | **U-Ones** | 불확실 표현도 실제 양성인 경우 많음 |
| Enlarged Cardiomediastinum | 5.4% | 없음 | **U-Ones** | Cardiomegaly와 연관 |
| Pleural Effusion | 5.0% | 강함 | **U-Ones** | U-MultiClass와 유사 성능, 구현 단순 |
| Cardiomegaly | 3.5% | 강함 | **U-Zeros** | U-MultiClass 최적이나 구현 복잡, U-Zeros 차선 |
| Lung Opacity | 2.3% | 없음 | U-Zeros | 상위 우산 클래스, 불확실 영향 미미 |
| Pneumothorax | 1.4% | 없음 | U-Zeros | 불확실 비율 낮음 |
| Pleural Other | 0.9% | 없음 | U-Zeros | 희귀 클래스 |
| Lung Lesion | 0.6% | 없음 | U-Zeros | 희귀 클래스 |
| Support Devices | 0.5% | 없음 | U-Zeros | 비병변 |
| Fracture | 0.3% | 없음 | U-Zeros | 희귀 클래스 |
| No Finding | 0.0% | 없음 | U-Zeros | 구조적으로 -1 라벨 불가 |

### 6.3.4 최종 채택 정책

위 분석을 종합하여 본 프로젝트가 채택한 per-class 정책은 다음과 같다.

```
U-Ones   : Atelectasis, Edema, Pleural Effusion, Enlarged Cardiomediastinum
U-Ignore : Consolidation, Pneumonia
U-Zeros  : Cardiomegaly, Lung Opacity, Pneumothorax, Lung Lesion,
           Pleural Other, Fracture, Support Devices, No Finding
```

추가 주의사항으로, U-Zeros 처리 시 `No Finding` 또는 `Support Devices`만 양성인 행은 제거해야 한다. 보조기구만 있는 정상 폐가 라벨 노이즈로 작용하기 때문이다(U-Ignore·U-Ones에서는 해당 행 유지).

### 6.3.5 결론 및 향후 개선 방향

본 실험의 결론은 세 가지다. 첫째, 논문 근거로 정책이 명확한 라벨은 Atelectasis(U-Ones)·Consolidation(U-Ignore)·Edema(U-Ones)이다. 둘째, 불확실 비율이 높으면서 논문 공백인 유일한 미지수는 Pneumonia(8.3%)로, 실제 실험으로 검증할 가치가 있다. 셋째, 나머지 9개 라벨은 정책보다 데이터 품질과 모델 구조가 훨씬 중요하다.

향후 개선 방향으로, 하드 라벨(-1을 0 또는 1로 강제 변환) 대신 **Label Smoothing**(soft target, 예: 0.55 사용)을 적용하고 병변 계층 구조(Consolidation ⊂ Lung Opacity)를 반영하는 접근이 있다. Pham 등(2021, Neurocomputing, arXiv:1911.06475)은 이 방식으로 CheXpert 리더보드 1위(mean AUC 0.930)를 달성하였다. 더 나아가 Rep-GLS(2025, arXiv:2508.02495)는 방사선 판독문의 "probable/possible/likely" 같은 표현을 LLM으로 읽어 표본별 연속 스무딩률을 적용한다. 코드 수준에서는 현재 `prepare_chexpert_df`가 단일 `policy`로 전체를 통일하는 것을, 위 표와 같은 병변별 dict 형태(`PER_CLASS_POLICY`)로 개선하는 것이 권장된다.

## 6.4 LIRICAL LR 희귀질환 랭킹 검증

Phase 5의 LIRICAL LR 엔진은 v4 DDL 적용 후 정상 작동을 확인하였다. 4.6.4절의 LAM 사례에서 보듯, 호흡곤란(LR 30배)·흉막삼출(LR 30.6배) 같은 표현형이 결합되면 사전확률 보정(log_prior = -5.30) 이후에도 lr_value가 임계값 5.0을 초과하여 후보 목록에 정상 피스팅됨을 검증하였다. 출력 레코드는 `total_listed_count`·`top_lr_score`·`top_lr_orphacode`를 포함하며, 단독 invoke 기준 약 70초 내 완료된다.

## 6.5 통합 지식베이스 구축 성과

| 영역 | 목표 | 결과 |
|---|---|---|
| SooNet 모델 | AUROC 0.80+ | 달성 (CheXpert 0.8094) |
| 2클래스 검증 | 이진 분류 성능 | 확인 (PE vs PTX 0.8863) |
| 지식베이스 | 일반·기타 + 희귀 통합 | 일반·기타 105개 + 희귀 322개 |
| Backend MVP | FastAPI 다단계 | `/api/v1/diagnose` 운영 |
| 희귀질환 엔진 | HPO IC 가중 매칭 | LIRICAL LR 엔진 구현 |
| 임상 스코어 | NEWS2·qSOFA·CURB-65·PESI | 4종 스코어링 + 37 VRH 파라미터 |
| Lab 파서 | PDF 자동 파싱 | 실물 결과지 → 89항목 자동 매핑 |
| RAG | Orphanet + PubMed | 5개 외부 API 병렬 호출 구현 |

## 6.6 파이프라인 통합 실행 검증

Step Functions State Machine을 통한 종단간(end-to-end) 실행 검증 결과, ASL v1 실행은 총 5분 30초에 상태 SUCCEEDED로 종결되었다. 단계별로 Phase 2(0.9초)·Phase 3(16초)·RAG(12초)는 정상 완료되었으나, 초기 버전에서 Phase 5 LR과 Phase 4(mode=real)가 VPC 내 동시 cold start 시 5분 timeout에 도달하는 이슈가 관찰되었다. Phase 5 LR은 단독 invoke 시 약 70초에 정상 완료되므로, 원인은 알고리즘 자체가 아니라 동시성(ENI·RDS 커넥션 경쟁)으로 진단되었다.

v2 이후 (1) Phase 3를 순차 실행으로 분리하여 Phase 5와의 동시 cold start를 회피하고, (2) Phase 4를 mock 모드로 전환하여 Bedrock 호출 hang을 우회하며, (3) RAG에 Catch+Pass 패턴을 적용하여 일부 Phase 실패와 무관하게 전체 execution이 SUCCEEDED로 종결되도록 보강하였다. 이로써 파이프라인은 부분 실패에 내성(fault tolerance)을 갖춘 형태로 안정화되었다. 잔여 과제인 Phase 4·5 동시 cold start의 근본 해결책은 10장에서 논한다.

---

# 7. 지식베이스 구축

RareLinkAI의 진단 능력은 모델만이 아니라 정교하게 구축된 지식베이스에서 나온다. 지식베이스는 일반·기타 폐질환과 희귀 폐질환의 두 축으로 구성되며, 모든 가중치는 SSOT YAML 파일에 집약된다.

## 7.1 일반·기타 폐질환 데이터베이스

일반·기타 폐질환 DB는 J코드 폐질환 53개(일반)와 J코드 외 폐관련 질환 52개(기타)를 합한 **105개 질환**을 포괄한다.

| 파일 | 버전 | 크기 | 역할 |
|---|---|---|---|
| 일반_폐질환_데이터베이스_v9.xlsx | v9 | 56.7 KB | J코드 폐질환 53개 |
| 기타_폐관련_질환_데이터베이스_v9.xlsx | v9 | 48.8 KB | 기타 폐관련 52개 |
| lung_disease_profiles_v3_7.yaml | v3.7 | 2.3 MB | 105개 질환 S·L·R·M 가중치 SSOT |
| lab_reference_ranges_v9_5.yaml | v9.5 | 345.9 KB | 혈액검사 정상 기준치 |

각 Excel DB는 4개 시트(질병 리스트·HPO 표현형·영상/진단 포인트·변경 이력)로 구성된다. 일반 DB는 823행, 기타 DB는 869행의 HPO 표현형 매핑을 보유한다. 가중치 SSOT인 `lung_disease_profiles_v3_7.yaml`은 GOLD 2026 보고서를 반영하여 COPD 계열 7건의 가중치를 갱신한 최신 버전이다.

## 7.2 희귀 폐질환 데이터베이스

희귀 폐질환 DB는 OrphaCode 기반의 **322개 활성 질환**을 포함한다. 이는 초기 버전의 536개에서 비폐질환·비활성 항목(혈액암·안질환·간질환 등 9개 등)을 정리하여 재편한 결과로, 폐질환 감별이라는 시스템 목적에 맞게 범위를 정밀화한 것이다.

| 파일 | 버전 | 크기 | 역할 |
|---|---|---|---|
| 희귀_폐질환_데이터베이스 | v5(S3) / v7.1(로컬) | 341.7 KB | 322개 희귀 폐질환 |
| rare_disease_profiles_v3_1.yaml | v3.1 | 1.1 MB | 322개 희귀질환 임상 프로필 SSOT |
| phenotype.hpoa | 2026-02-16 | 35.3 MB | Orphanet HPO 표현형 (LR 분자) |
| hpo_background_freq.json | 2026-05-18 | 328.4 KB | HPO 배경 빈도 (LR 분모) |

희귀 DB는 질병 리스트(OrphaCode·ICD-10/11/9 다중 매핑·유전형·lr_category), HPO 표현형 리스트(**5,135행** — 322개 질환 × HPO 코드, 빈도·발현율 포함), 영상/진단 포인트, 비폐질환 검토의 4개 시트로 구성된다. 핵심 데이터 소스는 Orphadata(11,456종 희귀질환), HPO(증상 표준 코드), MIMIC-IV(실제 환자 EHR로 혈액 수치 보충), NCBI MedGen(유전질환·유전자 정보)의 4가지다.

## 7.3 데이터 거버넌스 — 버전 관리와 SSOT 원칙

지식베이스 운영의 핵심 원칙은 **단일 진실 원천(SSOT)**이다. 모든 가중치는 YAML 한 파일에 집약되어, 이를 수정하면 시스템 전체에 자동 반영된다. 또한 버전 관리에서 Slack 캔버스 버전과 S3 버전이 불일치할 경우 **S3 콘솔을 기준으로 삼는다**. 실제로 2026년 5월 19일 검증 시 Slack 기준 최신(lung v3.3 / lab v9.4)보다 S3가 더 최신(lung v3.7 / lab v9.5)임이 확인되어, S3를 정본(authoritative source)으로 채택하였다. 신규 참조 파일로 `chexpert_label_reference_v1.yaml`(CheXpert 14라벨 정의), `icd10_reference_v1.json`, `korean_hpo_dictionary_v1.json`(한국어 HPO 사전), `multilingual_phenotype_lexicon_v1.json`(다국어 표현형 사전)이 추가되어 다국어·표준코드 연동을 강화하였다.

---

# 8. 규제 정합성·윤리·데이터 거버넌스

의료 AI는 기술적 완성도만으로 현장에 진입할 수 없다. 본 장은 RareLinkAI의 규제 위치, 윤리 설계, 데이터 라이선스를 정리한다. 프로젝트는 인프라 리포지토리에 별도 규제 정합성 문서(`infra/aws/MEDICAL_COMPLIANCE.md`)를 두어 FDA·MFDS·EU AI Act·HIPAA 정합성을 관리한다.

## 8.1 의료기기 규제상의 위치

RareLinkAI는 현재 **식품의약품안전처(MFDS) 의료기기 허가를 취득하지 않은 상태**로, 연구·교육 목적으로만 사용이 허용된다. 시스템 UI와 모든 산출 리포트에는 면책 조항이 명시된다. 임상 현장 진입을 위해서는 소프트웨어 의료기기(SaMD) 분류에 따른 인허가 절차가 선행되어야 한다.

규제 환경을 요약하면, 한국 MFDS는 AI 기반 진단 보조 소프트웨어를 SaMD로 분류하여 임상시험·성능평가를 요구하며, 미국 FDA는 유사 제품을 510(k) 또는 De Novo 경로로 심사한다. 유럽연합의 AI Act는 의료 진단 AI를 고위험(high-risk) 등급으로 분류하여 위험관리·데이터 거버넌스·투명성·인간 감독 의무를 부과한다. 환자 식별정보를 다루는 운영 단계에서는 미국 HIPAA 적용 시 AWS와의 BAA(Business Associate Addendum) 체결이 필요하며, 본 프로젝트의 배포 가이드는 Bedrock을 HIPAA 환경에서 운영할 경우 AWS Artifact 콘솔을 통한 BAA 체결을 절차에 포함하고 있다. 국내 적용 시에는 개인정보보호법·의료법 및 보건의료데이터 활용 가이드라인을 추가로 준수해야 한다.

## 8.2 윤리 설계 — CDSS 원칙과 환각 방지

본 시스템의 윤리 설계는 "AI는 의사를 대체하지 않는다"는 임상 의사결정 지원 시스템(CDSS)의 제1원칙 위에 서 있다. 효과적인 CDSS는 의사의 판단을 대체하는 것이 아니라 임상 결정 지점(decision point)에서 정보를 제공한다. RareLinkAI는 (1) X-ray 판독 보조, (2) 감별진단 후보 제시, (3) 희귀질환 경보, (4) 치료 근거 제시의 네 결정 지점에 개입하되, **최종 진단 권한은 항상 담당 의사에게 귀속**된다.

LLM 환각의 위험은 다층 방어로 통제된다. Phase 4의 LLM은 Context 외 정보 사용 금지, 존재하지 않는 PMID 생성 금지, 일반 의학 지식 사용 금지를 강제받으며, RAG는 Tier 1(Orphanet 구조화 데이터)을 우선 사용하고 모든 주장에 출처 인용을 의무화한다. 또한 출력 JSON의 `self_check` 필드가 "모든 PMID가 컨텍스트 내에 존재하는가", "Negative HPO가 반영되었는가", "HIGH 신뢰도 항목에 근거가 있는가"를 스스로 점검한다. 이는 의사가 "AI가 그렇게 말했다"가 아니라 검증 가능한 1차 출처를 근거로 의사결정하도록 보장하기 위한 장치다.

## 8.3 데이터 라이선스

| 데이터·API | 라이선스 | 영리 사용 |
|---|---|---|
| Orphanet / Orphadata | CC BY 4.0 | 허용 (출처 명시 조건) |
| Monarch Initiative | CC BY 3.0 | 허용 (출처 명시 조건) |
| ClinicalTrials.gov | 공공 도메인 | 자유 사용 가능 |
| PubMed E-utilities | 논문별 저작권 상이 | 초록 처리·변환 가능, 원문 재배포 불가 |
| PubCaseFinder | 조건부 | 의사 전용, 상업적 사용은 별도 확인 필요 |
| CheXpert / MIMIC | 연구용 데이터 사용 협약 | 학술·연구 목적 |

데이터 라이선스 검토는 권미라 팀원이 담당하였으며, 영리 서비스화 시 PubCaseFinder의 상업적 사용 조건과 PubMed 원문 재배포 제한이 핵심 점검 사항으로 식별되었다.

---

# 9. 적용 시나리오 및 사업화 전략

## 9.1 적용 시나리오

RareLinkAI는 단일 제품이 아니라, Phase 조합에 따라 여러 임상 현장에 대응하는 플랫폼으로 설계되었다.

**시나리오 A — 지방·도서 1차 의료기관(진단 공백 해소).** 임상 결정 지점은 "이 환자를 CT가 있는 상급 병원으로 전원해야 하는가"이다. CT·MRI가 없는 환경에서 Phase 2(SooNet 14소견 + Grad-CAM)와 Phase 3(혈액검사 89항목 + NEWS2/qSOFA)로 위험도를 수치화하여, 37.8%의 불확실 구간에서 의사 직관에만 의존하던 전원 결정에 객관적 근거를 제공한다. Phase 4 리포트는 전원 의뢰서를 대체할 수 있다.

**시나리오 B — 희귀질환 전문센터(진단 방랑 단축).** 임상 결정 지점은 "이 ILD가 일반 특발성 섬유화인가 희귀 유전질환인가, 어떤 유전자 검사를 먼저 할 것인가"이다. Phase 1 트리거(나이<40 + ILD 패턴)가 발동하면 Orphanet HPO IC 보정으로 Top-N 희귀질환 후보를 제시하고, 단일검사/패널/WES/WGS를 자동 선별한다.

**시나리오 C — 대형병원 영상의학과(워크리스트 트리아지).** 24시간 판독 큐의 수백 건 중 무엇을 먼저 볼지 결정하는 지점이다. Phase 2의 고성능 소견 탐지(Pleural Effusion 0.9179, Edema 0.9501)로 긴급 소견을 자동 플래그하여, 응급 처치 골든타임을 확보한다.

**시나리오 D — 원격의료·공중보건 스크리닝.** 영상의학 전문 인력이 없는 지역에서 REST API(`/api/v1/diagnose`)로 X-ray 경로 + Lab JSON을 받아 JSON으로 응답하여, 어떤 플랫폼에도 연동 가능하다.

**시나리오 E — 국가 건강검진 사업 연계.** 건강검진 X-ray를 SooNet으로 즉시 분석하고 공단 혈액검사 PDF를 자동 파싱하여, 연간 1,000만 건 규모의 검진 영상에 일괄 적용 가능성을 제시한다.

**시나리오 F — 희귀질환 환자 2차 의견.** 환자가 증상 목록(한국어 → HPO 자동 변환)과 혈액검사 PDF를 업로드하면 희귀질환 후보 Top-N과 다음 검사 단계를 제안한다.

## 9.2 시장 분석과 사업화 모델

글로벌 의료 AI 영상 분석 시장은 2023년 20.9억 달러에서 2030년 81.5억 달러(CAGR 21.7%)로, 국내 의료 AI 시장은 2023년 3,680억 원에서 2028년 1조 2,000억 원(CAGR 26.7%)으로 성장이 전망된다. 그 안에서 X-ray·혈액검사·HPO·희귀질환 스크리닝·RAG 치료 근거를 통합한 서비스는 상용 사례가 확인되지 않는 명확한 시장 공백이다.

| 사업 모델 | 대상 | 수익 구조 |
|---|---|---|
| B2B SaaS | 1·2차 의료기관 영상의학과 | API 월정액 + 케이스당 과금 |
| B2B 전문 클리닉 | 유전 클리닉·대학병원 희귀질환센터 | 진단 보조 소견서 리포트당 과금 |
| B2G 공공의료 | 보건복지부·건강보험심사평가원 | 정부 조달·공공 계약(MFDS 허가 후) |
| 글로벌 저자원 시장 | 개발도상국 보건부·NGO | UN·WHO 파트너십, 오픈소스 라이선스 |

사업화 우선순위는 MFDS 허가 취득을 분기점으로, 허가 이전에는 연구·교육용 라이선스와 B2B 전문 클리닉 파일럿을, 허가 이후에는 B2B SaaS와 B2G 공공의료로 확장하는 단계적 접근이 적절하다.

---

# 10. 한계 및 향후 과제

## 10.1 기술적 한계와 완화 전략

| 리스크 | 현황 | 완화 전략 |
|---|---|---|
| RAG 환각 | LLM의 희귀질환 지식 공백 | Tier 1 구조화 데이터 우선 + 출처 인용 의무화 + 의사 최종 확인 |
| 데이터 도메인 시프트 | CheXpert(미국) 학습 편향, 교차도메인 0.7384 | 단기 TTA·CLAHE 전처리 통일 / 중기 VinDr-CXR(베트남 15,000장) 아시아 파인튜닝 |
| Bedrock 비용 | LLM 호출 단가 | Claude Haiku 우선 + 토큰 제한 + rag_api_cache TTL 7일 캐싱 |
| 법적 책임 | AI 오진 책임 소재 | 보조 시스템 명시 + 최종 판단 의사 귀속 + UI 면책 조항 |
| Lab→HPO 변환 공백 | LAB_HPO_MAP 미완성 | best-effort 매핑, LOINC→HPO 매핑 테이블 확장 진행 중 |

## 10.2 운영상 알려진 이슈

통합 검증에서 식별된 운영 이슈는 다음과 같다. 첫째, Phase 4·5 Lambda를 VPC 내에서 동시 cold start할 때 ENI·RDS 커넥션 경쟁으로 timeout이 발생한다(단독 invoke는 정상). 둘째, 우리 팀의 `rare-link-pipeline-dev`와 별도 세션에서 생성된 `say2-2team-rare-link-pipeline` 두 State Machine의 통합·정리가 필요하다. 셋째, ASL의 Phase 1이 임시 Pass state로 남아 있어 실제 `phase1-symptom-dev` Lambda 호출 task로 교체해야 한다. 넷째, 배포 편의를 위해 `fhir-ec2-role`에 임시 부착한 광범위 IAM 정책 9종(Lambda·CloudFormation·S3·IAM FullAccess 등)은 보안상 배포 종료 후 분리하고, 의료데이터용 역할과 배포 전용 역할을 분리해야 한다.

## 10.3 향후 과제(Backlog)

단기 과제로는 (1) Phase 4·5 동시 cold start 해결(Provisioned Concurrency 도입, 순차 실행, 서브넷 확장 중 택일), (2) Phase 4의 Bedrock 리전 수정 검증 후 mock 모드에서 real 모드 복구, (3) ASL Phase 1의 Task state 전환, (4) Phase 2 vision Lambda에 `phase_execution_log` 통합 로깅 추가, (5) `LAB_HPO_MAP`의 LOINC→HPO 매핑 확장이 있다.

중기 과제로는 모델 측면의 Label Smoothing 도입과 병변 계층 구조 반영(6.3.5절), 한국·아시아 데이터셋(VinDr-CXR 등) 추가 파인튜닝을 통한 도메인 적응, 운영 측면의 CloudWatch 운영 대시보드(`recent_errors`, `phase_success_rates_24h` 뷰) 구축, Secrets Manager VPC Endpoint 추가를 통한 NAT 우회·지연 개선, 그리고 규제 측면의 MFDS SaMD 인허가 준비를 위한 임상 성능평가 설계가 포함된다.

장기적으로는 흉부 CT 영상으로의 모달리티 확장, 유전체 데이터의 본격 통합(현재 설계상 4번째 축으로 예정), 그리고 다기관 전향적 임상 검증을 통한 실사용 근거(real-world evidence) 축적이 과제로 남는다.

---

# 11. 결론

본 프로젝트는 흉부 X-ray·혈액검사·임상소견의 3축 멀티모달 데이터를 단일 파이프라인으로 통합하여, 일반 폐질환과 희귀 폐질환을 동시에 감별하는 근거 기반 임상 의사결정 지원 시스템 **Rare-Link AI**를 설계·구현·배포하였다.

기술적으로 본 시스템은 네 가지 성취를 보였다. 첫째, 영상·검체·텍스트라는 이종 모달리티를 HPO 표현형 온톨로지와 LIRICAL 우도비 통계 위에서 정합적으로 결합하는 아키텍처를 완성하였다. 둘째, CheXpert 14 라벨 분류기 SooNet이 검증셋 AUROC 0.8094를 달성하였으며, 불확실 라벨 처리를 병변별 차등 정책으로 최적화하였다. 셋째, 일반·기타 폐질환 105개와 희귀 폐질환 322개를 포괄하는 가중치 기반 지식베이스를 SSOT 원칙으로 구축하였다. 넷째, Lambda·Step Functions·Aurora·Bedrock·CloudFront로 이어지는 AWS 서버리스 아키텍처 위에 전 과정을 재현 가능하게 배포하였다.

임상적으로 본 시스템은 1차 의료기관의 전원 결정 공백과 희귀질환의 진단 방랑이라는 두 공백을 동시에 겨냥하며, RAG 기반 출처 인용과 CDSS 설계 원칙을 통해 "의사를 대체하지 않고 보조하는" 윤리적 위치를 분명히 하였다. 동시에 본 보고서는 도메인 시프트, 동시성 cold start, 규제 인허가 미취득 등 시스템이 현장에 진입하기 위해 넘어야 할 한계 또한 가감 없이 기록하였다.

Rare-Link AI는 부트캠프 기간 내에 개념 증명을 넘어 종단간 동작하는 클라우드 시스템으로 구현되었다는 점에서 의미가 있으며, 향후 모델 고도화·도메인 적응·임상 검증·규제 대응을 단계적으로 수행한다면 지역 의료 격차 해소와 희귀질환 조기 진단이라는 사회적 가치로 연결될 잠재력을 가진다.

---

# 참고문헌

## 데이터셋·X-ray AI

1. Irvin J, Rajpurkar P, et al. **CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.** AAAI 2019. arXiv:1901.07031.
2. Rajpurkar P, Irvin J, et al. **CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.** arXiv:1711.05225, 2017.
3. Johnson AEW, Pollard TJ, et al. **MIMIC-CXR: A De-identified Publicly Available Database of Chest Radiographs with Free-text Reports.** Scientific Data, 2019.
4. Smit A, Jain S, Rajpurkar P, et al. **CheXbert: Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT.** EMNLP 2020.
5. Pham HH, et al. **Interpreting Chest X-rays via CNNs that Exploit Hierarchical Disease Dependencies and Uncertainty Labels.** Neurocomputing, 2021. arXiv:1911.06475.
6. Rep-GLS: **Report-guided Per-sample Label Smoothing for Chest X-ray Classification.** arXiv:2508.02495, 2025.

## 용어 표준·영상의학

7. Hansell DM, Bankier AA, MacMahon H, et al. **Fleischner Society: Glossary of Terms for Thoracic Imaging.** Radiology 2008;246(3):697–722.
8. Brant and Helms. **Fundamentals of Diagnostic Radiology**, 7th ed., Wolters Kluwer, 2018.
9. Webb WR, Müller NL, Naidich DP. **High-Resolution CT of the Lung**, 5th ed.

## 희귀질환·RAG·표현형 AI

10. Robinson PN, et al. **Interpretable Clinical Genomics with a Likelihood Ratio Paradigm (LIRICAL).** American Journal of Human Genetics, 2020. PMID:32755546.
11. **RDguru: An Intelligent Agent for Rare Diseases.** PMC, 2025. (PMC12099370)
12. **A Phenotype-based AI Pipeline Outperforms Human Experts in Differentiating Rare Diseases (PhenoBrain).** Nature npj Digital Medicine, 2025.
13. **DeepRare: Evidence-linked Rare Disease Diagnosis.** 2026.
14. **TreatRAG: A Framework for Personalized Treatment Recommendation.** ACM, 2024.
15. **Improving LLM in Biomedicine with RAG: A Meta-Analysis.** JAMIA, 2025.

## 지식베이스·온톨로지

16. **Orphanet / Orphadata.** https://www.orpha.net (ORPHA code, CC BY 4.0)
17. **Human Phenotype Ontology (HPO).** https://hpo.jax.org
18. **Monarch Initiative.** https://monarchinitiative.org (CC BY 3.0)
19. 질병관리청 희귀질환 헬프라인. https://helpline.kdca.go.kr

## 정책·시장

20. 국내 흉부 X선 AI-CAD 임상 적용 현황. Journal of the Korean Society of Radiology(JKSR), 2024.

---

# 부록 A. AWS 리소스 인벤토리

| 분류 | 리소스 | 식별자·비고 |
|---|---|---|
| 계정·리전 | AWS Account / Region | 666803869796 / ap-northeast-2 |
| Lambda | phase1-symptom-dev | Phase 1 — 임상소견 HPO 추출 |
| Lambda | say2-2team-phase2-vision | Phase 2 — SooNet X-ray 분석 |
| Lambda | phase3-scorer-dev | Phase 3 — 멀티모달 스코어링 (2048 MB / 300s) |
| Lambda | phase4-verifier-dev | Phase 4 — LLM 검증·리랭킹 |
| Lambda | phase5-lr-dev | Phase 5 — LIRICAL LR (1024 MB / 300s) |
| Lambda | phase5-rag-dev | RAG — 최종 보고서 생성 |
| Step Functions | rare-link-pipeline-dev | ASL v4 메인 파이프라인 |
| 데이터베이스 | patient-db-cluster | Aurora PostgreSQL 16.4, 스키마 rarelinkai |
| 네트워크 | vpc-06dd0ad1f2335ea74 | private subnet 2개(2a·2c), NAT GW, VPC Endpoint |
| 보안그룹 | sg-03b9bc5d95699b797 | fhir-ec2-sg, RDS 5432 ingress |
| 스토리지·CDN | say2-2team-bucket / d300v14l8u0wx7.cloudfront.net | S3 + CloudFront |
| 생성형 AI | AWS Bedrock | Claude 3.5 Sonnet / Claude Haiku / Claude Sonnet |
| 관측·보안 | CloudWatch / X-Ray / GuardDuty | 로그·알람·트레이싱·위협탐지 |
| IaC | AWS SAM | CloudFormation 스택 6종 |

# 부록 B. 데이터·모델 파일 버전 관리 대장 (S3 정본 기준)

| 파일 | 버전 | 크기 | 용도 |
|---|---|---|---|
| anatomy_soonet_v5_best.pth | v5 | 43.4 MB | SooNet 배포 모델 가중치 |
| unet_lung_heart_ep5.pth | — | 57.9 MB | 폐·심장 영역 분할 |
| lung_disease_profiles_v3_7.yaml | v3.7 | 2.3 MB | 일반·기타 105개 질환 가중치 SSOT |
| rare_disease_profiles_v3_1.yaml | v3.1 | 1.1 MB | 희귀 322개 질환 임상 프로필 SSOT |
| lab_reference_ranges_v9_5.yaml | v9.5 | 345.9 KB | 혈액검사 정상 기준치 |
| 일반_폐질환_데이터베이스_v9.xlsx | v9 | 56.7 KB | J코드 폐질환 53개 |
| 기타_폐관련_질환_데이터베이스_v9.xlsx | v9 | 48.8 KB | 기타 폐관련 52개 |
| 희귀_폐질환_데이터베이스 | v5(S3)/v7.1(로컬) | 341.7 KB | 희귀 폐질환 322개 |
| phenotype.hpoa | 2026-02-16 | 35.3 MB | Orphanet HPO 표현형 (LR 분자) |
| hpo_background_freq.json | 2026-05-18 | 328.4 KB | HPO 배경 빈도 (LR 분모) |
| hpo_official.json | 2026-05 | 22 MB | 전체 HPO 온톨로지 |
| chexpert_label_reference_v1.yaml | v1 | 65.9 KB | CheXpert 14라벨 정의 |

# 부록 C. CheXpert 14 라벨 – 희귀질환 연결 매핑표

| CheXpert 라벨 | 주 연관 희귀질환 범주 | 대표 ORPHA 사례 |
|---|---|---|
| Pneumothorax (반복/양측) | 구조적 + 유전/대사 | LAM(538), BHD(122), PLCH(79127), AATD(60), Marfan(558), CF(586) |
| Lung Lesion / 공동성 Consolidation | 감염/면역 | GPA(900), EGPA(183), 폐결핵, 아스페르길루스종 |
| Edema (비심인성·반복) | 혈관/순환 | PVOD(31837), PCH(199241), 만성혈전색전성 폐고혈압 |
| Lung Opacity (미만성·만성) | 섬유화/간질성 | IPF(2032), NSIP, 과민성 폐렴, PAP(747) |
| Pleural Other | 종양성 + 직업성 | 악성 중피종(50251), 흉막 고립성 섬유종양(1335) |
| Cardiomegaly (젊은 환자) | 유전/대사 | HCM(99739), Fabry병, 아밀로이드증, Noonan증후군 |
| Fracture (병적 골절) | 유전/대사 | 골형성부전증(666), 다발골수종, 골전이 |
| Enlarged Cardiomediastinum | 구조적 + 종양성 | Castleman병(160), 림프종, 흉선종, 섬유성 종격동염 |
| Atelectasis (반복 중엽) | 구조/형태 | 원발성 섬모운동이상증(244), 낭성섬유증(586) |

---

> **면책 조항.** 본 시스템 Rare-Link AI는 진단 보조(Clinical Decision Support) 도구이며, MFDS 의료기기 허가를 취득하지 않은 연구·교육 목적의 시스템이다. 흉부 X-ray 판독과 최종 진단은 반드시 전문의에게 의뢰해야 하며, 모든 임상적 최종 결정 권한은 담당 의사에게 귀속된다.

> **보고서 정보.** 본 보고서는 AWS S3 `say2-2team-bucket`의 시스템 문서·코드와 CloudFront 배포본을 정본(authoritative source)으로 하고, 팀 Notion 워크스페이스의 설계·실험 기록을 보강 자료로 종합하여 작성되었다. 작성일 2026-05-22. SKKU AWS SAY 2기 2팀 · Rare-Link AI.
