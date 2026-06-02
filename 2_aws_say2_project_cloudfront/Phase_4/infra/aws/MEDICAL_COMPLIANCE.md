# 의료 AI 진단보조 도구 — 규제·문헌 검증

본 문서는 Phase 3 / Phase 4 AWS 아키텍처가 의료 진단보조 도구로서 갖춰야 할
규제·표준·임상 근거를 정리한다. **사견 배제, fact 기반, 검증된 reference만 인용**.
각 reference는 "영향 이유 + 실재성 자체검증" 메타 포함 (memory `feedback_reference_discipline.md`).

> **본 문서의 한계**: 본 검증은 *설계 단계 적합성*만 다룬다. 임상 배포는
> 추가 임상시험·전향적 validation·기관 IRB·국가별 인허가 절차가 필수.

---

## 1. 규제 분류 — 본 도구는 어떤 의료기기인가

### 1.1 FDA 기준 (미국)

**Software as a Medical Device (SaMD)** vs **non-device CDS** 판단:

FDA *Clinical Decision Support Software* 최종 가이던스 (2022-09-28) — Section
III.B에 따라 21st Century Cures Act §520(o)(1)(E) 4가지 기준을 모두 충족하면
*non-device CDS* (FDA 규제 면제):

1. 의료영상·신호·패턴을 *직접 처리* 하지 않을 것
2. 환자 *의료정보를 표시·분석·인쇄* 하는 것
3. 의료 전문가에게 *권고를 제공* 할 것
4. 의료 전문가가 권고의 *근거를 독립적으로 검토* 가능하게 할 것

**본 도구 매핑**:
| 기준 | Phase 3 | Phase 4 | 판단 |
|---|---|---|---|
| #1 (영상 직접 처리 X) | △ Phase 1 결과(파생 라벨)만 입력 | ○ 영상 미처리 | 경계선. 파이프라인 전체로 보면 Phase 1이 #1 위반 가능 → 전체 시스템은 SaMD 가능성 |
| #2 (정보 표시·분석) | ○ | ○ | OK |
| #3 (HCP 권고) | ○ ranking 제공 | ○ 재 ranking + alert | OK |
| #4 (근거 독립 검토) | ○ evidence 필드 (matched HPO/lab/findings) | ○ citation + rationale 의무 | OK — Phase 4의 citation 강제가 #4 만족 핵심 |

**결론**: Phase 3+4 단독은 #4를 강하게 충족하나, 전체 파이프라인(Phase 1 영상)
포함 시 SaMD 분류 가능. **510(k) 또는 De Novo pathway 검토 권고**.

### 1.2 IMDRF SaMD 위험 분류

IMDRF *Software as a Medical Device (SaMD): Possible Framework for Risk
Categorization* (N12FINAL:2014, 현재도 글로벌 reference framework):

| 의료 상황 | 정보 사용 목적 | 카테고리 |
|---|---|---|
| Critical (생명 위협, 심각 손상) | drives clinical management | IV |
| Critical | informs management | III |
| Serious (심각 질환) | drives | III |
| Serious | informs | II |
| Non-serious | drives/informs | II/I |

**본 도구**: 폐렴·ARDS·VAP·결핵 등 다수 *serious* 카테고리 포함. ranking 결과가
임상의 *informs*에 해당 → **Class II SaMD** 가장 가능성 높음. ARDS/VAP 같은
critical 케이스를 drives로 해석하면 Class III 가능.

### 1.3 한국 MFDS

식약처 *의료기기법*에 따른 의료기기 SW 분류 — 임상 의사결정 보조 SW는
2급 또는 3급. 식약처 인공지능 의료기기 가이드라인 시리즈 (2017-2022 다수
발간) 검토 필요. **본 시점 정확한 가이드라인 버전 확인은 사용자 직접 확인
권고** (제목·일자 상세 검증 미완).

### 1.4 EU AI Act (2024-08-01 발효)

*Regulation (EU) 2024/1689* — 기계 안전성과 관련된 AI 또는 Annex III 의료기기에
포함된 AI 시스템은 **High-Risk AI** 분류 (Article 6).

본 도구는 의료기기 SW로서 high-risk 해당. 다음 의무 적용 (단계 발효):
- Risk management system (Art.9)
- Data and data governance (Art.10)
- Technical documentation (Art.11)
- Record-keeping / logging (Art.12)
- Transparency (Art.13)
- Human oversight (Art.14)
- Accuracy / robustness / cybersecurity (Art.15)

**적용 시점**: General-purpose AI 의무는 2026-08-02부터, high-risk 의무 대부분은 2027-08-02부터.

---

## 2. 본 아키텍처의 안전 설계 매핑

### 2.1 FDA *Good Machine Learning Practice (GMLP) Guiding Principles* — 2021-10
공동 발행: US FDA + Health Canada + MHRA UK. 10원칙.

| 원칙 | 본 아키텍처 매핑 |
|---|---|
| 1. Multi-disciplinary expertise | 가중치는 가이드라인+학술지(memory `project_2026-04-17_v9_weight_audit`)에 근거. 폐 전문의 검토 필요. |
| 2. Good SW + security practices | SAM IaC, IAM least-privilege, KMS 암호화 권고 (template.yaml) |
| 3. Clinical study participants representative | **별건 — 임상 validation 필요** |
| 4. Train/test data independent | **별건 — MIMIC-IV ETL은 별도 트랙** |
| 5. Selected reference dataset based on best methods | lab v9.5 (memory `project_2026-04-30_lab_v9_5_*`) |
| 6. Model design tailored to data and clinical use | 4축 가중치 + LLM verifier 분리 (Phase 3+4) |
| 7. Focus on performance of human-AI team | Phase 4 citation + rationale → clinician review 가능 |
| 8. Testing demonstrates device performance | **별건 — clinical evaluation study** |
| 9. Users provided clear, essential information | API 출력에 evidence·citation·confidence 포함 |
| 10. Deployed models monitored for performance | CloudWatch + X-Ray + alarm (template.yaml) |

**검증 상태**: 출처 실재 (FDA 공식 발행물). PDF 검색 가능 — fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles.

### 2.2 FDA *PCCP for AI-Enabled Device Software Functions* — 최종 2024-12-04

AI/ML 모델 갱신을 사전 승인하는 절차 (Predetermined Change Control Plan).
**본 도구의 가중치 갱신 / Phase 4 prompt 갱신 / 모델 ID 변경에 직접 적용 가능**.

핵심 요소:
- Description of Modifications
- Modification Protocol
- Impact Assessment

→ 본 프로젝트의 `data/lung_disease_v6_to_v7_변경대비표.xlsx` + auto-sync hook
(memory `project_2026-05-07_yaml_excel_autosync.md`) 가 Modification Protocol의
audit trail 일부로 활용 가능.

**검증 상태**: 출처 실재. fda.gov/regulatory-information/search-fda-guidance-documents.

### 2.3 WHO *Ethics and governance of AI for health: Guidance on large multi-modal models* — 2024-01

LLM 기반 임상 도구에 직접 해당. 6대 원칙 + 거버넌스 권고.
Phase 4의 citation 강제·Guard Rail·fallback이 다음 원칙과 정합:
- Protect autonomy (HCP 최종 판단)
- Promote safety (Guard Rail + fallback)
- Ensure transparency (raw_llm_response audit trail)
- Foster responsibility / accountability
- Ensure inclusiveness / equity (별건 검증)
- Promote sustainability (별건)

**검증 상태**: WHO 공식 발행물. iris.who.int 검색 가능.

### 2.4 NIST *AI Risk Management Framework (AI RMF 1.0)* — 2023-01

GOVERN / MAP / MEASURE / MANAGE 사이클. EU AI Act + FDA 가이던스와 호환.
본 아키텍처의 CloudWatch 알람·X-Ray·Guard Rail 보고서가 MEASURE/MANAGE 단계에 매핑.

**검증 상태**: NIST 공식. nist.gov/itl/ai-risk-management-framework.

---

## 3. AWS 서비스 의료 적합성

### 3.1 HIPAA 적격성

| 서비스 | HIPAA-eligible | 비고 |
|---|---|---|
| AWS Lambda | ○ | 2017부터 |
| API Gateway | ○ | |
| Amazon Bedrock | ○ | 2024 추가 (AWS 공식 발표 기준) |
| CloudWatch Logs | ○ | |
| KMS | ○ | |
| Step Functions | ○ | |

**필수 조건**: AWS와 BAA (Business Associate Addendum) 사전 체결.
**검증 방법**: aws.amazon.com/compliance/hipaa-eligible-services-reference 에서
최신 목록 확인 (분기마다 업데이트).

### 3.2 Bedrock 데이터 처리 정책

AWS Bedrock FAQ (공식): "Amazon Bedrock에 입력한 데이터·이전 모델 출력은
모델 학습에 사용되지 않으며, 제3자와 공유되지 않음."
(Anthropic 모델 포함 — Bedrock은 모델 제공자 격리 유지).

**의미**: PHI 입력 시 Anthropic 학습에 누출되지 않음. 단 BAA + HIPAA-eligible 모드로 사용 시 한정.

### 3.3 한국 PHIA (개인정보 보호법)

한국 환자 데이터를 사용 시 *개인정보 보호법* + *의료법* 적용.
가명처리/익명화 + 데이터 처리 동의 + 국외이전 별도 동의 필수.
**Bedrock 모델이 한국 외 리전(us-east-1)이면 국외이전 해당** → 환자 동의 필수.

---

## 4. Phase 4 LLM의 임상 안전 설계

### 4.1 Hallucination 방어층 (다층)

| 층 | 메커니즘 |
|---|---|
| Prompt | AUTHORITATIVE_SOURCES 50+ 권위 출처만 인용 강제 (system prompt) |
| Sampling | temperature=0.0 (deterministic) |
| Schema | JSON 응답 스키마 강제, parse 실패 시 fallback |
| Guard Rail #1 | hp_id_validation — 입력에 없는 HPO 인용 차단 |
| Guard Rail #2 | icd_mapping_validation — citation의 ICD가 disease와 매칭 |
| Guard Rail #3 | citation_required — 모든 ranking·alert에 citation 1+ |
| Guard Rail #4 | confidence_threshold — overall_confidence < 0.5면 reject |
| Guard Rail #5 | hallucination_keyword — "may", "possibly" 등 비확정어 빈도 검사 |
| Guard Rail #6 | schema_validation — Phase4Result schema 적합성 |
| Fallback | Guard Rail 한 개라도 실패 → Phase 3 ranking 그대로 반환 |

이 설계는 **safety-first ML failure mode** (FDA GMLP 원칙 #6 + #7)에 정합.

### 4.2 Citation의 권위성

`AUTHORITATIVE_SOURCES`는 다음 카테고리 (lung_dx/phase4_llm_verify/prompt_builder.py):
- 학회 가이드라인 (ATS, IDSA, ESC, ERS, GOLD, GINA, KSR 등)
- 표준 교과서 (Harrison's, Mandell, Murray-Nadel)
- PubMed indexed 논문 (PMID 부착)

memory `project_2026-04-29_v3_2_build.md`: "19권위출처 PubMed 검증·2건 PMID
정정·2건 fabricated 제거" — 이미 fabrication 제거 사이클 1회 완료.

### 4.3 임상 의사 최종 판단 보장

본 도구는 **diagnostic decision-making을 자동화하지 않음**.
- ranking 제공 (probability, not classification)
- evidence + citation 노출
- 임상의가 거부·수정·추가 검사 결정

→ FDA Cures Act §520(o)(1)(E) 기준 #3, #4 충족.

---

## 5. 최근 (2024-2026) 학술지·가이드라인 검토 권고

본 섹션은 **사용자 직접 검색 권고 항목**. 메모리 `feedback_reference_discipline.md`
정합 — 미확인 PMID는 인용 금지 원칙으로 본 문서에는 구체 PMID 미기재.

### 5.1 검색 권고 키워드 (PubMed)

| 주제 | 권장 검색식 |
|---|---|
| LLM 임상 진단 | `("large language model" OR "LLM") AND ("clinical decision support" OR "diagnostic accuracy") AND 2024:2026[dp]` |
| AI hallucination 의료 | `"hallucination" AND ("medical" OR "clinical") AND ("AI" OR "language model") AND 2024:2026[dp]` |
| Pneumonia AI 진단 | `("artificial intelligence" OR "deep learning") AND pneumonia AND diagnosis AND 2024:2026[dp]` |
| LLM evaluation framework | `("evaluation" OR "benchmark") AND "large language model" AND clinical AND 2024:2026[dp]` |

### 5.2 최신 가이드라인 갱신 모니터링 (본 도구 관련)

| 가이드라인 | 최근 update | 본 도구에 영향 |
|---|---|---|
| ATS/IDSA Community-Acquired Pneumonia | 2019 (Metlay et al. AJRCCM, PMID 31573350) — **2025/26 update 발표 여부 확인 필요** | Phase 4 prompt AUTHORITATIVE_SOURCES |
| GOLD Report (COPD) | 매년 갱신, 2026 버전 발행 시 prompt update 필요 | 동상 |
| GINA (천식) | 매년 갱신 | 동상 |
| Berlin ARDS Definition | 2012 + 2023 update (ATS+ESICM+SCCM) — 본 코드 참조 확인 필요 | 가중치 critical care 부분 |
| ECIL PJP guideline | 2016, 차기 update 모니터링 | Phase 4 missed alert |
| WHO TB guideline | 2022 → 후속 공지 모니터링 | 결핵 sub-code 21건 |
| ESC/ERS PH guideline | 2022 (PMID 36017548) | 폐고혈압 |

→ 본 가이드라인 versioning을 *registry YAML reference 필드*에 명시 + PCCP의
"Modification Protocol" 일부로 정착 권고.

### 5.3 최근 LLM-CDS 안전성 이슈 (이론적 known-issue)

다음은 2023-2025 학계에서 반복 보고된 LLM 임상 응용 이슈. 본 도구 설계에서
대응 방법 명시:

| 이슈 | 본 도구 대응 |
|---|---|
| Citation fabrication | Guard Rail #1 #2 #3 + AUTHORITATIVE_SOURCES whitelist |
| Confidence miscalibration | Guard Rail #4 + temperature=0.0 |
| Drug name confusion | 별건 — 본 도구는 진단 ranking 전용 (처방 추천 X) |
| Demographic bias | **별건 — fairness audit 필요** (race/sex/age subgroup 성능 측정) |
| Prompt injection | API GW WAF + 입력 sanitization (별건) |
| Adversarial patient input | Phase 1/2 영상 단계에서 별도 점검 필요 (별건) |

---

## 6. 배포 전 필수 확인 체크리스트

### 6.1 기술
- [ ] Lambda Layer 250MB 한도 확인
- [ ] Bedrock 모델 액세스 활성화 (해당 리전)
- [ ] AWS BAA 체결
- [ ] CloudWatch 로그 PHI redaction 확인
- [ ] X-Ray 활성화 + segment naming
- [ ] WAF rate limit + geo block (필요 시)
- [ ] VPC endpoint (PHI 운영 시)

### 6.2 임상 / 데이터
- [ ] Disease registry 가중치 폐 전문의 review 통과
- [ ] AUTHORITATIVE_SOURCES PMID 100% 검증 (지속 사이클)
- [ ] 가이드라인 최신 버전 반영 (5.2 표)
- [ ] MIMIC-IV (또는 동등) 검증 데이터셋 sensitivity/specificity 측정
- [ ] Subgroup 공정성 측정 (sex/age/race)

### 6.3 규제
- [ ] FDA pathway 결정 (510k / De Novo / non-device CDS) — 법무팀 자문
- [ ] MFDS 의료기기 분류 결정 + 시판 전 절차
- [ ] EU 시판 시 CE marking + MDR + AI Act 정합
- [ ] HIPAA / PHIA / GDPR 정합 검토
- [ ] 임상시험 계획서 + IRB 승인

### 6.4 운영
- [ ] PCCP 작성 (모델·가중치 갱신 사전 등록)
- [ ] Incident response plan (오진단 보고)
- [ ] User training material (HCP 대상)
- [ ] 라벨링 / IFU (Instructions for Use) 작성
- [ ] Post-market surveillance plan

---

## 7. 검증된 reference 목록 (실재성 자체검증 완료)

| # | Reference | 발행기관 | 연도 | 검증 |
|---|---|---|---|---|
| R1 | Good Machine Learning Practice for Medical Device Development: Guiding Principles | FDA + Health Canada + MHRA | 2021-10 | ○ FDA 공식 게시 (URL 확인 가능) |
| R2 | Clinical Decision Support Software (Final Guidance) | FDA | 2022-09-28 | ○ FDA 공식 |
| R3 | Marketing Submission Recommendations for a PCCP for AI-Enabled Device Software Functions (Final) | FDA | 2024-12-04 | ○ FDA 공식 (최근) |
| R4 | Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket Submissions (Final) | FDA | 2023-09 | ○ FDA 공식 |
| R5 | Ethics and governance of AI for health: Guidance on large multi-modal models | WHO | 2024-01-18 | ○ WHO iris 공식 |
| R6 | Regulation (EU) 2024/1689 — AI Act | EU | 2024-07 | ○ EUR-Lex 공식 |
| R7 | SaMD: Possible Framework for Risk Categorization (N12FINAL:2014) | IMDRF | 2014 | ○ IMDRF 공식 (현행) |
| R8 | AI Risk Management Framework (AI RMF 1.0) | NIST | 2023-01 | ○ NIST 공식 |
| R9 | ATS/IDSA Community-Acquired Pneumonia Guidelines | Metlay JP et al. AJRCCM 2019 | 2019 | ○ PMID 31573350 (lung_dx 코드 인용) |

**불확실 / 사용자 검증 권고**:
- 한국 MFDS 인공지능 의료기기 가이드라인 — 정확한 최신 버전 사용자 확인 필요
- 2025-2026 LLM-CDS 시스템적 리뷰 논문 — 5.1 검색식으로 사용자 직접 검색 권고
- ATS/IDSA CAP 2025/26 update — IDSA 공식 사이트 모니터링

**원칙**: 본 문서는 *발행되었음을 직접 확인 가능한* 가이드라인만 인용. 미확인
PMID는 인용 거부 (memory `feedback_reference_discipline.md` "실재성 자체검증" 정합).

---

## 8. 결론

**본 AWS 아키텍처는 의료 진단보조 도구로서의 *기술적 설계 적합성*은 확보됨**:
- HIPAA-eligible 서비스 사용 (BAA 전제)
- FDA GMLP 10원칙 + WHO LMM 6원칙 정합
- Guard Rail 6종 + Fallback으로 LLM hallucination 다층 방어
- API에 evidence + citation 노출로 clinician 독립 검토 보장
- IaC + 알람 + X-Ray로 post-market monitoring 기반

**그러나 임상 배포는 다음 별도 트랙 *필수***:
1. 임상 validation study (subgroup fairness 포함)
2. 국가별 인허가 절차
3. 가이드라인 버전 동기 (지속)
4. PCCP 작성 + 모델 갱신 audit trail
5. 폐 전문의 + 임상 약사 + 임상 정보학 + 법무 다학제 검토

**현 시점 상태**: 개발/검증용 인프라. 환자 진료 직접 사용 금지 (라벨링 명시 필요).
