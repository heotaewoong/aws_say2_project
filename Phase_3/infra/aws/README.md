# AWS 인프라 — Phase 3 / Phase 4 (팀 분담 산출물)

## 목적
v2 진단 파이프라인의 두 단계를 **AWS Lambda + Bedrock**로 독립 배포하여
각 팀이 병렬 개발하고, 추후 Step Functions로 통합한다.

## 진단 파이프라인 전체 흐름

```
환자 입력
  ↓
[Phase 1] 문진 / HPO 매칭          (다른 팀)
  ↓
[Phase 2] X-ray (CheXNet+GradCAM) (다른 팀)
  ↓
[Phase 3] 가중치 통합 채점         ← 본 폴더 phase3/
  ↓
[Phase 4] LLM 검증 + 재 ranking   ← 본 폴더 phase4/
  ↓
[Phase 5] 희귀질환 (LIRICAL+RAG)   (다른 팀, 독립 트랙)
  ↓
[Final]   최종 보고서 생성
```

## Phase 3 vs Phase 4 책임 분리 (feedback_phase_responsibility.md 정합)

| | Phase 3 | Phase 4 |
|---|---|---|
| 역할 | 가중치 + 임계값 (deterministic) | LLM 검증 + 재 ranking + 누락 alert |
| AWS 핵심 | Lambda + Layer (registry) | Lambda + Bedrock (Sonnet 4.6) |
| 외부 호출 | 없음 | bedrock-runtime:InvokeModel |
| Fallback | — | Guard Rail 실패 → Phase 3 ranking 그대로 |
| 비용 단위 | Lambda 실행시간 (≈$0.0000167/GB-s) | Lambda + Bedrock 토큰 |

## 폴더 구조

```
infra/aws/
├── README.md                  # 본 문서
├── MEDICAL_COMPLIANCE.md      # 규제·학술 검증 (FDA/MFDS/EU AI Act/문헌)
├── phase3/
│   ├── ARCHITECTURE.md        # 다이어그램 + 입출력 JSON 스펙
│   ├── template.yaml          # SAM (Lambda + Layer + API GW + IAM)
│   ├── lambda/
│   │   ├── handler.py         # event → DiagnosticScorer.score_all() → JSON
│   │   └── requirements.txt
│   ├── layer/
│   │   └── build_layer.sh     # data/*.yaml + *.xlsx + lung_dx 코드 → layer.zip
│   ├── deploy.sh
│   ├── invoke_local.sh
│   └── events/sample_event.json
└── phase4/
    ├── ARCHITECTURE.md
    ├── template.yaml          # Lambda + API GW + IAM (Bedrock 권한)
    ├── lambda/
    │   ├── handler.py         # event → Phase4Verifier.verify() → JSON
    │   └── requirements.txt
    ├── deploy.sh
    ├── invoke_local.sh
    └── events/sample_event.json
```

## 추후 통합 (현재 미구현, 산출물 모이면 진행)

```yaml
# infra/aws/integration/state-machine.asl.json (예정)
StartAt: Phase1
States:
  Phase1: { Type: Task, Resource: arn:phase1, Next: Phase2 }
  Phase2: { Type: Task, Resource: arn:phase2, Next: Phase3 }
  Phase3: { Type: Task, Resource: arn:phase3, Next: Phase4 }
  Phase4: { Type: Task, Resource: arn:phase4, Next: Phase5 }
  Phase5: { Type: Task, Resource: arn:phase5, Next: FinalReport }
```

## 사전 준비

### 공통
- AWS CLI v2 (`aws --version` ≥ 2.x)
- AWS SAM CLI (`sam --version` ≥ 1.100)
- Python 3.11 (Lambda 런타임과 일치)
- 배포할 AWS 계정 IAM (관리자 또는 Lambda/IAM/API GW/CloudFormation 권한)

### Phase 4 추가
- AWS Bedrock 모델 액세스 활성화 (콘솔에서 Anthropic Claude Sonnet 모델 enable 필요)
- Bedrock 지원 리전 (us-east-1 / us-west-2 / ap-northeast-1 등 — 모델별 차이)
- 의료 데이터(PHI) 처리 시 **AWS BAA** 사전 체결 (HIPAA-eligible 서비스 사용)

## 의료 진단보조 도구로서의 사전 점검
**MEDICAL_COMPLIANCE.md** 참조. 핵심 요약:
- FDA Clinical Decision Support 가이던스 (2022 final) 정합 설계 — Phase 4 출력은 citation + rationale 의무로 *clinician independent review* 가능
- FDA Good ML Practice (2021) 10원칙 매핑
- FDA Predetermined Change Control Plan (2024-12 final) — 모델 갱신 절차 사전 등록 권고
- WHO LMM ethics guidance (2024-01) 책임/감독/형평성 원칙
- EU AI Act (Reg 2024/1689) — 의료기기 AI = high-risk, 2026-08부터 일부 의무 발효
- HIPAA: Lambda/API GW/Bedrock 모두 eligible (BAA 필수)

## 라이선스 / 데이터 출처
- 질병 레지스트리 — `data/*.yaml`, `data/*.xlsx` (프로젝트 내부)
- HPO terms — Human Phenotype Ontology (CC-BY 4.0)
- ICD-10 — WHO (사용 라이선스 별도 확인)
- 가이드라인 인용 — fair use / educational citation (배포 시 출판사 라이선스 재확인 필요)
