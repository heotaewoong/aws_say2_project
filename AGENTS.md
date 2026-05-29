# AGENTS.md — Rare-Link AI 하네스 인덱스

> Claude Agent가 이 프로젝트에서 작업 전 반드시 먼저 읽는 파일.
> 상세 규칙은 CLAUDE.md. 코드 설명은 README.md 참고.

## 핵심 규칙 (위반 불가)
- AWS region: `ap-northeast-2` 고정
- 새 AWS 리소스 생성 시 태그 `pre-{서비스}-2-2-team` 필수
- 외부 API 호출 시 반드시 실패 → 빈값 반환 처리
- 의료 진단 출력에 disclaimer 필수 (SYSTEM_PROMPT §2)
- AWS 키 코드/문서 하드코딩 절대 금지

## 폴더 구조
```
aws_say2_project_vision/
├── rag_pipeline.py      # 메인 파이프라인 (5단계 확정 구조)
├── inference_engine.py  # 비전 모델 추론 (SooNet)
├── vision_engine.py     # X-ray 분석 엔진
├── knowledge_base.py    # 내부 의료 지식베이스 (DynamoDB)
├── reporter.py          # Bedrock 기반 소견서 생성
├── app.py               # FastAPI 진입점
├── rag/                 # 외부 API fetcher 패키지
│   ├── clinicaltrials_fetcher.py   # ClinicalTrials.gov
│   ├── monarch_fetcher.py          # Monarch Initiative (HPO 변환)
│   ├── orphanet_fetcher.py         # Orphanet (희귀질환 DB)
│   ├── pubcasefinder.py            # PubCaseFinder
│   ├── pubmed_fetcher.py           # PubMed 논문 검색
│   └── bedrock_extractor.py        # HPO 추출 (Bedrock)
├── sagemaker/           # SageMaker 학습 스크립트
├── frontend/            # React 프론트엔드
└── aws_architecture/    # 아키텍처 문서 및 다이어그램
```

## 반복 업무 → 스킬 매핑
| 업무 | 스킬 | 호출 |
|------|------|------|
| 새 RAG fetcher 추가 | rag-fetcher-add | `/rag-fetcher-add` |
| AWS 리소스 생성 | (CLAUDE.md 체크리스트 참고) | 수동 |
| 구현 후 문서화 | (README + Notion 업데이트) | 수동 |

## 에이전트 작업 전 체크
1. `rag/` 폴더 기존 fetcher 구조 확인 (clinicaltrials_fetcher.py 참고)
2. `rag_pipeline.py` import 섹션 및 RAG_PARALLEL_TIMEOUT 확인
3. `requirements.txt` 새 패키지 추가 여부 확인
