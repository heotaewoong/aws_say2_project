# CLAUDE.md — Claude 행동 규칙 & 반복 패턴

> Claude Code가 세션마다 자동으로 읽는 파일.
> 프로젝트 문서는 README.md 참고. 여기는 Claude가 반드시 기억할 규칙만 담음.

---

## 프로젝트 개요

**폐질환 진단 보조 AI 시스템** — 흉부 X-ray + 혈액검사 + 임상소견 + 유전체 데이터 종합 진단

- 비전 모델 학습: `aws_say2_project_vision/`

---

## AWS 설정 (항상 이 값 사용)

```bash
# 환경변수로 설정 (코드에 하드코딩 금지)
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_DEFAULT_REGION="ap-northeast-2"
```

코드에서 region 지정 시 항상 `ap-northeast-2` 사용.
AWS 키는 반드시 환경변수로만 설정하고, 코드나 문서에 직접 기입 금지.

---

## S3 버킷

```
say2-2team-bucket (ap-northeast-2)
├── cheXpert_data/valid_only/     # CheXpert validation 이미지
├── data/mimic-cxr-448/           # MIMIC-CXR 448x448 전처리 이미지
├── models/mimic-only/            # SageMaker 학습 결과 가중치
└── csv/                          # mimic-cxr-2.0.0-chexpert.csv

say1-pre-project-5 (ap-northeast-2)
└── data/mimic-cxr-jpg/files/     # MIMIC-CXR 원본 이미지
```

---

## AWS 리소스 태그 규칙 — 반드시 적용

모든 AWS 리소스 생성 시 태그 필수. 팀별 크레딧 판별 기준이므로 누락 금지.

```
태그 형식: pre-{서비스명}-{기수}-{팀번호}-team
예시: pre-sagemaker-2-2-team
```

```python
Tags=[{"Key": "project", "Value": "pre-{서비스}-2-2-team"}]
```

---

## 구현 전 체크리스트

- [ ] 새 AWS 리소스 생성 → 태그 포함했는가
- [ ] region → `ap-northeast-2` 맞는가
- [ ] 외부 API 호출 → 실패 시 빈값 반환 처리됐는가
- [ ] 새 패키지 필요 → `requirements.txt`에 추가했는가
- [ ] `paths.py` 수정 시 → `MODEL_DIR` 정의 순서 확인 (BEST_THRESHOLDS_JSON보다 먼저)

## 구현 후 체크리스트

- [ ] `README.md` 업데이트했는가
- [ ] 변경사항 Notion에 업데이트 및 위치 구조화 잘해서 반영했는가
