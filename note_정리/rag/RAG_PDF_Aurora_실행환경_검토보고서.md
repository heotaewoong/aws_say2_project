# RAG 최종 보고서 저장 방식 — 실행 환경 검토 보고서

> 작성자: 허태웅 | 작성일: 2026-05-13  
> 대상: rag_llm_3.py (S3: `say2-2team-bucket/RAG/rag_llm_3.py`)

---

## 1. 현황 요약

현재 `rag_llm_3.py`는 다음 순서로 최종 보고서를 저장합니다.

```
Bedrock Claude 추론
  → JSON 파싱
  → Markdown 변환 (json_to_markdown)
  → PDF 렌더링 (weasyprint)
  → S3 업로드 (say2-2team-bucket/final_reports/{session_id}/report.pdf)
  → Aurora INSERT (rarelinkai.final_report)
```

**Aurora 사용은 확정 사항입니다.** 현재 코드에서 PDF 생성 실패와 Aurora INSERT는 독립적으로 동작합니다 — PDF가 실패해도 Aurora에는 저장됩니다.

---

## 2. 문제: weasyprint와 실행 환경

### weasyprint란?

`weasyprint`는 Markdown → HTML → PDF로 변환하는 Python 라이브러리입니다.  
한국어 폰트 지원이 우수하고, CSS 기반 레이아웃으로 의료 보고서 품질의 PDF를 생성합니다.

### 왜 Lambda에서 안 되는가?

weasyprint는 **OS 수준의 시스템 라이브러리**(Pango, Cairo, GObject)가 필요합니다.  
AWS Lambda는 읽기 전용 파일시스템이므로 이 라이브러리를 설치할 수 없습니다.

| 라이브러리 | 역할 | Lambda 설치 가능 여부 |
|-----------|------|------------------|
| Pango | 텍스트 레이아웃 (한글 포함) | ❌ 불가 |
| Cairo | 벡터 그래픽 렌더링 | ❌ 불가 |
| GObject | 런타임 바인딩 | ❌ 불가 |

Lambda Layer로 패키징을 시도할 수 있으나:
- 압축 해제 후 250MB 용량 제한 초과
- 빌드 환경이 Lambda 런타임과 일치해야 하므로 유지보수 난이도 높음
- 실제 프로덕션 적용 사례 극소수 (비권장)

---

## 3. 실행 환경별 비교

| 항목 | Lambda | EC2 | 로컬 Mac |
|------|--------|-----|---------|
| weasyprint (PDF) | ❌ 시스템 라이브러리 없음 | ✅ yum/apt 설치 가능 | ✅ brew 설치 가능 |
| Aurora 연결 | ✅ VPC 내 접근 | ✅ VPC 내 접근 | ❌ VPC 외부 |
| Bedrock 호출 | ✅ IAM Role | ✅ IAM Role | ✅ IAM 자격증명 |
| S3 접근 | ✅ | ✅ | ✅ |
| 스케줄 자동 실행 | ✅ EventBridge 트리거 | ✅ cron / Step Functions | ❌ |
| 비용 | 요청 당 과금 (무료 티어 포함) | 시간 당 과금 (t3.small ≈ $0.02/h) | 없음 |
| 운영 관리 | 관리 불필요 | OS/보안 패치 필요 | 개인 장비 의존 |

---

## 4. 대안 분석

### 대안 A. EC2에서 실행 (권장)

**구성**: EC2 인스턴스(t3.small, VPC 내 배치) → 스크립트 직접 실행 또는 cron

**장점**
- weasyprint + Aurora 모두 정상 동작
- 현재 코드 수정 불필요
- 한국어 PDF 품질 우수
- 조장 슬랙 지침("Lambda 필요 없다")과 일치

**단점**
- EC2 상시 운영 시 비용 발생 (t3.small ≈ $15/월)
- 호출 시에만 실행 시 수동 실행 또는 EventBridge + SSM Run Command 구성 필요

**구현 난이도**: 낮음 (EC2 생성 + pip3 install + yum install 3~4줄)

---

### 대안 B. Lambda + PDF 제거, Aurora만 저장

**구성**: Lambda에서 실행, weasyprint 제거, markdown_report만 Aurora에 저장

**장점**
- Lambda 그대로 유지
- 별도 인프라 불필요
- Aurora 저장은 완전히 가능

**단점**
- PDF 파일 미생성 → 의사에게 PDF 형태 보고서 제공 불가
- 추후 PDF 필요 시 별도 변환 서비스 추가 필요

**코드 변경**: `save_final_report()`에서 `render_pdf_from_markdown()` 호출 제거, `s3_uri_pdf = None` 고정

**구현 난이도**: 낮음 (코드 10줄 수정)

---

### 대안 C. Lambda + reportlab (순수 Python PDF)

**구성**: weasyprint 대신 `reportlab` 라이브러리 사용 (시스템 라이브러리 불필요)

**장점**
- Lambda에서 동작
- Aurora 저장 가능
- PDF 파일 생성 가능

**단점**
- **한국어 지원 매우 취약** — 별도 TTF 폰트 파일 포함 필요
- 레이아웃 코드를 직접 작성해야 함 (CSS 기반 아님 → 개발 공수 높음)
- 의료 보고서 수준의 가독성 확보 어려움

**구현 난이도**: 높음 (폰트 설정 + 레이아웃 코드 전면 재작성)

---

### 대안 D. Lambda + HTML → S3 저장 (PDF 미생성)

**구성**: weasyprint 대신 HTML string을 S3에 저장, Aurora에는 `s3_uri_html` 필드 사용

**장점**
- Lambda에서 동작
- 브라우저로 직접 열람 가능
- CSS 스타일 그대로 유지 가능

**단점**
- PDF가 아닌 HTML 파일 → 출력/공유 불편
- 현재 Aurora 스키마에 `s3_uri_html` 컬럼이 이미 있음 (NULL→ 활용 가능)

**구현 난이도**: 낮음 (weasyprint 호출을 HTML string 저장으로 교체)

---

## 5. 권고사항

### 1순위: **대안 A (EC2 실행)**

> 현재 코드를 그대로 유지하면서, PDF + Aurora 모두 완전히 구현 가능한 유일한 방법입니다.  
> 조장의 지침("Lambda 필요 없다")과도 일치합니다.  
> 발표용이라면 EC2 t3.small 1대로 충분하며, 발표 후 종료하면 비용 0입니다.

### 2순위: **대안 B (PDF 제거, Aurora만)**

> 발표 시간이 촉박하거나 EC2 설정이 어려운 경우 선택.  
> Aurora 저장은 완벽하게 동작하며 코드 수정도 간단합니다.  
> 단, 의사에게 PDF 형태 보고서를 전달할 수 없습니다.

---

## 6. 현재 코드 동작 보증 사항

코드는 이미 **graceful degradation** 방식으로 설계되어 있습니다:

```
weasyprint 미설치 또는 실패
  → 경고 출력 후 계속 진행
  → Aurora INSERT: s3_uri_pdf = NULL로 저장 (정상 완료)
  → 로컬 JSON 백업도 항상 저장
```

즉, **어떤 환경에서 실행해도 Aurora 저장은 항상 실행됩니다.**

---

## 7. EC2 실행 시 세팅 절차 (대안 A 선택 시)

```bash
# 1. 패키지 설치 (Amazon Linux 2 기준)
sudo yum install -y pango cairo gobject-introspection
pip3 install psycopg2-binary markdown weasyprint aiohttp requests pandas boto3

# 2. S3에서 스크립트 다운로드
aws s3 cp s3://say2-2team-bucket/RAG/rag_llm_3.py .

# 3. 실행
python3 rag_llm_3.py
```

EC2 IAM Role에 필요한 권한:
- `bedrock:InvokeModel`
- `s3:PutObject` (say2-2team-bucket)
- `secretsmanager:GetSecretValue` (rare-link-ai/aurora/app-user)

---

*이 보고서는 2026-05-13 기준 rag_llm_3.py (S3 업로드 완료 버전) 분석 결과입니다.*
