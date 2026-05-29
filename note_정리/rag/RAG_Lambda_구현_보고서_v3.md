# Rare-Link AI — RAG & Lambda 완전 구현 보고서 v3

> 작성일: 2026-05-15
> 대상 독자: RAG·클라우드·AI 비전공 팀원 (완전 초보자용)
> 코드 기준: `rag_llm_3.py` (1,201줄) + `handler.py` (206줄) — 2026-05-15 배포 완료본
> 배포 완료: `phase5-rag-dev` 스택, `ap-northeast-2` (서울 리전)

---

## 목차

1. [RAG가 뭔가요? — 30초 이해](#1-rag가-뭔가요--30초-이해)
2. [Lambda가 뭔가요? — 30초 이해](#2-lambda가-뭔가요--30초-이해)
3. [우리 프로젝트 전체 그림 — 5 Phase](#3-우리-프로젝트-전체-그림--5-phase)
4. [Phase 5 (RAG) 가 하는 일](#4-phase-5-rag-가-하는-일)
5. [rag_llm_3.py 전체 구조](#5-rag_llm_3py-전체-구조)
6. [외부 API 5개 — 어떤 데이터를 가져오나](#6-외부-api-5개--어떤-데이터를-가져오나)
7. [Aurora DB — 어떻게 연결되나](#7-aurora-db--어떻게-연결되나)
8. [캐시 시스템 — 왜 필요하고 어떻게 동작하나](#8-캐시-시스템--왜-필요하고-어떻게-동작하나)
9. [Bedrock AI — 어떻게 소견서를 만드나](#9-bedrock-ai--어떻게-소견서를-만드나)
10. [Lambda handler.py — 요청 처리 진입점](#10-lambda-handlerpy--요청-처리-진입점)
11. [SAM 배포 — Lambda를 AWS에 올리는 방법](#11-sam-배포--lambda를-aws에-올리는-방법)
12. [VPC 네트워크 구조 — 왜 복잡한가](#12-vpc-네트워크-구조--왜-복잡한가)
13. [배포된 실제 AWS 리소스 목록](#13-배포된-실제-aws-리소스-목록)
14. [실패 대비 설계 — 뭔가 터져도 괜찮은 이유](#14-실패-대비-설계--뭔가-터져도-괜찮은-이유)
15. [이번 세션에서 구현/수정한 것 전체 목록](#15-이번-세션에서-구현수정한-것-전체-목록)
16. [테스트 방법](#16-테스트-방법)
17. [용어집](#17-용어집)

---

## 1. RAG가 뭔가요? — 30초 이해

**RAG = Retrieval-Augmented Generation**
"검색해서 찾은 정보를 AI에게 먹여서 더 나은 답변을 만드는 방식"

일반 AI(GPT 등)의 문제점:
- 2024년에 학습 종료 → 최신 논문·임상시험 정보를 모름
- 희귀질환처럼 희소한 데이터는 학습 자체가 부족

RAG를 쓰면:
1. 실시간으로 PubMed, ClinicalTrials.gov 등에서 **관련 정보를 검색**
2. 그 정보를 AI 프롬프트에 **통째로 붙여서** 질문
3. AI가 외부 데이터에 근거해서 답변 → "오픈북 시험" 방식

```
일반 AI: "이 병이 뭔지 알아?" → AI 기억 속에서만 답변
RAG  AI: 먼저 최신 논문 검색 → "이 논문들을 보고 대답해줘" → 더 정확한 답변
```

---

## 2. Lambda가 뭔가요? — 30초 이해

**AWS Lambda = 서버 없이 코드를 실행하는 서비스**

보통 코드를 실행하려면:
- 서버 컴퓨터를 켜두고 → 코드를 설치하고 → 항상 돌아가게 유지

Lambda는:
- 코드만 AWS에 올려두면 됨
- 요청이 올 때만 실행 (평소엔 꺼져 있음)
- 실행 시간만 요금 부과 (사용한 만큼만)

```
기존 서버 방식:  [서버 24시간 켜둠] → 요청 들어오면 처리
Lambda 방식:   요청 들어오면 → [Lambda 즉시 켜짐] → 처리 → [꺼짐]
```

우리 프로젝트에서 Lambda가 RAG 파이프라인 전체를 실행합니다.
요청이 없을 땐 비용이 0원. 요청이 오면 최대 5분(300초) 동안 실행 가능.

---

## 3. 우리 프로젝트 전체 그림 — 5 Phase

```
[환자 데이터]
  ├─ 흉부 X-ray 이미지
  ├─ 증상 텍스트
  ├─ 혈액검사 수치
  └─ 기본 정보 (나이/성별)
         │
         ▼
┌─────────────────────────────────────────┐
│  Phase 1: 증상 텍스트 → HPO 코드 변환   │ (Bedrock Haiku AI)
│  Phase 2: X-ray 이미지 → HPO 코드 변환  │ (SooNet DenseNet-121 모델)
│  Phase 3: 혈액검사 수치 → HPO 코드 변환 │ (Rule-based 임계값)
└─────────────────────────────────────────┘
         │ HPO 코드 목록
         ▼
┌─────────────────────────────────────────┐
│  Phase 4: 질환 스코어링                  │
│  - 일반 폐질환 Top 10 (가중치 방식)      │
│  - 희귀질환 LR 랭킹 (LIRICAL 방식)      │
│  → Top 3 후보 질환 확정                 │
└─────────────────────────────────────────┘
         │ Top 3 질환
         ▼
┌─────────────────────────────────────────┐ ← 이 부분이 Phase 5
│  Phase 5: RAG + AI 소견서 생성          │ ← Lambda가 실행하는 구간
│  (rag_llm_3.py → handler.py)           │
│                                         │
│  ① DB에서 Phase 4 결과 읽기             │
│  ② 5개 외부 API 병렬 호출 (근거 수집)   │
│  ③ Bedrock Claude 3.5 Sonnet 호출       │
│  ④ AI 소견서 JSON → Aurora DB 저장      │
└─────────────────────────────────────────┘
         │
         ▼
  JSON 진단 보조 소견서
  → Aurora DB final_report 테이블 저장
  → Frontend에서 조회해서 의사에게 표시
```

**핵심**: Phase 1~4는 다른 팀원들이 구현. Phase 5(RAG)는 허태웅 담당.
모든 Phase는 `session_id`라는 UUID(고유번호)로 연결됩니다.

---

## 4. Phase 5 (RAG) 가 하는 일

Phase 4가 "어떤 질환이 의심되는가?" 를 정하면,
Phase 5는 "그 질환에 대한 최신 의학 근거를 수집해서 AI에게 최종 소견서를 쓰게" 합니다.

### Phase 5 내부 순서 (run_with_session_id 메서드)

```
[시작] Lambda가 session_id 받음
  │
  ▼
Step 0: DB에 "지금 Phase 5 실행 중" 기록
  (diagnosis_session 테이블 → status='running', current_phase=5)
  │
  ▼
Step 1: DB에서 Phase 4 결과 읽기
  (pos_hpos: 양성 HPO 목록, local_top_3: Top 3 후보 질환)
  │
  ▼
Step 2: 캐시 확인
  (이전에 같은 질환 데이터를 가져온 적 있으면 DB 캐시에서 꺼냄 → API 재호출 안 함)
  │
  ▼
Step 3: 5개 외부 API 병렬 호출 (RAG 핵심)
  ├─ PubMed: 최신 케이스리포트 논문
  ├─ PubCaseFinder: HPO 기반 희귀질환 후보
  ├─ Monarch Initiative: 인과 유전자 정보
  ├─ ClinicalTrials.gov: 현재 모집 중 임상시험
  └─ Orphanet (로컬): 희귀질환 역학/유전 정보
  │
  ▼
Step 4: Bedrock Claude 3.5 Sonnet 호출
  (모든 데이터를 AI에게 전달 → JSON 소견서 생성)
  │
  ▼
Step 5: DB에 결과 저장
  ├─ final_report 테이블: AI 소견서 JSON
  ├─ rag_api_cache 테이블: API 결과 캐시 (다음 번 재사용)
  └─ diagnosis_session 테이블: status='completed'
  │
  ▼
[완료] Lambda가 {"session_id": ..., "status": "completed"} 반환
```

---

## 5. rag_llm_3.py 전체 구조

이 파일은 1,201줄로 Phase 5 로직 전체를 담습니다.
S3에 저장됨: `s3://say2-2team-bucket/RAG/rag_llm_3.py`

### 클래스 구조

```python
class RareLinkHybridDualRAG:
    """Phase 5 RAG 파이프라인 전체를 담당하는 클래스."""

    # ── 초기화 ────────────────────────────────────────────
    def __init__(self, orphadata_csv_path=None)
        # HPO JSON을 S3에서 읽어서 메모리에 올림
        # Bedrock 클라이언트 초기화

    # ── DB 연결 (모듈 레벨 함수) ──────────────────────────
    _get_db_conn()                   # Secrets Manager → Aurora 접속

    # ── DB 읽기/쓰기 ──────────────────────────────────────
    _set_session_running()           # Phase 5 시작 기록
    _read_phase4_from_db()           # Phase 4 결과 (HPO + Top3) 읽기
    _read_api_cache()                # 이전 API 결과 캐시 읽기
    _save_to_db()                    # 최종 소견서 + 캐시 저장
    _mark_session_failed()           # 실패 시 status='failed' 기록

    # ── 캐시 키 생성 ──────────────────────────────────────
    _make_cache_key()                # "pubmed:lymphangioleiomyomatosis" 형태

    # ── 외부 API 호출 (모두 async) ────────────────────────
    fetch_pubmed_cases()             # PubMed 케이스리포트
    fetch_pubmed_guidelines()        # PubMed 가이드라인
    fetch_monarch()                  # Monarch 유전자 정보
    fetch_clinicaltrials()           # ClinicalTrials.gov
    fetch_pcf_disease_data()         # PubCaseFinder

    # ── RAG 데이터 조합 ────────────────────────────────────
    gather_rag_data()                # Top 3 질환 × 5개 API 병렬 호출

    # ── Bedrock AI 호출 ────────────────────────────────────
    call_bedrock()                   # Claude 3.5 Sonnet 호출

    # ── 메인 실행 ──────────────────────────────────────────
    run_with_session_id()            # Lambda가 호출하는 진입점
```

### 핵심 메서드별 설명

#### `__init__` — 초기화

```python
def __init__(self, orphadata_csv_path=None):
```

Lambda가 처음 실행될 때(Cold Start) 한 번만 호출됩니다.
- `hpo_official.json` 파일을 S3에서 읽어 메모리에 올림 (HPO 코드 ↔ 영어 이름 매핑)
- Bedrock 클라이언트 생성 (`boto3.client("bedrock-runtime", ...)`)
- 이후 요청은 같은 인스턴스를 재사용 (초기화 비용 절감)

#### `run_with_session_id` — 메인 실행

```python
async def run_with_session_id(self, session_id: str) -> dict:
```

Lambda의 `handler.py`가 호출하는 진입점입니다.
`session_id`(UUID) 하나만 받아서 5단계를 순서대로 실행합니다.

#### `gather_rag_data` — 병렬 API 호출

```python
async def gather_rag_data(self, pos_hpos, local_top_3, api_cache=None) -> list:
```

Top 3 질환 각각에 대해 5개 API를 **동시에** 호출합니다.
캐시에 있으면 API 호출 건너뜀. 없으면 실시간 호출.

---

## 6. 외부 API 5개 — 어떤 데이터를 가져오나

Phase 5에서 가장 핵심적인 부분입니다. 인터넷에서 최신 의학 정보를 긁어옵니다.

### API 호출 구조

```
Top 3 후보 질환 (예: LAM, IPF, 폐렴)
          │
          ▼
[각 질환마다 아래 5개 API를 병렬 호출]
          │
    ┌─────┼─────┬─────┬─────┐
    ▼     ▼     ▼     ▼     ▼
  PubMed Pubmed  PCF Monarch  CT
  Cases  Guide  기관 유전자  임상시험
```

### [1] PubMed — 케이스리포트 수집

- **운영**: NIH (미국 국립의학도서관)
- **가져오는 것**: 해당 질환의 최신 케이스리포트(실제 환자 사례) 논문 초록
- **검색 방식**: `"질환명"[Title/Abstract] AND case reports[Title/Abstract]`
- **최대 3편** 수집
- **반환 예시**:
  ```
  PMID:38765432 | A case of LAM presenting with... (2024)
  "Background: LAM is a rare disease..."
  ```
- **코드 위치**: `fetch_pubmed_cases()` 메서드
- **왜 필요한가**: AI가 "이 질환에서 비슷한 사례가 있었다"를 근거로 소견서 작성

### [2] PubMed Guidelines — 치료 가이드라인 수집

- **가져오는 것**: 해당 질환의 치료 가이드라인 논문
- **검색 방식**: `"질환명"[Title/Abstract] AND practice guidelines[Publication Type]`
- **코드 위치**: `fetch_pubmed_guidelines()` 메서드
- **왜 필요한가**: AI가 "가이드라인에서 이렇게 치료하라고 한다"를 소견서에 포함

### [3] PubCaseFinder — HPO 기반 희귀질환 매칭

- **운영**: DBCLS (일본 생명과학통합DB센터)
- **엔드포인트**: `https://pubcasefinder.dbcls.jp/api/get_diseases`
- **가져오는 것**: 환자의 HPO 조합과 가장 유사한 희귀질환 목록 + 매칭 점수
- **입력**: HPO 코드 목록 (예: `HP:0002094,HP:0002202`)
- **반환 예시**:
  ```json
  {"disease_id": "OMIM:617300", "score": 0.95, "matched_hpo_id": "HP:0002094,..."}
  ```
- **코드 위치**: `fetch_pcf_disease_data()` 메서드
- **왜 필요한가**: "이 HPO 조합은 과거 케이스리포트에서 어떤 희귀질환과 함께 나왔나" 확인

### [4] Monarch Initiative — 인과 유전자 확인

- **운영**: Monarch Initiative Consortium (국제 연구 컨소시엄)
- **엔드포인트**: `https://api.monarchinitiative.org/v3/api/entity`
- **가져오는 것**: 해당 질환의 원인 유전자 목록 (예: TSC1, TSC2)
- **반환 예시**:
  ```
  Lymphangioleiomyomatosis → 인과 유전자: TSC1, TSC2
  ```
- **코드 위치**: `fetch_monarch()` 메서드
- **왜 필요한가**: "이 질환이 유전성이라면 어떤 유전자 검사를 권고해야 하는가"

### [5] ClinicalTrials.gov — 임상시험 검색

- **운영**: NIH (미국 국립보건원)
- **엔드포인트**: `https://clinicaltrials.gov/api/v2/studies`
- **필터**: `overallStatus=RECRUITING` (현재 모집 중인 것만)
- **가져오는 것**: 해당 질환으로 지금 참여 가능한 임상시험
- **반환 예시**:
  ```
  NCT05123456 | Sirolimus for LAM | Phase 2 | RECRUITING
  참여기관: 서울대병원, Mayo Clinic
  ```
- **코드 위치**: `fetch_clinicaltrials()` 메서드
- **왜 필요한가**: 환자에게 "이런 임상시험에 참여할 수 있다"는 옵션 제공

### 병렬 호출 — 왜 동시에 실행하나

순서대로 실행하면:
```
PubMed(3s) → Guide(3s) → PCF(3s) → Monarch(3s) → CT(3s) = 총 15초
```

`asyncio` 병렬 실행하면:
```
모두 동시 시작 → 가장 느린 것이 끝날 때까지 대기 = 총 3~5초
```

코드에서는 `asyncio.gather()`로 여러 API를 동시에 호출합니다:

```python
non_pubmed_tasks = {
    "pcf_genes": self.fetch_pcf_disease_data(session, d_id),
    "monarch": self.fetch_monarch(session, d_id, original_name),
    "clinical_trials": self.fetch_clinicaltrials(session, original_name),
}
non_pubmed_dict = dict(zip(
    non_pubmed_tasks.keys(),
    await asyncio.gather(*non_pubmed_tasks.values())
))
# 동시에 PubMed도 호출
pubmed_cases, pubmed_guidelines = await asyncio.gather(
    self.fetch_pubmed_cases(session, original_name, d_id),
    self.fetch_pubmed_guidelines(session, original_name),
)
```

### aiohttp.ClientTimeout — 왜 따로 객체를 만드나

```python
# 잘못된 방식 (aiohttp 3.9.5에서 오류)
async with session.get(url, timeout=10) as resp:

# 올바른 방식
async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
```

`aiohttp` 라이브러리 3.9.5 버전부터 `timeout`에 숫자(정수)를 직접 넣으면 오류가 납니다.
반드시 `aiohttp.ClientTimeout(total=N)` 객체로 감싸야 합니다.
이번 세션에서 코드 전체 7곳에서 이 버그를 수정했습니다.

---

## 7. Aurora DB — 어떻게 연결되나

Aurora PostgreSQL은 우리 프로젝트의 중앙 데이터베이스입니다.
Phase 1~5의 모든 입력/출력이 여기에 저장됩니다.

### 연결 정보

```
호스트:  patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com
포트:    5432
DB명:    rarelink
스키마:  rarelinkai
유저:    app_user
비밀번호: Secrets Manager에서 가져옴 (코드에 직접 쓰지 않음)
```

### Secrets Manager — 비밀번호를 안전하게 관리하는 방법

비밀번호를 코드에 직접 쓰면 Git에 올라가는 순간 보안 사고가 납니다.
AWS Secrets Manager는 비밀번호를 AWS 클라우드에 암호화해서 저장하고,
코드에서는 "비밀 이름"(`rare-link-ai/aurora/app-user`)만 적으면
Secrets Manager가 실제 비밀번호를 돌려줍니다.

```python
def _get_db_conn():
    sm = boto3.client("secretsmanager", region_name="ap-northeast-2")
    secret_str = sm.get_secret_value(SecretId="rare-link-ai/aurora/app-user")["SecretString"]
    pwd = json.loads(secret_str)["password"]  # 실제 비밀번호 꺼냄
    return psycopg2.connect(host=DB_HOST, port=5432, database=DB_NAME,
                            user=DB_USER, password=pwd,
                            options="-c search_path=rarelinkai")
```

### 4-Layer DB 아키텍처

우리 DB는 4개의 레이어(층)로 설계되어 있습니다:

```
Layer 0: Raw EMR (원본 데이터)
  └─ emr_raw_event   : 원본 환자 이벤트 (텍스트 덩어리)

Layer 1: Canonical (정제된 데이터)
  ├─ patient          : 환자 기본 정보
  ├─ encounter        : 내원 정보
  ├─ lab_result       : 혈액검사 수치
  └─ vitals           : 활력 징후

Layer 2: Phase IO (각 Phase 입출력)
  ├─ diagnosis_session      : 전체 세션 상태 추적 (어느 Phase까지 완료됐나)
  ├─ phase1_nlp_result      : Phase 1 결과 (NLP HPO)
  ├─ phase2_vision_result   : Phase 2 결과 (X-ray HPO)
  ├─ phase3_lab_result      : Phase 3 결과 (Lab HPO)
  ├─ phase4_scoring_result  : Phase 4 결과 (Top3 후보 질환) ← Phase 5가 여기서 읽음
  └─ rag_api_cache          : API 응답 캐시 (시간이 지나면 만료)

Layer 3: Outcome (최종 결과)
  └─ final_report            : AI 소견서 JSON ← Phase 5가 여기에 씀
```

### Phase 5가 읽는 테이블: `phase4_scoring_result`

```sql
SELECT
    positive_hpo_json,   -- ["HP:0002094", "HP:0002202", ...]
    general_top10_json,  -- [{"name": "폐렴", "score": 0.659}, ...]
    rare_listing_json,   -- [{"name": "LAM", "orpha_code": "723", ...}, ...]
    top3_json            -- [{"name": "LAM", ...}, {"name": "IPF", ...}, ...]
FROM phase4_scoring_result
WHERE session_id = %s
ORDER BY created_at DESC LIMIT 1;
```

### Phase 5가 쓰는 테이블: `final_report`

```sql
INSERT INTO final_report (
    session_id,
    diagnosis_json,    -- AI가 생성한 소견서 전체 JSON
    model_id,          -- 사용한 Bedrock 모델 ID
    created_at
) VALUES (%s, %s, %s, NOW())
ON CONFLICT (session_id) DO UPDATE SET
    diagnosis_json = EXCLUDED.diagnosis_json,
    updated_at = NOW();
```

### Phase 5가 기록하는 테이블: `diagnosis_session`

세션의 현재 상태를 추적합니다:

| status | 의미 |
|--------|------|
| `pending` | 아직 시작 안 됨 |
| `running` | 현재 Phase 5 실행 중 |
| `completed` | 정상 완료 |
| `failed` | 오류 발생 |

Phase 5 시작 시 `running`, 성공 시 `completed`, 오류 시 `failed`로 바꿉니다.

---

## 8. 캐시 시스템 — 왜 필요하고 어떻게 동작하나

### 왜 필요한가

같은 질환(예: LAM)에 대해 PubMed를 매번 호출하면:
- 시간 낭비: 매번 3초씩
- 비용: API 호출 횟수만큼 요금
- PubMed Rate Limit: 초당 3회 제한 걸림

같은 질환 데이터는 일정 기간 동안 거의 변하지 않습니다.
그래서 한번 가져온 결과를 DB에 저장해두고 재사용합니다.

### `rag_api_cache` 테이블 구조

```sql
-- Aurora DB의 rag_api_cache 테이블
cache_key     TEXT         -- 예: "pubmed:lymphangioleiomyomatosis"
response_json JSONB        -- 실제 API 응답 결과
created_at    TIMESTAMP
expires_at    TIMESTAMP    -- 이 시간이 지나면 다시 API 호출
```

### 캐시 키 형식

```python
@staticmethod
def _make_cache_key(prefix: str, value: str) -> str:
    import re
    normalized = re.sub(r'[^a-zA-Z0-9_:-]', '_', value.lower())[:200]
    return f"{prefix}:{normalized}"
```

예시:
- `_make_cache_key("pubmed", "Lymphangioleiomyomatosis")` → `"pubmed:lymphangioleiomyomatosis"`
- `_make_cache_key("monarch", "ORPHA:723")` → `"monarch:orpha_723"`
- `_make_cache_key("clinicaltrials", "LAM")` → `"clinicaltrials:lam"`

공백, 특수문자는 `_`로 교체, 소문자로 통일, 최대 200자 제한.

### 캐시 유효기간 (TTL)

| API | 캐시 유효기간 | 이유 |
|-----|-------------|------|
| PubMed 논문 | 7일 | 논문은 자주 안 바뀜 |
| Monarch 유전자 | 30일 | 유전자 DB는 거의 안 바뀜 |
| ClinicalTrials | 1일 | 임상시험 상태는 자주 바뀜 |

### 캐시 동작 흐름

```
[Step 2] 캐시 확인
  cache_keys = [
    "pubmed:lam", "pubmed:ipf", "pubmed:폐렴",       ← Top3 × PubMed
    "monarch:orpha_723", "monarch:orpha_555", ...      ← Top3 × Monarch
    "clinicaltrials:lam", "clinicaltrials:ipf", ...    ← Top3 × CT
  ]
  api_cache = self._read_api_cache(cache_keys)
  → DB에서 아직 만료 안 된 캐시만 가져옴

[Step 3] API 호출
  for disease in top_3:
    if cached_pubmed:    # 캐시 있으면
        pubmed_data = cached_pubmed  # DB에서 꺼냄 (API 호출 안 함)
    else:               # 캐시 없으면
        pubmed_data = await self.fetch_pubmed_cases(...)  # 실시간 호출

[Step 5] 저장
  # 새로 가져온 데이터는 캐시에 저장
  INSERT INTO rag_api_cache (cache_key, response_json, expires_at)
  VALUES ('pubmed:lam', {...}, NOW() + INTERVAL '7 days')
  ON CONFLICT (cache_key) DO UPDATE SET response_json=..., expires_at=...
```

---

## 9. Bedrock AI — 어떻게 소견서를 만드나

### 사용 모델

```
모델: anthropic.claude-3-5-sonnet-20241022-v2:0
호출: AWS Bedrock Runtime (VPC Endpoint 경유)
온도: 0.0 (항상 같은 답변 → 의료용 재현성)
최대 토큰: 4096
```

### 왜 Bedrock인가

일반 Claude API를 쓰면:
- 환자 데이터가 Anthropic 서버로 전송
- 개인정보보호법(HIPAA, 한국 개보법) 위반 가능성

Bedrock을 쓰면:
- AWS 내부에서만 처리
- 환자 데이터가 AWS 밖으로 나가지 않음
- VPC Endpoint를 쓰면 인터넷도 거치지 않음 (완전 내부 통신)

### AI에게 전달하는 프롬프트 구조

```
=== 시스템 프롬프트 (고정) ===
- 역할: 호흡기내과 전문의 보조 AI
- 규칙: 근거 없는 추측 금지, 모든 주장은 RAG 데이터 기반
- 출력 형식: JSON만 (마크다운 금지)
- 희귀질환이 Top3에 있으면 MDT 협진 권고 필수
- 환자 MRN(개인번호) 포함 금지

=== 유저 프롬프트 (환자마다 동적 생성) ===
섹션 1: 환자 기본정보 (나이/성별, MRN 제외)
섹션 2: 증상 원문 (양성/음성 소견)
섹션 3: HPO 프로파일 (각 코드의 출처 포함)
섹션 4: Lab 수치
섹션 5: 일반 폐질환 Top10 (로컬 DB 기반)
섹션 6: 희귀질환 리스팅 (LR 기반)
섹션 7: RAG 검색 결과 (PubMed/Monarch/PCF/CT)
```

### AI 출력 JSON 형식

```json
{
  "recommendation": {
    "immediate_workup": ["즉시 시행할 검사"],
    "specialist_referral": ["협진 권고"],
    "treatment_guideline": ["치료 가이드라인"],
    "genetic_test": ["유전자 검사 권고"],
    "additional_lab": ["추가 혈액검사"]
  },
  "clinical_notes": {
    "summary": "환자 종합 요약",
    "top1_reasoning": "1순위 진단 근거",
    "differential_note": "2~3순위 감별진단",
    "rag_evidence": "RAG에서 찾은 주요 근거",
    "case_comparison": "PubMed 사례 비교",
    "epidemiology_note": "역학 정보 (희귀질환만)",
    "disclaimer": "AI 결과는 진단 보조이며..."
  }
}
```

---

## 10. Lambda handler.py — 요청 처리 진입점

`handler.py`는 206줄의 짧은 파일입니다.
Lambda가 실행되면 제일 먼저 호출되는 함수(`lambda_handler`)가 여기 있습니다.

### 싱글턴 패턴 — Cold Start 최적화

```python
_rag_system = None  # 전역 변수

def _get_rag_system():
    global _rag_system
    if _rag_system is None:
        # 처음 요청(Cold Start)에서만 실행
        from rag_llm_3 import RareLinkHybridDualRAG
        _rag_system = RareLinkHybridDualRAG()  # HPO JSON 로딩 등 초기화 (~3초)
    return _rag_system  # 이후 요청에서는 이미 만들어진 것 재사용
```

**Cold Start**: Lambda가 처음 켜질 때 초기화 작업이 필요합니다.
싱글턴 패턴을 쓰면 초기화를 최초 1회만 합니다.
이후 요청은 이미 초기화된 인스턴스를 재사용해서 빠릅니다.

### 이벤트 처리 — 두 가지 방식

`handler.py`는 두 가지 방식의 요청을 모두 처리합니다:

**방식 1: API Gateway (HTTP 요청)**
```
POST /run
Content-Type: application/json
{"session_id": "00000000-0000-0000-0000-000000000001"}
```

**방식 2: Step Functions (직접 호출)**
```json
{"session_id": "00000000-0000-0000-0000-000000000001"}
```

둘 다 `session_id`를 추출해서 `rag.run_with_session_id(session_id)`를 호출합니다.

### /health 엔드포인트

```
GET /health
→ {"status": "healthy", "stage": "dev", "elapsed_ms": 0}
```

Lambda가 정상 작동하는지 확인하는 용도입니다.

### 응답 형식

성공:
```json
{
  "statusCode": 200,
  "body": "{\"session_id\": \"...\", \"status\": \"completed\", \"elapsed_ms\": 45320}"
}
```

실패 (session_id 없음):
```json
{"statusCode": 400, "body": "{\"error\": \"session_id is required...\"}"}
```

서버 오류:
```json
{"statusCode": 500, "body": "{\"error\": \"Internal error: RuntimeError\"}"}
```

---

## 11. SAM 배포 — Lambda를 AWS에 올리는 방법

### SAM이 뭔가

**SAM (Serverless Application Model)** = AWS에서 만든 Lambda 배포 도구

일반 Lambda 배포 방식:
1. 코드를 zip으로 압축
2. AWS 콘솔에 접속
3. Lambda 함수 만들기
4. zip 파일 업로드
5. API Gateway 만들기
6. IAM 권한 설정
7. CloudWatch 알람 설정
8. Layer 만들기...

SAM 방식:
1. `template.yaml` 파일 하나에 모든 설정 작성
2. `sam build` 실행 (코드 컴파일)
3. `sam deploy` 실행 (AWS에 자동 배포)

SAM이 `template.yaml`을 읽고 AWS에 필요한 모든 리소스를 자동으로 만들어줍니다.

### 파일 구조

```
phase5_build/infra/aws/phase5/
├── template.yaml          ← SAM 설정 파일 (모든 AWS 리소스 정의)
├── deploy.sh              ← 배포 자동화 스크립트
├── lambda/
│   ├── handler.py         ← Lambda 진입점
│   └── rag_llm_3.py       ← RAG 로직 (배포 시 S3에서 다운로드)
└── layer/
    ├── build_layer.sh     ← 의존성 빌드 스크립트
    └── deps-build/python/ ← pip 설치된 패키지들 (173MB)
```

### template.yaml — 핵심 설정 설명

```yaml
# Lambda 함수 정의
Phase5RagFunction:
  Type: AWS::Serverless::Function
  Properties:
    FunctionName: phase5-rag-dev
    Runtime: python3.11          # Python 3.11 실행 환경
    Handler: handler.lambda_handler  # handler.py의 lambda_handler 함수
    Timeout: 300                 # 최대 5분 실행 (RAG가 오래 걸림)
    MemorySize: 1024             # 1GB 메모리 (pandas 등 사용)

    # VPC 설정 (Aurora 접근하려면 필수)
    VpcConfig:
      SubnetIds:
        - subnet-02eed659772bac6aa  # private-a (NAT Gateway)
        - subnet-08f8d0eaa597b4f04  # private-b (NAT Gateway)
      SecurityGroupIds:
        - sg-08d35c498d8886a98      # say2-2team-sg-lambda

    # IAM 권한
    Policies:
      - AWSLambdaVPCAccessExecutionRole   # VPC 안에서 실행 권한
      - bedrock:InvokeModel 권한           # Bedrock 호출 권한
      - s3:GetObject 권한                 # S3 읽기 권한
      - secretsmanager:GetSecretValue 권한 # 비밀번호 읽기 권한

    # API Gateway 엔드포인트
    Events:
      RunPost:
        Path: /run
        Method: POST
      HealthGet:
        Path: /health
        Method: GET
```

### Lambda Layer — 의존성 패키지

Lambda 함수 코드 크기에는 50MB 제한이 있습니다.
우리가 쓰는 Python 패키지들(`aiohttp`, `psycopg2`, `pandas`)은 합계 173MB입니다.

이걸 해결하는 게 **Lambda Layer**입니다:
- Layer에 패키지를 따로 올려두면 (최대 250MB)
- 여러 Lambda 함수가 같은 Layer를 공유
- Lambda 함수 코드는 실제 코드만 (작게 유지)

```
Lambda 실행 환경:
/opt/python/           ← Layer (aiohttp, psycopg2, pandas, requests)
/var/task/             ← Lambda 코드 (handler.py, rag_llm_3.py)
```

#### 왜 manylinux2014로 빌드해야 하나

Mac에서 `pip install psycopg2`를 하면 Mac용 바이너리가 설치됩니다.
Lambda는 Amazon Linux 2 (Linux x86_64)에서 실행됩니다.
Mac 바이너리를 Lambda에 올리면 실행 오류가 납니다.

```bash
# layer/build_layer.sh
# Amazon Linux 2 환경에서 패키지 빌드
pip install psycopg2-binary==2.9.9 \
    --target ./python \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --python-version 3.11 \
    --only-binary=:all: \
    --no-cache-dir
```

`manylinux2014_x86_64` = "Amazon Linux 2에서 돌아가는 Linux 64비트 바이너리"

### deploy.sh — 배포 자동화

```bash
# 5단계로 자동 배포
Step 1: S3에서 최신 rag_llm_3.py 다운로드
Step 2: Lambda Layer 빌드 (manylinux2014 바이너리)
Step 3: sam build (코드 컴파일)
Step 4: sam deploy (AWS에 배포)
Step 5: CloudFormation 결과 출력
```

사용법:
```bash
chmod +x deploy.sh
./deploy.sh dev      # 개발 환경 배포
./deploy.sh staging  # 스테이징 환경 배포
```

### Python 3.11 필요 이유

SAM은 `sam build` 시 로컬에 Python 3.11이 설치되어 있어야 합니다.
Lambda 런타임이 Python 3.11이기 때문에 빌드도 Python 3.11로 해야 합니다.

이번 배포에서 Python 3.14만 있어서 빌드가 처음에 실패했습니다.
`brew install python@3.11` 으로 설치 후 해결.

---

## 12. VPC 네트워크 구조 — 왜 복잡한가

VPC(Virtual Private Cloud) = AWS 안의 사설 네트워크. 인터넷과 분리됨.

우리 Lambda는 VPC 안에 있습니다. 이유:
- Aurora DB가 VPC 안 Private Subnet에 있기 때문
- Lambda가 Aurora에 접근하려면 같은 VPC에 있어야 함

하지만 VPC 안에 있으면 인터넷(PubMed, Monarch 등)에 접근이 안 됩니다.
이걸 해결하는 3가지 구성:

```
[Lambda] ← VPC 안 Private Subnet에 배치
    │
    ├─→ [Aurora DB] ← 같은 VPC → 직접 접근 가능 (포트 5432)
    │
    ├─→ [Bedrock] ← VPC Endpoint (vpce-0154ce36d821b29be)
    │      인터넷 거치지 않고 AWS 내부망으로 Bedrock 호출
    │
    ├─→ [S3] ← S3 Gateway Endpoint (vpce-06427d6b33b6e607f)
    │      인터넷 거치지 않고 S3 접근
    │
    └─→ [PubMed, Monarch, CT 등 외부 API] ← NAT Gateway 경유
           VPC 안에서 인터넷으로 나가는 관문
           Private Subnet → NAT Gateway → 인터넷
```

### VPC 관련 에러: Aurora SG 설정

Lambda(`sg-08d35c498d8886a98`)가 Aurora(`sg-019a357627f1594db`)에 접근하려면
Aurora의 Security Group(방화벽)이 Lambda의 SG를 허용해야 합니다.

이번 세션에서 수동으로 추가:
```bash
aws ec2 authorize-security-group-ingress \
    --group-id sg-019a357627f1594db \     # Aurora SG
    --source-group sg-08d35c498d8886a98 \  # Lambda SG
    --protocol tcp \
    --port 5432                            # PostgreSQL 포트
```

---

## 13. 배포된 실제 AWS 리소스 목록

2026-05-15 배포 완료. 스택명: `phase5-rag-dev`

| 리소스 | ID / 이름 | 용도 |
|--------|---------|------|
| **Lambda Function** | `phase5-rag-dev` | RAG 파이프라인 실행 |
| **Lambda Layer** | `phase5-deps-dev:1` | Python 패키지 (173MB) |
| **API Gateway** | `phase5-rag-api-dev` | HTTP 엔드포인트 |
| **IAM Role** | `Phase5RagFunctionRole` | Lambda 실행 권한 |
| **CloudWatch Log** | `/aws/lambda/phase5-rag-dev` | 실행 로그 (90일 보관) |
| **CloudWatch Alarm** | `phase5-rag-dev-errors` | 에러율 5회 이상 시 경보 |
| **CloudWatch Alarm** | `phase5-rag-dev-latency-p99` | 280초 이상 시 경보 |

**엔드포인트**:
```
Base URL: https://1nvbtwc3wd.execute-api.ap-northeast-2.amazonaws.com/dev
Run URL:  https://1nvbtwc3wd.execute-api.ap-northeast-2.amazonaws.com/dev/run
Health:   https://1nvbtwc3wd.execute-api.ap-northeast-2.amazonaws.com/dev/health
Lambda ARN: arn:aws:lambda:ap-northeast-2:666803869796:function:phase5-rag-dev
```

**S3 저장 위치**:
```
s3://say2-2team-bucket/RAG/rag_llm_3.py           ← RAG 코드 (최신)
s3://say2-2team-bucket/RAG/infra/template.yaml    ← SAM 템플릿
s3://say2-2team-bucket/RAG/infra/deploy.sh        ← 배포 스크립트
s3://say2-2team-bucket/RAG/infra/lambda/handler.py ← Lambda 핸들러
```

---

## 14. 실패 대비 설계 — 뭔가 터져도 괜찮은 이유

각 컴포넌트가 실패해도 파이프라인 전체가 멈추지 않도록 설계됐습니다.

### 각 컴포넌트별 실패 처리

| 컴포넌트 | 실패 시 처리 |
|----------|-------------|
| DB 연결 실패 | `_get_db_conn()` → `None` 반환, 이후 로직은 DB 없이 계속 |
| Phase 4 결과 없음 | `ValueError` 발생 → Lambda가 400 에러로 응답 |
| PubMed API 실패 | `"N/A"` 반환, 해당 섹션 비워두고 계속 |
| Monarch API 실패 | `"N/A"` 반환, 유전자 정보 없이 계속 |
| ClinicalTrials 실패 | `"N/A"` 반환, 임상시험 섹션 없이 계속 |
| PubCaseFinder 실패 | `"N/A"` 반환, PCF 섹션 없이 계속 |
| Bedrock 호출 실패 | `ClientError` 포착 → 에러 메시지 반환 |
| 캐시 읽기 실패 | 빈 dict 반환, 모든 API 새로 호출 |
| 캐시 쓰기 실패 | 경고 로그만, 파이프라인 계속 |
| Aurora 저장 실패 | 경고 로그만, Lambda는 성공 응답 |
| JSON 파싱 실패 | raw 텍스트 그대로 저장 |

### 예외 처리 패턴

```python
async def fetch_pubmed_cases(self, session, disease_name, ...):
    try:
        # API 호출
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=12)) as resp:
            data = await resp.json()
            return data
    except Exception as e:
        print(f"⚠️ PubMed 호출 실패: {e}")
        return "N/A"  # 실패해도 "N/A" 반환 → 파이프라인 계속
```

모든 외부 API 호출에 `try/except`가 있고, 실패 시 `"N/A"`를 반환합니다.
`"N/A"`가 프롬프트에 들어가도 AI는 "해당 데이터 없음"으로 처리하고 소견서를 작성합니다.

### `_mark_session_failed` — 실패 기록

파이프라인 전체가 오류로 중단되면 DB에 실패를 기록합니다:

```python
def _mark_session_failed(self, session_id: str, error_msg: str):
    # diagnosis_session 테이블: status='failed', error_message=error_msg
```

프론트엔드나 팀원이 "이 세션 왜 아직도 pending이야?"를 방지합니다.

---

## 15. 이번 세션에서 구현/수정한 것 전체 목록

### 새로 구현한 메서드 3개

**`_set_session_running(session_id)`**
- Phase 5 시작 시 `diagnosis_session.status = 'running'`으로 업데이트
- 이유: Phase 5가 오래 걸려서 "아직 실행 중"임을 DB에 기록해야 함

**`_make_cache_key(prefix, value)`**
- 캐시 키 생성 헬퍼 (예: `"pubmed:lam"`)
- 특수문자/공백 정규화, 소문자 통일, 200자 제한

**`_read_api_cache(cache_keys)`**
- DB `rag_api_cache` 테이블에서 만료되지 않은 캐시 일괄 조회
- `WHERE cache_key IN (...) AND expires_at > NOW()`

### 수정한 메서드 2개

**`gather_rag_data(pos_hpos, local_top_3, api_cache=None)`**
- 기존: 모든 API를 항상 새로 호출
- 수정: `api_cache` 파라미터 추가 → 캐시 있으면 API 호출 건너뜀

**`_save_to_db(session_id, ...)`**
- 기존: PubMed 캐시만 저장
- 수정: PubMed(7일) + Monarch(30일) + ClinicalTrials(1일) 캐시 모두 저장

### 버그 수정 7건 — aiohttp.ClientTimeout

`aiohttp 3.9.5`에서 `timeout=숫자` 형식이 더 이상 안 됨.
모두 `timeout=aiohttp.ClientTimeout(total=숫자)` 형식으로 수정.

수정 위치:
1. `fetch_pcf_disease_data` — timeout=10
2. `fetch_monarch` (Monarch entity API) — timeout=10
3. `fetch_monarch` (HPO name API) — timeout=5
4. `fetch_clinicaltrials` — timeout=12
5. `fetch_pubmed_cases` (esearch) — timeout=10
6. `fetch_pubmed_cases` (efetch) — timeout=10
7. `fetch_pubmed_guidelines` — timeout=10

### template.yaml 수정

**VpcConfig 추가**:
```yaml
VpcConfig:
  SubnetIds:
    - subnet-02eed659772bac6aa
    - subnet-08f8d0eaa597b4f04
  SecurityGroupIds:
    - sg-08d35c498d8886a98
```
이유: Aurora에 접근하려면 같은 VPC 안에 있어야 함

**AWSLambdaVPCAccessExecutionRole 추가**:
이유: VPC 안에서 Lambda를 실행하려면 ENI(네트워크 인터페이스) 생성 권한 필요

### deploy.sh 수정

`--use-container` 옵션 제거:
- 기존: `sam build --use-container` (Docker 컨테이너 안에서 빌드)
- 수정: `sam build` (로컬에서 빌드)
- 이유: Docker가 실행 중이 아니면 빌드 실패. Layer는 이미 manylinux2014로 빌드됨.

### Aurora SG 설정 (수동)

```bash
aws ec2 authorize-security-group-ingress \
    --group-id sg-019a357627f1594db \
    --source-group sg-08d35c498d8886a98 \
    --protocol tcp --port 5432
```

이유: Aurora SG가 Lambda SG를 허용하지 않아서 DB 연결 실패

---

## 16. 테스트 방법

### 헬스 체크

```bash
curl "https://1nvbtwc3wd.execute-api.ap-northeast-2.amazonaws.com/dev/health"
# 결과: {"status": "healthy", "stage": "dev", "elapsed_ms": 0}
```

### 실제 RAG 실행

```bash
curl -X POST "https://1nvbtwc3wd.execute-api.ap-northeast-2.amazonaws.com/dev/run" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "실제-aurora-세션-UUID"}'
```

Aurora DB에 해당 `session_id`의 Phase 4 결과가 있어야 정상 실행됩니다.

### AWS 콘솔에서 테스트

1. AWS 콘솔 → Lambda → `phase5-rag-dev`
2. [테스트] 탭 클릭
3. 이벤트 JSON 입력:
   ```json
   {"session_id": "00000000-0000-0000-0000-000000000001"}
   ```
4. [테스트] 버튼 클릭

### 로그 확인

```bash
# AWS CLI로 최신 로그 조회
aws logs tail /aws/lambda/phase5-rag-dev --follow --region ap-northeast-2
```

또는 AWS 콘솔 → CloudWatch → 로그 그룹 → `/aws/lambda/phase5-rag-dev`

### 재배포 방법 (코드 수정 후)

```bash
# 1. rag_llm_3.py 수정 후 S3 업로드
aws s3 cp rag_llm_3.py s3://say2-2team-bucket/RAG/rag_llm_3.py

# 2. 재배포
cd /tmp/phase5_build/infra/aws/phase5
export PATH="/opt/homebrew/bin:$PATH"
./deploy.sh dev
```

---

## 17. 용어집

| 용어 | 설명 |
|------|------|
| **HPO** | Human Phenotype Ontology. 증상을 국제 표준 코드로 표현. `HP:0002094` = 호흡곤란 |
| **OrphaCode** | Orphanet 희귀질환 DB 고유 ID. `ORPHA:723` = LAM |
| **OMIM ID** | 유전성 질환 DB 고유 ID. `OMIM:617300` |
| **LR** | Likelihood Ratio. 우도비. "이 증상 조합이 이 질환일 가능성이 일반인 대비 몇 배인가" |
| **LIRICAL** | LR 기반 희귀질환 스코어링 알고리즘 (Jacobsen et al. 2020) |
| **Lambda** | AWS의 서버리스 실행 환경. 요청이 올 때만 실행 |
| **Lambda Layer** | Lambda 함수에서 공유하는 패키지 묶음 (우리는 173MB) |
| **SAM** | Serverless Application Model. Lambda 배포 도구 |
| **VPC** | Virtual Private Cloud. AWS 내 사설 네트워크 |
| **NAT Gateway** | VPC 안에서 인터넷으로 나가는 관문 |
| **VPC Endpoint** | VPC 안에서 AWS 서비스(S3, Bedrock)에 인터넷 없이 접근하는 통로 |
| **Security Group** | 방화벽 규칙. 어떤 IP/포트를 허용할지 정의 |
| **Aurora PostgreSQL** | AWS의 고성능 PostgreSQL 호환 DB. 우리 중앙 DB |
| **Secrets Manager** | AWS 비밀번호/키를 안전하게 저장하는 서비스 |
| **Bedrock** | AWS의 AI 모델 실행 서비스. Claude, Titan 등을 API로 제공 |
| **Cold Start** | Lambda가 처음 켜질 때 초기화 시간이 걸리는 현상 |
| **Singleton** | 객체를 한 번만 만들고 재사용하는 패턴. Cold Start 최적화에 사용 |
| **asyncio** | Python 비동기 실행 라이브러리. 여러 API를 동시에 호출할 때 사용 |
| **aiohttp** | asyncio 기반 HTTP 클라이언트 라이브러리 |
| **psycopg2** | Python에서 PostgreSQL 연결하는 라이브러리 |
| **manylinux2014** | Amazon Linux 2 호환 Linux 바이너리 빌드 플랫폼 |
| **session_id** | 각 환자 진단 세션의 UUID. 모든 Phase를 연결하는 공통 키 |
| **MDT** | Multi-Disciplinary Team. 희귀질환은 여러 과 의사가 협진 |
| **TTL** | Time To Live. 캐시 유효기간 |
| **CloudFormation** | AWS 인프라를 코드로 관리하는 서비스. SAM이 내부적으로 사용 |
| **ENI** | Elastic Network Interface. VPC 안 Lambda의 네트워크 카드 |

---

*이 보고서는 2026-05-15 배포 완료 코드 기준으로 작성됐습니다.*
*`rag_llm_3.py` 수정 시 해당 섹션을 업데이트하세요.*
*문의: 허태웅 (Slack `#skku-2기-2팀`)*
