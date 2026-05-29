# 에러 처리 현황 분석 및 Phase 3/4/5 업데이트 보고서

> 작성일: 2026-05-17 (최초)
> 최종 수정: 2026-05-17 (`phase_execution_log` 테이블 발견 후 §0, §3 옵션 갱신)
> 대상: `say2-2team-bucket` 의 Phase 1~5, RAG 코드베이스
> 목적: Phase 1/2/RAG의 에러 처리 패턴을 분석하고, 같은 방식을 Phase 3/4/5에 적용하기 위한 가이드

---

## 0. ⭐ 핵심 결론 (먼저 읽으세요)

**팀이 이미 완벽한 에러 로그 테이블을 설계해뒀습니다.** 따로 새 테이블을 만들 필요 없습니다.

- DDL 위치: `database/system-log-schema-ddl.sql`, `scripts/4-layer-schema-ddl-v1.1.sql`
- 테이블 이름: **`soopulai.phase_execution_log`**
- 상태: 스키마는 Aurora 16.4에 배포 완료 (`4-layer-db-team-guide.md` 명시), 그러나 **현재 코드 어디서도 INSERT를 안 함**
- 우리 작업: 새 테이블 생성이 아니라 **이미 있는 테이블에 INSERT 코드 추가**

`phase_execution_log` 에 이미 정의된 컬럼:

| 컬럼 | 타입 | 용도 |
|---|---|---|
| `log_id` | UUID PK | 자동 생성 |
| `session_id` | UUID FK | `diagnosis_session` 연결 |
| `patient_id` | VARCHAR(64) | 환자 ID |
| `phase_name` | VARCHAR(16) | 'phase1'..'phase5', 'rag', 'orchestrator' |
| `phase_step` | VARCHAR(64) | 세부 단계 (예: 'unet_mask', 'api_pubmed') |
| `status` | VARCHAR(16) | 'started','succeeded','failed','timeout','retrying' |
| `started_at`, `completed_at`, `duration_ms` | 타이밍 | 실행 시간 추적 |
| `lambda_function`, `lambda_request_id`, `lambda_memory_mb`, `lambda_billed_ms` | Lambda 메타 | 과금/디버깅 |
| `input_summary`, `output_summary` | JSONB | 입출력 요약 |
| **`error_code`** | VARCHAR(64) | 'LLM_TIMEOUT', 'SAGEMAKER_ERROR' 등 |
| **`error_message`** | TEXT | 메시지 전문 |
| **`error_stacktrace`** | TEXT | ★ Python traceback 전체 |
| **`error_category`** | VARCHAR(32) | 'infra','model','data','external_api','validation' |
| `retry_count`, `retry_of_log_id` | 재시도 추적 |
| `external_calls`, `model_versions` | JSONB | 외부 API 호출, 모델 버전 |

**미리 만들어진 뷰도 있음:**
- `recent_errors` — 최근 100개 실패
- `phase_success_rates_24h` — 24시간 Phase별 성공률

→ 자세한 적용 방법은 §3.5 (`_record_phase_log` 헬퍼) 참조

---

## 1. 현재 상태 — 어디서 어떤 에러 처리를 하고 있나

| 모듈 | 에러 로깅 | DB 업데이트 | 위치 |
|---|---|---|---|
| **Phase 1** `symptom_llm_4.py` | ✅ 로컬 파일 (`logs/*.json`) | ❌ **없음** | `_log_error()` L63-81, `extract_hpo_from_clinical_note()` L257-259 |
| **Phase 2** `soo_net_5.py` | ✅ 로컬 파일 (`logs/*.json`) | ❌ **없음** | `_log_error()` L164-182, `predict()` L191-193, `get_cam_visualize()` L233-235 |
| **RAG** `rag_llm_3.py` | ✅ 파일 (`/tmp/logs` on Lambda) | ✅ **있음** | `_log_error()` L257, `_mark_session_failed()` L738, `run_with_session_id()` L1385-1388 |
| **Phase 3** Lambda `handler.py` | ⚠️ CloudWatch 로그만 (`logger.exception`) | ❌ **없음** | `_server_error()` L175 |
| **Phase 4** Lambda `handler.py` | ⚠️ CloudWatch 로그만 (`logger.exception`) | ❌ **없음** | `_server_error()` L80 |
| **Phase 5** Lambda `handler.py` (RAG 래퍼) | ⚠️ CloudWatch (래퍼) + RAG 내부 DB | ✅ **부분적** | RAG 내부 `_mark_session_failed`는 동작, 그러나 cold-start/import 실패 시 DB 누락 |

### 핵심 발견
사용자 추정대로 **Phase 1, 2의 `_log_error`는 로컬 파일 저장뿐**입니다. DB 연동은 없습니다. RAG만이 다음 두 가지를 함께 합니다.

1. `_log_error()` → `/tmp/logs/error_*.json`
2. `_mark_session_failed(session_id, error_msg)` → `diagnosis_session` 테이블에 `status='failed', error_message=..., completed_at=NOW()` 업데이트

---

## 1.5 왜 DB에 에러를 기록해야 하나 — 적용 이유

### 🔴 현재 구조의 5가지 문제점

#### ① CloudWatch 로그는 "환자 단위"로 검색이 어렵다
지금 Phase 3/4는 `logger.exception()`으로 CloudWatch에만 에러를 남깁니다. 그런데 CloudWatch 로그는 **시간순 텍스트 덤프**입니다. "환자 ID `P-12345`의 진단이 왜 실패했지?"를 알려면:

- Lambda 함수 4~5개의 로그 그룹을 각각 열고
- 시간 범위를 추정해서 필터링하고
- `request_id`를 찾아서 텍스트 검색
- 운이 좋으면 스택 트레이스 발견

반면 DB 패턴은 **한 줄 SQL**로 끝납니다.
```sql
SELECT current_phase, status, error_message
FROM soopulai.diagnosis_session WHERE session_id = 'P-12345';
```

#### ② 프론트엔드가 "사용자에게 뭘 보여줄지" 모른다
지금 Phase 3 Lambda가 500을 던지면 프론트는 `{"error": "internal_server_error", "type": "KeyError"}` 만 받습니다. 사용자에게는:

- "어디까지 진행됐는지" — 모름
- "다시 시도해도 되는지" — 모름
- "관리자가 알고 있는지" — 모름

DB에 `current_phase=3, status='failed'`가 있으면 프론트가 **"Phase 3 (영상 분석)에서 실패했습니다. 다시 시도하시겠어요?"** 같은 구체적 메시지를 띄울 수 있습니다.

#### ③ Step Functions에서 분기 처리가 안 된다
파이프라인이 Phase 1 → 2 → 3 → 4 → 5 순서로 흐를 때, 중간이 실패하면:

- **현재**: Step Functions가 그냥 멈춤. 운영자가 손으로 어디서 죽었는지 찾아야 함
- **DB 적용 후**: `SELECT WHERE status='failed'` 로 매일 자동 리포트, 또는 Step Functions의 다음 단계가 "Phase 3가 성공했나?"를 DB로 확인하고 진행

#### ④ 같은 에러가 반복돼도 알 수 없다
CloudWatch만으로는 "지난 1시간 동안 같은 종류의 에러가 50번 났다"를 알 수 없습니다. DB에 쌓이면:
```sql
SELECT error_message, COUNT(*) FROM diagnosis_session
WHERE status='failed' AND completed_at > NOW() - INTERVAL '1 hour'
GROUP BY error_message ORDER BY 2 DESC;
```
→ 즉시 알람/대시보드 구성 가능.

#### ⑤ RAG/Phase 5만 DB 기록 → 운영팀에 인지 부하
지금 운영팀은 "Phase 5는 DB 보고, Phase 3·4는 CloudWatch 보고, Phase 1·2는 로컬 파일 본다"는 **3가지 다른 디버깅 절차**를 외워야 합니다. 한 가지 패턴으로 통일하면 신규 팀원 온보딩도, 장애 대응도 빨라집니다.

### 🟢 왜 하필 "RAG 패턴"을 따라야 하나
- **이미 운영 중** — 검증된 코드라 새 표준을 만드는 것보다 리스크가 낮음
- **테이블이 이미 있음** — `diagnosis_session` 에 필요한 컬럼(`status`, `error_message`, `current_phase`, `completed_at`)이 다 존재
- **권한·VPC가 이미 구성됨** — Phase 5 Lambda가 쓰는 Secrets Manager/RDS 권한을 그대로 복사하면 됨
- **`current_phase` 컬럼이 결정적** — 이 한 컬럼만으로 "환자가 지금 어느 단계에 있는지"가 명확해짐. 새로 도입하는 게 아니라 **현재 누락된 phase 3, 4의 기록만 채우는 것**

---

## 2. RAG가 DB에 에러를 기록하는 표준 패턴

```python
# 1) DB 연결 (Secrets Manager + psycopg2)
DB_HOST = "patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com"
DB_NAME = "soopul"
DB_USER = "app_user"
DB_SECRET_ID = "soopul/aurora/app-user"

def _get_db_conn():
    if not DB_AVAILABLE:
        return None
    try:
        sm = boto3.client("secretsmanager", region_name="ap-northeast-2")
        secret_str = sm.get_secret_value(SecretId=DB_SECRET_ID)["SecretString"]
        pwd = json.loads(secret_str).get("password", secret_str)
        return psycopg2.connect(
            host=DB_HOST, port=5432, database=DB_NAME, user=DB_USER, password=pwd,
            options="-c search_path=soopulai", connect_timeout=10
        )
    except Exception as e:
        print(f"⚠️ DB 연결 실패: {e}")
        return None

# 2) 에러 시 DB UPDATE
def _mark_session_failed(self, session_id: str, error_msg: str):
    conn = _get_db_conn()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE diagnosis_session
            SET status='failed', error_message=%s, completed_at=NOW()
            WHERE session_id=%s
        """, (str(error_msg)[:1000], session_id))
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()

# 3) 메인 파이프라인 최상단 except에서 호출
try:
    ...
except Exception as e:
    self._log_error("run_with_session_id", session_id, e)
    self._mark_session_failed(session_id, str(e))
    raise
```

**테이블 스키마 (확인됨):** `soopulai.diagnosis_session` 에 `status`, `error_message`, `completed_at`, `current_phase` 컬럼이 존재합니다.

---

## 3. Phase 3, 4, 5에 적용할 수정 방안

### ⚠️ 사전 결정 필요 (적용 전 확인해야 할 것)

1. **session_id 전달 경로** — Phase 3, 4의 현재 입력은 dataclass(`patient_lab_findings`, `phase3_ranking` 등)만 받고 `session_id`가 없습니다. 호출자(Step Functions/프론트엔드)가 `event["session_id"]`를 함께 보내도록 해야 합니다.
2. **current_phase 의미** — RAG는 자기 시작 시 `current_phase=5`로 업데이트합니다. Phase 3/4도 같은 패턴(`current_phase=3`, `current_phase=4`)을 유지하면 어디서 죽었는지 즉시 알 수 있어 추천합니다.
3. **에러 분류** — `_bad()`(400, 사용자 입력 오류)는 DB 실패 처리할지 결정 필요. RAG는 `ValueError`를 `_bad`로만 응답하고 DB 업데이트는 안 합니다. Phase 3/4도 동일하게 가져가는 것을 권장합니다.

---

### 📋 3.5 [추천] `phase_execution_log` 통합 헬퍼 — Phase 1~5 공통

`diagnosis_session.error_message` 컬럼 UPDATE 외에, **상세 로그를 `phase_execution_log`에 INSERT** 하는 헬퍼를 추가합니다. 이것이 사용자 요구("에러 전체를 DB에 저장")를 가장 잘 만족합니다.

```python
import json, os, time, uuid, traceback, boto3, psycopg2
from psycopg2.extras import Json as PgJson

def _classify_error(exc: Exception) -> str:
    """error_category 자동 분류 (DDL 정의: infra/model/data/external_api/validation)"""
    name = type(exc).__name__
    if name in ("ValueError", "TypeError", "KeyError"): return "validation"
    if name in ("TimeoutError",): return "infra"
    if "boto" in name.lower() or "ClientError" in name: return "external_api"
    if "torch" in str(exc).lower() or "cuda" in str(exc).lower(): return "model"
    return "infra"

def _record_phase_log(
    session_id: str | None,
    patient_id: str | None,
    phase_name: str,                       # 'phase1'..'phase5', 'rag'
    phase_step: str = "",
    status: str = "started",               # 'started','succeeded','failed','timeout','retrying'
    started_at: float | None = None,       # time.time() 값
    input_summary: dict | None = None,
    output_summary: dict | None = None,
    error: Exception | None = None,
    lambda_request_id: str = "",
    lambda_function: str = "",
    model_versions: dict | None = None,
) -> str | None:
    """phase_execution_log에 1행 INSERT. 신규 log_id 반환."""
    conn = _get_db_conn()
    if not conn: return None
    log_id = str(uuid.uuid4())
    try:
        cur = conn.cursor()

        error_code = error_message = error_stacktrace = error_category = None
        if error:
            error_code      = type(error).__name__.upper()[:64]
            error_message   = str(error)[:8000]
            error_stacktrace = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )[:16000]
            error_category  = _classify_error(error)

        duration_ms = int((time.time() - started_at) * 1000) if started_at else None

        cur.execute("""
            INSERT INTO phase_execution_log (
                log_id, session_id, patient_id,
                phase_name, phase_step, status,
                started_at, completed_at, duration_ms,
                lambda_function, lambda_request_id,
                input_summary, output_summary,
                error_code, error_message, error_stacktrace, error_category,
                model_versions
            ) VALUES (
                %s, %s, %s,
                %s, %s, %s,
                COALESCE(to_timestamp(%s), NOW()), NOW(), %s,
                %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s
            )
        """, (
            log_id, session_id, patient_id,
            phase_name, phase_step, status,
            started_at, duration_ms,
            lambda_function, lambda_request_id,
            PgJson(input_summary) if input_summary else None,
            PgJson(output_summary) if output_summary else None,
            error_code, error_message, error_stacktrace, error_category,
            PgJson(model_versions) if model_versions else None,
        ))
        conn.commit()
        return log_id
    except Exception as e:
        print(f"⚠️ phase_execution_log INSERT 실패: {e}")
        return None
    finally:
        conn.close()
```

#### 사용 예 (Phase 3 핸들러)

```python
def lambda_handler(event, context):
    session_id = event.get("session_id") or body.get("session_id")
    patient_id = body.get("patient_id")
    request_id = getattr(context, "aws_request_id", "local")
    t0 = time.time()

    # ① 시작 로그
    _record_phase_log(
        session_id=session_id, patient_id=patient_id,
        phase_name="phase3", phase_step="score_all", status="started",
        started_at=t0,
        lambda_function=context.function_name,
        lambda_request_id=request_id,
        input_summary={"lab_count": len(body.get("patient_lab_findings", []))},
    )

    try:
        results = _SCORER.score_all(...)
    except Exception as e:
        # ② 실패 로그 (스택 트레이스 포함 자동 저장)
        _record_phase_log(
            session_id=session_id, patient_id=patient_id,
            phase_name="phase3", phase_step="score_all", status="failed",
            started_at=t0,
            lambda_function=context.function_name,
            lambda_request_id=request_id,
            error=e,
        )
        _mark_session_failed(session_id, str(e), phase=3)  # 세션 상태도 함께
        return _server_error(e)

    # ③ 성공 로그
    _record_phase_log(
        session_id=session_id, patient_id=patient_id,
        phase_name="phase3", phase_step="score_all", status="succeeded",
        started_at=t0,
        lambda_function=context.function_name,
        lambda_request_id=request_id,
        output_summary={"results_count": len(results)},
    )
    return _ok(_serialize_results(results, ...))
```

→ 이렇게 하면 **diagnosis_session(상태) + phase_execution_log(상세)** 가 함께 남아 운영팀이 둘 다 활용 가능합니다.

---

### 📋 Phase 3 (`Phase_3/infra/aws/phase3/lambda/handler.py`) 수정안

**(A) import 및 DB 헬퍼 추가** (파일 상단 import 블록 아래)

```python
import boto3

try:
    import psycopg2
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

DB_HOST = "patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com"
DB_NAME = "soopul"
DB_USER = "app_user"
DB_SECRET_ID = "soopul/aurora/app-user"

def _get_db_conn():
    if not DB_AVAILABLE:
        return None
    try:
        sm = boto3.client("secretsmanager", region_name="ap-northeast-2")
        secret_str = sm.get_secret_value(SecretId=DB_SECRET_ID)["SecretString"]
        pwd = json.loads(secret_str).get("password", secret_str)
        return psycopg2.connect(
            host=DB_HOST, port=5432, database=DB_NAME, user=DB_USER, password=pwd,
            options="-c search_path=soopulai", connect_timeout=10,
        )
    except Exception as e:
        logger.warning("DB 연결 실패: %s", e)
        return None

def _mark_session_failed(session_id: str, error_msg: str, phase: int = 3):
    if not session_id:
        return
    conn = _get_db_conn()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE diagnosis_session
            SET status='failed', error_message=%s, completed_at=NOW(), current_phase=%s
            WHERE session_id=%s
        """, (str(error_msg)[:1000], phase, session_id))
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()

def _set_session_running(session_id: str, phase: int = 3):
    if not session_id:
        return
    conn = _get_db_conn()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE diagnosis_session
            SET status='running', current_phase=%s
            WHERE session_id=%s
        """, (phase, session_id))
        conn.commit()
    except Exception as e:
        logger.warning("세션 상태(running) 업데이트 실패: %s", e)
    finally:
        conn.close()
```

**(B) `lambda_handler` 본체 수정** — `session_id` 추출 + 두 군데 except에서 DB 업데이트

```python
def lambda_handler(event: dict, context) -> dict:
    request_id = getattr(context, "aws_request_id", "local")
    session_id = event.get("session_id")  # ★ 추가

    path = event.get("path") or event.get("rawPath") or ""
    if path.endswith("/health"):
        return _ok({"status": "ok", "registry_loaded": _SCORER is not None})

    try:
        _ensure_initialized()
    except Exception as e:
        _mark_session_failed(session_id, f"init: {e}", phase=3)  # ★ 추가
        return _server_error(e)

    body = event.get("body") if isinstance(event.get("body"), (str, dict)) else event
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return _bad("body is not valid JSON")
    if not isinstance(body, dict):
        return _bad("expected JSON object")

    # body 내부에 session_id가 들어오는 경우도 지원
    session_id = session_id or body.get("session_id")  # ★ 추가
    _set_session_running(session_id, phase=3)           # ★ 추가 (옵션)

    try:
        lab_findings = [_to_lab_finding(x) for x in body.get("patient_lab_findings", [])]
        # ... (이하 동일)
    except (TypeError, ValueError) as e:
        return _bad(f"input dataclass construction failed: {e}")

    t0 = time.perf_counter()
    try:
        results = _SCORER.score_all(...)
    except Exception as e:
        _mark_session_failed(session_id, f"score_all: {e}", phase=3)  # ★ 추가
        return _server_error(e)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return _ok(_serialize_results(results, elapsed_ms, request_id))
```

---

### 📋 Phase 4 (`Phase_4/infra/aws/phase4/lambda/handler.py`) 수정안

Phase 3와 구조가 거의 동일하므로 같은 헬퍼(`_get_db_conn`, `_mark_session_failed`, `_set_session_running`)를 추가하고, `lambda_handler` 안에서:

- 상단에서 `session_id = event.get("session_id") or body.get("session_id")`
- `_get_verifier(mode)` 실패, `verifier.verify(...)` 실패의 두 `except Exception`에서 `_mark_session_failed(session_id, f"phase4: {e}", phase=4)` 호출

**Phase 4 핵심 변경 라인 (handler.py:117-128 부근):**

```python
try:
    verifier = _get_verifier(mode)
    input_data = _to_phase4_input(body)
    hp_id_to_term = body.get("hp_id_to_term")
except Exception as e:
    _mark_session_failed(session_id, f"verifier init: {e}", phase=4)  # ★
    return _server_error(e)

t0 = time.perf_counter()
try:
    result = verifier.verify(input_data, hp_id_to_term=hp_id_to_term)
except Exception as e:
    _mark_session_failed(session_id, f"verify: {e}", phase=4)  # ★
    return _server_error(e)
```

---

### 📋 Phase 5 (`Phase_5/infra/aws/phase5/lambda/handler.py`) 수정안

Phase 5는 RAG 내부에서 이미 `_mark_session_failed`가 동작합니다. **다만 cold-start/import 단계 실패는 DB에 안 남습니다.** 안전망만 추가합니다.

**핵심 변경 (handler.py:174-205 outer try/except):**

```python
try:
    rag = _get_rag_system()         # ← 여기서 import 실패 가능
    result = rag.run_with_session_id(session_id)
    # ...
except ValueError as exc:
    logger.warning(...)
    return _bad(str(exc), request_id=request_id)

except Exception as exc:
    elapsed_ms = (time.monotonic() - t_start) * 1000
    logger.exception(...)
    _mark_session_failed_safety_net(session_id, f"phase5: {exc}", phase=5)  # ★ 추가
    return _server_error(
        f"Internal error: {type(exc).__name__}",
        request_id=request_id,
    )
```

여기서 `_mark_session_failed_safety_net`은 RAG 인스턴스 없이도 DB에 직접 쓸 수 있어야 하므로 Phase 3/4에 넣은 모듈-레벨 `_mark_session_failed`를 그대로 사용하면 됩니다. **이미 RAG 내부에서 마킹됐을 수 있으니 `status='failed'`인 행을 다시 `UPDATE`해도 멱등(idempotent)** 입니다.

---

## 4. Lambda Layer (`requirements.txt`) 점검

**확인된 사실:**
- Phase 5 Layer: 이미 `psycopg2-binary` 포함 ✅ (`layer/build_layer.sh` 주석 참조)
- Phase 3, 4 Layer: 명시 없음 → **`layer/build_layer.sh`에 `psycopg2-binary` 추가 필요** (boto3는 Lambda 런타임 기본 제공)

---

## 5. 단계별 적용 순서 (권장)

| 단계 | 작업 | 영향 |
|---|---|---|
| 1️⃣ | Phase 3/4 `build_layer.sh` 에 `psycopg2-binary` 추가 → Layer 재빌드 | Lambda 재배포 필요 |
| 2️⃣ | 호출자(Step Functions/프론트)가 모든 Phase 3/4/5 호출에 `session_id` 전달하도록 수정 | 프론트 호출 페이로드 변경 |
| 3️⃣ | Phase 3 `handler.py` 패치 → SAM `sam build && sam deploy` | 단독 배포 가능 |
| 4️⃣ | Phase 4 `handler.py` 패치 → SAM 배포 | 단독 배포 가능 |
| 5️⃣ | Phase 5 `handler.py` 안전망 추가 → SAM 배포 | 단독 배포 가능 |
| 6️⃣ | 검증: 일부러 잘못된 입력으로 호출 → `SELECT session_id, status, error_message, current_phase FROM soopulai.diagnosis_session WHERE session_id=...` | RAG `check_sessions.py` 재사용 가능 |

---

## 6. 적용 후 달라지는 것 — Before / After

### 시나리오 A: 환자 P-12345의 진단이 실패했다

**Before (현재)**
```
운영자: "P-12345 진단이 안 돼요"
개발자: (CloudWatch 열고) → Phase 1 로그그룹 검색 → 없음
        → Phase 2 로그그룹 검색 → 없음
        → Phase 3 로그그룹 검색 → KeyError 발견? 근데 이게 P-12345 맞나?
        → request_id로 다시 추적 → 약 15~30분 소요
```

**After (적용 후)**
```sql
SELECT current_phase, status, error_message, completed_at
FROM soopulai.diagnosis_session WHERE session_id='P-12345';

-- 결과 (5초)
-- current_phase | status | error_message               | completed_at
-- 3             | failed | score_all: KeyError 'icd10' | 2026-05-17 14:23:01
```
→ Phase 3에서 입력 데이터의 `icd10` 키 누락이 원인. 즉시 수정 가능.

---

### 시나리오 B: 프론트엔드에 실시간 진행 상태를 보여주고 싶다

**Before (현재)**
- Phase 5(RAG)만 status를 업데이트 → 나머지 단계는 "분석 중..."만 빙글빙글 돌릴 수밖에 없음
- 사용자가 Phase 3에서 실패해도 "그냥 안 되네" 메시지뿐

**After (적용 후)**
프론트엔드가 1초마다 `current_phase`를 폴링하거나 WebSocket으로 받으면:
```
[●●●○○]  Phase 3 / 5 진행 중 — 영상 분석
[●●●●○]  Phase 4 / 5 진행 중 — LLM 검증
[●●●●●]  완료 — 보고서 보기 →

또는

[●●●✕]   Phase 3에서 실패: 영상 데이터 형식 오류
         [다시 시도]  [관리자에게 문의]
```

---

### 시나리오 C: "오늘 진단 성공률이 어떻게 되나요?" 라는 임원 질문

**Before**
- CloudWatch Insights로 5개 Lambda 각각 쿼리 → 합산 → 1시간 소요

**After**
```sql
SELECT
  status,
  COUNT(*) AS cnt,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM soopulai.diagnosis_session
WHERE created_at >= CURRENT_DATE
GROUP BY status;

-- status    | cnt | pct
-- completed | 142 | 87.7
-- failed    |  18 | 11.1
-- running   |   2 |  1.2
```
→ 5초 만에 답변. Grafana/Metabase 대시보드 연동도 가능.

---

### 시나리오 D: 장애가 났는데 어느 Phase의 어떤 에러가 가장 흔한지 알고 싶다

**After (전에는 불가능했던 분석)**
```sql
SELECT
  current_phase,
  SUBSTRING(error_message FROM '^[^:]+') AS error_type,
  COUNT(*) AS occurrences
FROM soopulai.diagnosis_session
WHERE status='failed' AND completed_at > NOW() - INTERVAL '24 hours'
GROUP BY 1, 2
ORDER BY 3 DESC LIMIT 10;

-- current_phase | error_type                  | occurrences
-- 3             | score_all                   | 12
-- 4             | verifier init               |  5
-- 5             | phase5                      |  3
-- 3             | init                        |  1
```
→ "Phase 3의 score_all이 가장 많이 실패하니 거기부터 패치하자" 같은 우선순위 결정 가능.

---

### 시나리오 E: Step Functions 워크플로우에서 자동 복구

**Before**
- 중간 단계 실패 → Step Functions 전체 종료 → 처음부터 재실행 (비용 낭비)

**After**
- 다음 단계가 DB에서 `current_phase`를 읽어 "Phase 3까지는 성공했네, Phase 4부터 재시작" 가능
- 자동 알람 Lambda가 `status='failed'`인 세션을 매시간 스캔 → Slack 통보 → 재시도 큐에 등록

---

### 종합 효과 요약

| 영역 | Before | After |
|---|---|---|
| 장애 원인 추적 | 15~30분 (CloudWatch 검색) | 5초 (SQL 한 줄) |
| 사용자 UX | "분석 중..." 무한 로딩 | 실시간 단계별 진행률 + 명확한 실패 안내 |
| 운영 가시성 | 로그그룹 5개 따로 봐야 함 | 테이블 1개로 통합 |
| 자동 알람 | 불가능 | SQL 임계값으로 즉시 가능 |
| 재시도 전략 | 처음부터 다시 | 실패 지점부터 재개 |
| 신규 팀원 온보딩 | "3가지 디버깅 절차 외워야 함" | "diagnosis_session 한 곳만 보면 됨" |
| CS 응대 | "확인 후 연락드리겠습니다" | 즉시 답변 가능 |

---

### 잠재적 부작용 (정직하게)

1. **DB 부하 미세 증가** — 매 요청마다 INSERT/UPDATE 1~2회 추가. 다만 Aurora는 이 정도 트래픽은 무시할 수준. 1만 요청/일 기준 일 2만 UPDATE → 부하 < 1%.
2. **Secrets Manager 호출 비용** — Lambda cold start마다 1회. 비용 약 $0.05/만 회. 무시 가능.
3. **VPC Lambda cold start 지연** — Phase 3/4가 현재 VPC 밖이라면 VPC 진입 시 ENI 초기화로 cold start 1~2초 증가. **이미 RAG/Phase 5가 VPC 안이라면 동일 설정 복사로 추가 지연 없음**.
4. **에러 메시지 1000자 제한** — 긴 스택 트레이스는 잘림. 풀 트레이스는 여전히 CloudWatch에 있으니 보완재로 활용.
5. **부분 실패 처리 필요** — DB 업데이트 자체가 실패하면? RAG 패턴처럼 `except Exception: pass`로 삼키고 CloudWatch에는 남음 → 본 응답에는 영향 없음.

---

## 7. 부가 권장 (선택)

- **Phase 1, 2도 동일 패턴 적용 가능** — 둘은 Lambda가 아닌 스크립트형이지만, `session_id`를 인자로 받아 같은 `_mark_session_failed`를 호출하도록 추가하면 전체 파이프라인이 일관된 에러 추적을 가집니다.
- **에러 디테일 보존이 더 필요하면** `phase_error_log` 같은 별도 테이블을 두고 `session_id, phase, error_class, error_message, traceback, occurred_at`을 INSERT 하는 패턴이 RAG의 단일 컬럼(`error_message`) 방식보다 디버깅에 유리합니다. (현 구조 유지 → 빠른 출시 / 신규 테이블 → 더 견고)
- **VPC/Secrets Manager 권한** — Phase 3, 4 Lambda 실행 역할에 `secretsmanager:GetSecretValue` 및 RDS VPC 접근 권한이 없으면 추가 필요. RAG/Phase 5 정책을 참고해서 동일 권한을 부여하면 됩니다.

---

## 부록 — 참조 파일 경로

```
s3_clone/
├── Phase_1/symptom_llm_4.py                              # _log_error (파일만)
├── Phase_2/soo_net_5.py                                  # _log_error (파일만)
├── Phase_3/infra/aws/phase3/lambda/handler.py            # 수정 대상
├── Phase_3/infra/aws/phase3/layer/build_layer.sh         # psycopg2 추가 필요
├── Phase_4/infra/aws/phase4/lambda/handler.py            # 수정 대상
├── Phase_4/infra/aws/phase4/layer/build_layer.sh         # psycopg2 추가 필요
├── Phase_5/infra/aws/phase5/lambda/handler.py            # 안전망 추가
├── Phase_5/infra/aws/phase5/lambda/requirements.txt      # psycopg2 이미 OK
├── RAG/rag_llm_3.py                                      # 참조 표준 패턴
├── RAG/check_sessions.py                                 # 검증용 (재사용 가능)
└── RAG/check_schema.py                                   # 스키마 확인용
```
