"""phase_execution_log (+ system_health_metric + 뷰 2개) DDL 적용.

전제: check_phase_log.py 실행 결과 테이블이 없을 때만 실행.
DDL 출처: s3://say2-2team-bucket/database/system-log-schema-ddl.sql

사용법 (VPC 내부 환경에서):
    python apply_phase_log_ddl.py

idempotent: CREATE TABLE IF NOT EXISTS 라서 여러 번 실행해도 안전.
주의: app_user 권한에 CREATE 가 있어야 함 — 없으면 DDL 적용용 superuser 계정 사용 필요.
"""
import json
import boto3
import psycopg2

DB_HOST = "patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com"
DB_NAME = "soopul"
DB_USER = "app_user"
DB_SECRET_ID = "soopul/aurora/app-user"

DDL = """
SET search_path TO soopulai;

CREATE TABLE IF NOT EXISTS phase_execution_log (
    log_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          UUID REFERENCES diagnosis_session(session_id),
    patient_id          VARCHAR(64),
    phase_name          VARCHAR(16) NOT NULL,
    phase_step          VARCHAR(64),
    status              VARCHAR(16) NOT NULL,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    duration_ms         INT,
    lambda_function     VARCHAR(128),
    lambda_request_id   VARCHAR(64),
    lambda_memory_mb    INT,
    lambda_billed_ms    INT,
    input_summary       JSONB,
    output_summary      JSONB,
    error_code          VARCHAR(64),
    error_message       TEXT,
    error_stacktrace    TEXT,
    error_category      VARCHAR(32),
    retry_count         INT DEFAULT 0,
    retry_of_log_id     UUID,
    external_calls      JSONB,
    model_versions      JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_log_session ON phase_execution_log (session_id, phase_name);
CREATE INDEX IF NOT EXISTS idx_log_patient ON phase_execution_log (patient_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_log_status  ON phase_execution_log (status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_log_phase   ON phase_execution_log (phase_name, status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_log_error   ON phase_execution_log (error_category, error_code) WHERE status = 'failed';
CREATE INDEX IF NOT EXISTS idx_log_time    ON phase_execution_log (started_at DESC);

CREATE OR REPLACE VIEW recent_errors AS
SELECT log_id, session_id, patient_id, phase_name, phase_step,
       error_code, error_category, error_message, started_at, duration_ms,
       lambda_function, retry_count
FROM phase_execution_log
WHERE status = 'failed'
ORDER BY started_at DESC
LIMIT 100;

CREATE OR REPLACE VIEW phase_success_rates_24h AS
SELECT phase_name,
       COUNT(*) AS total_executions,
       COUNT(*) FILTER (WHERE status = 'succeeded') AS success_count,
       COUNT(*) FILTER (WHERE status = 'failed') AS failed_count,
       ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'succeeded')
             / NULLIF(COUNT(*), 0), 2) AS success_rate_pct,
       ROUND(AVG(duration_ms) FILTER (WHERE status = 'succeeded'), 0) AS avg_duration_ms,
       MAX(duration_ms) FILTER (WHERE status = 'succeeded') AS max_duration_ms
FROM phase_execution_log
WHERE started_at > NOW() - INTERVAL '24 hours'
  AND status IN ('succeeded', 'failed')
GROUP BY phase_name
ORDER BY phase_name;
"""

sm = boto3.client("secretsmanager", region_name="ap-northeast-2")
secret_str = sm.get_secret_value(SecretId=DB_SECRET_ID)["SecretString"]
try:
    pwd = json.loads(secret_str)["password"]
except (json.JSONDecodeError, KeyError):
    pwd = secret_str

conn = psycopg2.connect(
    host=DB_HOST, port=5432, database=DB_NAME, user=DB_USER, password=pwd,
    options="-c search_path=soopulai", connect_timeout=10,
)
try:
    cur = conn.cursor()
    cur.execute(DDL)
    conn.commit()
    print("✅ DDL 적용 완료 (phase_execution_log + 인덱스 + 뷰 2개)")
except psycopg2.errors.InsufficientPrivilege as e:
    print(f"❌ 권한 부족: {e}")
    print("   → DBA에게 app_user 에게 CREATE 권한 부여 요청 필요")
    conn.rollback()
finally:
    conn.close()
