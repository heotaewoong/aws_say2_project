"""phase_execution_log 테이블 존재 여부 및 현재 행 수 확인.

사용법 (RAG/Phase 5가 도는 VPC 내부 — EC2/Lambda/Cloud9 등에서 실행):
    python check_phase_log.py

기대 결과:
    - 테이블 존재 → 컬럼 목록 + 현재 행 수 출력
    - 테이블 없음 → "❌ phase_execution_log 테이블 없음" → apply_phase_log_ddl.py 실행 권장
"""
import json
import boto3
import psycopg2

DB_HOST = "patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com"
DB_NAME = "soopul"
DB_USER = "app_user"
DB_SECRET_ID = "soopul/aurora/app-user"

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
cur = conn.cursor()

# 1) 테이블 존재 여부
cur.execute("""
    SELECT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema='soopulai' AND table_name='phase_execution_log'
    )
""")
exists = cur.fetchone()[0]

if not exists:
    print("❌ phase_execution_log 테이블이 존재하지 않습니다.")
    print("   → apply_phase_log_ddl.py 를 실행해 테이블을 생성하세요.")
    conn.close()
    raise SystemExit(1)

print("✅ phase_execution_log 테이블 존재")

# 2) 컬럼 목록
cur.execute("""
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_schema='soopulai' AND table_name='phase_execution_log'
    ORDER BY ordinal_position
""")
print("\n=== 컬럼 ===")
for r in cur.fetchall():
    print(f"  {r[0]:25s} {r[1]:30s} nullable={r[2]}")

# 3) 현재 행 수 + 최근 5개
cur.execute("SELECT COUNT(*) FROM soopulai.phase_execution_log")
total = cur.fetchone()[0]
print(f"\n현재 총 행 수: {total}")

if total > 0:
    cur.execute("""
        SELECT phase_name, status, started_at, error_code
        FROM soopulai.phase_execution_log
        ORDER BY started_at DESC LIMIT 5
    """)
    print("\n=== 최근 5개 ===")
    for r in cur.fetchall():
        print(f"  {r[0]:10s} {r[1]:10s} {r[2]} {r[3] or ''}")

# 4) View 존재 여부
cur.execute("""
    SELECT viewname FROM pg_views
    WHERE schemaname='soopulai'
      AND viewname IN ('recent_errors', 'phase_success_rates_24h')
""")
views = [r[0] for r in cur.fetchall()]
print(f"\n사전 정의 View: {views if views else '없음'}")

cur.close()
conn.close()
