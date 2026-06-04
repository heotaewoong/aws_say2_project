import json, boto3, psycopg2

sm = boto3.client("secretsmanager", region_name="ap-northeast-2")
pwd = sm.get_secret_value(SecretId="soopul/aurora/app-user")["SecretString"]
conn = psycopg2.connect(
    host="patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com",
    port=5432, database="soopul", user="app_user", password=pwd,
    options="-c search_path=soopulai", connect_timeout=10
)
cur = conn.cursor()

# diagnosis_session 스키마 확인
cur.execute("""
    SELECT column_name, data_type, column_default, is_nullable
    FROM information_schema.columns
    WHERE table_schema='soopulai' AND table_name='diagnosis_session'
    ORDER BY ordinal_position
""")
print("=== diagnosis_session 컬럼 ===")
for r in cur.fetchall():
    print(f"  {r[0]:30s} {r[1]:25s} nullable={r[3]} default={r[2]}")

# 기존 세션 존재 여부
cur.execute("SELECT COUNT(*) FROM soopulai.diagnosis_session")
count = cur.fetchone()[0]
print(f"\n기존 diagnosis_session 레코드 수: {count}")

if count > 0:
    cur.execute("SELECT session_id, created_at FROM soopulai.diagnosis_session ORDER BY created_at DESC LIMIT 3")
    print("최근 3개:")
    for r in cur.fetchall():
        print(f"  {r[0]}  {r[1]}")

cur.close()
conn.close()
