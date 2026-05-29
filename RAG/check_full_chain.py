import json, boto3, psycopg2

sm = boto3.client("secretsmanager", region_name="ap-northeast-2")
pwd = sm.get_secret_value(SecretId="soopul/aurora/app-user")["SecretString"]
conn = psycopg2.connect(
    host="patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com",
    port=5432, database="soopul", user="app_user", password=pwd,
    options="-c search_path=soopulai", connect_timeout=10
)
cur = conn.cursor()

# pg_catalog로 FK 체인 확인
cur.execute("""
    SELECT
        tc.table_name,
        kcu.column_name,
        ccu.table_name AS ref_table,
        ccu.column_name AS ref_col
    FROM pg_constraint c
    JOIN pg_class t ON t.oid = c.conrelid
    JOIN pg_namespace n ON n.oid = t.relnamespace
    JOIN information_schema.table_constraints tc ON tc.constraint_name = c.conname
        AND tc.table_schema = n.nspname
    JOIN information_schema.key_column_usage kcu ON kcu.constraint_name = c.conname
        AND kcu.table_schema = n.nspname
    JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = c.conname
    WHERE c.contype = 'f'
      AND n.nspname = 'soopulai'
    ORDER BY tc.table_name
""")
print("=== soopulai FK 체인 ===")
for r in cur.fetchall():
    print(f"  {r[0]}.{r[1]} → {r[2]}.{r[3]}")

# raw_emr_bundle 스키마
cur.execute("""
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_schema='soopulai' AND table_name='raw_emr_bundle'
    ORDER BY ordinal_position
""")
rows = cur.fetchall()
if rows:
    print("\n=== raw_emr_bundle 컬럼 ===")
    for r in rows:
        print(f"  {r[0]:30s} {r[1]:25s} nullable={r[2]}")
    cur.execute("SELECT COUNT(*) FROM soopulai.raw_emr_bundle")
    print(f"  레코드 수: {cur.fetchone()[0]}")
else:
    print("\nraw_emr_bundle 테이블 없음")

cur.close()
conn.close()
