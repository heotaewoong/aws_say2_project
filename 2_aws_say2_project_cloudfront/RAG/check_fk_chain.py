import json, boto3, psycopg2

sm = boto3.client("secretsmanager", region_name="ap-northeast-2")
pwd = sm.get_secret_value(SecretId="soopul/aurora/app-user")["SecretString"]
conn = psycopg2.connect(
    host="patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com",
    port=5432, database="soopul", user="app_user", password=pwd,
    options="-c search_path=soopulai", connect_timeout=10
)
cur = conn.cursor()

# patient_profile 스키마
cur.execute("""
    SELECT column_name, data_type, is_nullable, column_default
    FROM information_schema.columns
    WHERE table_schema='soopulai' AND table_name='patient_profile'
    ORDER BY ordinal_position
""")
print("=== patient_profile 컬럼 ===")
for r in cur.fetchall():
    print(f"  {r[0]:30s} {r[1]:25s} nullable={r[2]}")

cur.execute("SELECT COUNT(*) FROM soopulai.patient_profile")
print(f"\n기존 patient_profile 레코드 수: {cur.fetchone()[0]}")

# 전체 FK 체인 확인
cur.execute("""
    SELECT tc.table_name, kcu.column_name, ccu.table_name AS foreign_table, ccu.column_name AS foreign_column
    FROM information_schema.table_constraints AS tc
    JOIN information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name
    JOIN information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name
    WHERE tc.constraint_type = 'FOREIGN KEY'
      AND tc.table_schema = 'soopulai'
    ORDER BY tc.table_name
""")
print("\n=== FK 체인 전체 ===")
for r in cur.fetchall():
    print(f"  {r[0]}.{r[1]} → {r[2]}.{r[3]}")

cur.close()
conn.close()
