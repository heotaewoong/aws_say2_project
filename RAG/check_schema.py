import json, boto3, psycopg2
sm = boto3.client("secretsmanager", region_name="ap-northeast-2")
pwd = sm.get_secret_value(SecretId="soopul/aurora/app-user")["SecretString"]
conn = psycopg2.connect(
    host="patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com",
    port=5432, database="soopul", user="app_user", password=pwd,
    options="-c search_path=soopulai", connect_timeout=10
)
cur = conn.cursor()
cur.execute("""
    SELECT column_name, data_type, udt_name
    FROM information_schema.columns
    WHERE table_schema='soopulai' AND table_name='final_report'
    ORDER BY ordinal_position
""")
rows = cur.fetchall()
for r in rows:
    print(f"{r[0]:35s} {r[1]:20s} {r[2]}")
cur.close()
conn.close()
