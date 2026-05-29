import json, boto3, sys, uuid

try:
    import psycopg2
    from psycopg2.extras import Json as PgJson
    print("✅ psycopg2 + PgJson import 성공")
except ImportError as e:
    print(f"❌ psycopg2 import 실패: {e}"); sys.exit(1)

sm = boto3.client("secretsmanager", region_name="ap-northeast-2")
secret_string = sm.get_secret_value(SecretId="soopul/aurora/app-user")["SecretString"]
try:
    secret = json.loads(secret_string)
    username, password = secret["username"], secret["password"]
except (json.JSONDecodeError, KeyError):
    username, password = "app_user", secret_string

conn = psycopg2.connect(
    host="patient-db-cluster.cluster-cxmiyawwwhbt.ap-northeast-2.rds.amazonaws.com",
    port=5432, database="soopul", user=username, password=password,
    options="-c search_path=soopulai", connect_timeout=10,
)
print(f"✅ Aurora 연결 성공 (user={username})")

test_patient_id = "RAG_TEST_이환자"
test_session_id = str(uuid.uuid4())
test_bundle_id = str(uuid.uuid4())
fhir_stub = json.dumps({"resourceType": "Bundle", "id": test_bundle_id, "source": "rag_llm_3"})

try:
    cur = conn.cursor()

    # 1. raw_emr_bundle (FK 루트)
    cur.execute("""
        INSERT INTO soopulai.raw_emr_bundle
            (bundle_id, patient_id, source_system, fhir_bundle_json, fetched_at)
        VALUES (%s::uuid, %s, 'rag_llm_3', %s::jsonb, NOW())
        ON CONFLICT (bundle_id) DO NOTHING
    """, (test_bundle_id, test_patient_id, fhir_stub))
    print("✅ raw_emr_bundle INSERT 성공")

    # 2. patient_profile
    cur.execute("""
        INSERT INTO soopulai.patient_profile
            (patient_id, bundle_id, name_display, age_years, sex, created_at, updated_at)
        VALUES (%s, %s::uuid, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (patient_id) DO NOTHING
    """, (test_patient_id, test_bundle_id, "이환자", 42, "M"))
    print("✅ patient_profile INSERT 성공")

    # 3. diagnosis_session
    cur.execute("""
        INSERT INTO soopulai.diagnosis_session
            (session_id, patient_id, bundle_id, initiated_by, status, current_phase)
        VALUES (%s, %s, %s::uuid, 'rag_llm_3', 'completed', 3)
        ON CONFLICT (session_id) DO NOTHING
    """, (test_session_id, test_patient_id, test_bundle_id))
    print("✅ diagnosis_session INSERT 성공")

    # 4. final_report
    cur.execute("""
        INSERT INTO soopulai.final_report (
            session_id, diagnosis_json, markdown_report,
            rag_citations, rag_apis_used, self_check,
            s3_uri_pdf, s3_uri_html, pdf_sha256, pdf_size_bytes, pdf_generated_at,
            external_api_call_summary, llm_model, total_inference_time_ms
        ) VALUES (
            %(session_id)s, %(diagnosis_json)s, %(markdown_report)s,
            %(rag_citations)s, %(rag_apis_used)s, %(self_check)s,
            NULL, NULL, NULL, NULL, NOW(),
            %(external_api_call_summary)s, %(llm_model)s, NULL
        ) RETURNING generated_at
    """, {
        "session_id": test_session_id,
        "diagnosis_json": PgJson({"test": True, "items": [1, 2, 3]}),
        "markdown_report": "## 테스트 보고서\n내용입니다.",
        "rag_citations": PgJson([{"pmid": "12345678", "title": "test"}]),
        "rag_apis_used": ["PubMed", "Orphanet"],
        "self_check": PgJson({"score": 0.85}),
        "external_api_call_summary": PgJson([{"api": "PubMed"}]),
        "llm_model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    })
    generated_at = cur.fetchone()[0]
    conn.commit()
    print(f"\n🎉 ALL INSERT 성공! generated_at={generated_at}")
    print(f"   session_id={test_session_id}")

    # 정리 (FK 역순)
    cur.execute("DELETE FROM soopulai.final_report WHERE session_id = %s", (test_session_id,))
    cur.execute("DELETE FROM soopulai.diagnosis_session WHERE session_id = %s", (test_session_id,))
    cur.execute("DELETE FROM soopulai.patient_profile WHERE patient_id = %s", (test_patient_id,))
    cur.execute("DELETE FROM soopulai.raw_emr_bundle WHERE bundle_id = %s::uuid", (test_bundle_id,))
    conn.commit()
    print("✅ 테스트 데이터 정리 완료")

except Exception as e:
    import traceback; traceback.print_exc()
    print(f"❌ 실패: {e}")
finally:
    cur.close(); conn.close()
