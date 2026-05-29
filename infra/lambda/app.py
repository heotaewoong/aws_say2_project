"""Phase 5 (RAG) Lambda 핸들러

호출 방식 (Step Functions 또는 직접 테스트):
    {
        "session_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    }

처리 흐름:
    1) DB에서 이전 Phase 결과 읽기 (Phase1~4)
    2) 외부 API RAG 수집 (PubMed, Orphanet, Monarch, PubCaseFinder, ClinicalTrials)
    3) Bedrock (claude-3-5-sonnet-20241022-v2:0) 최종 보고서 생성
    4) final_report 테이블에 저장, diagnosis_session.status = 'completed'

반환 (Step Functions / 직접 호출 공용):
    {
        "statusCode": 200,
        "body": "{\"session_id\": \"...\", \"status\": \"completed\"}"
    }
    에러 시:
    {
        "statusCode": 500,
        "body": "{\"error\": \"...\"}"
    }
"""
import json

from rag_llm_3 import RareLinkHybridDualRAG

# Cold start 최적화 — Lambda 컨테이너 재사용 시 재초기화 생략
_rag_system = None


def _get_rag_system():
    global _rag_system
    if _rag_system is None:
        # orphadata_csv_path=None: Phase5는 로컬 CSV 불필요, DB에서 직접 읽음
        _rag_system = RareLinkHybridDualRAG()
    return _rag_system


def lambda_handler(event, context):
    try:
        # Step Functions은 dict로 직접 전달, API Gateway는 body가 JSON string
        body = event
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])

        session_id = body.get("session_id")
        if not session_id:
            return _err(400, "session_id required")

        rag = _get_rag_system()
        final_report = rag.run_with_session_id(session_id)

        return _ok({
            "session_id": session_id,
            "status": "completed",
        })

    except Exception as e:
        return _err(500, str(e))


def _ok(payload: dict):
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(payload, ensure_ascii=False),
    }


def _err(code: int, msg: str):
    return {
        "statusCode": code,
        "headers": {"Access-Control-Allow-Origin": "*"},
        "body": json.dumps({"error": msg}),
    }
