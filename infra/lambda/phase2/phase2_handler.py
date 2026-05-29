"""
Phase 2 Lambda — X-ray → 14-class HPO 변환

호출 방식 (Step Functions 또는 직접 호출):
    Event: {
        "xray_s3_key": "uploads/patient123/xray.jpg",
        "threshold": 0.3   # optional
    }

처리:
    1) S3에서 X-ray 이미지 읽기 (바이트 스트림)
    2) SageMaker Endpoint 호출 (VPC Endpoint 경유 → NAT 불필요)
    3) 14-class 확률값 파싱
    4) 임계값 이상 라벨 → HPO 코드 추출
    5) 결과 S3 저장 + 반환

반환:
    {
        "predictions": { label: {probability, hpo_code} },
        "positive_hpos": ["HP:0002202", ...],
        "xray_detail":   {...},   # rag_pipeline.py가 사용하는 포맷
        "result_s3_key": "results/xxx.json"
    }
"""
import base64
import json
import os
import uuid

import boto3

# ── 환경변수 (02-phase2-xray.yaml에서 주입) ──
ENDPOINT   = os.environ["SAGEMAKER_ENDPOINT"]
BUCKET     = os.environ["S3_BUCKET"]
THRESHOLD  = float(os.environ.get("XRAY_THRESHOLD", "0.3"))

# ── AWS 클라이언트 (cold start 시 1회만 초기화) ──
s3        = boto3.client("s3")
runtime   = boto3.client("sagemaker-runtime")


# 14-class 라벨 ↔ HPO 매핑 (soo_net.py와 동일하게 유지)
HPO_MAP = {
    "Atelectasis":                  "HP:0002095",
    "Cardiomegaly":                 "HP:0001640",
    "Consolidation":                "HP:0002113",
    "Edema":                        "HP:0002111",
    "Enlarged Cardiomediastinum":   "HP:0034251",
    "Fracture":                     "HP:0002757",
    "Lung Lesion":                  "HP:0025000",
    "Lung Opacity":                 "HP:0002088",
    "No Finding":                   "Normal (N/A)",
    "Pleural Effusion":             "HP:0002202",
    "Pleural Other":                "HP:0002102",
    "Pneumonia":                    "HP:0002090",
    "Pneumothorax":                 "HP:0002107",
    "Support Devices":              "Device (N/A)",
}


def lambda_handler(event, context):
    try:
        # 1) 입력 파싱
        xray_s3_key = event.get("xray_s3_key")
        if not xray_s3_key:
            return {"statusCode": 400, "body": json.dumps({"error": "xray_s3_key required"})}

        threshold = float(event.get("threshold", THRESHOLD))

        # 2) S3에서 X-ray 이미지 로드
        obj = s3.get_object(Bucket=BUCKET, Key=xray_s3_key)
        image_bytes = obj["Body"].read()

        # 3) SageMaker Endpoint 호출
        #    inference.py의 input_fn이 application/x-image 지원
        resp = runtime.invoke_endpoint(
            EndpointName=ENDPOINT,
            ContentType="application/x-image",
            Body=image_bytes,
        )
        predictions = json.loads(resp["Body"].read())
        # 반환 포맷:
        #   { "Atelectasis": {"probability": 0.12, "hpo_code": "HP:0002095"}, ... }

        # 4) HPO 추출 (임계값 이상 + N/A 제외)
        positive_hpos = []
        xray_detail   = {}
        for label, info in predictions.items():
            prob = float(info["probability"])
            hpo  = info.get("hpo_code") or HPO_MAP.get(label, "N/A")
            xray_detail[label] = [prob, hpo]  # rag_pipeline.py 호환 tuple 포맷
            if prob >= threshold and "N/A" not in hpo:
                positive_hpos.append(hpo)

        # 5) 결과 S3 저장
        result_key = f"Phase_2/results/phase2/{uuid.uuid4().hex}.json"
        result_payload = {
            "xray_s3_key":    xray_s3_key,
            "threshold":      threshold,
            "predictions":    predictions,
            "positive_hpos":  positive_hpos,
            "xray_detail":    xray_detail,
        }
        s3.put_object(
            Bucket=BUCKET,
            Key=result_key,
            Body=json.dumps(result_payload, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        result_payload["result_s3_key"] = result_key

        return {
            "statusCode": 200,
            "body": json.dumps(result_payload, ensure_ascii=False),
        }

    except runtime.exceptions.ModelError as e:
        return {"statusCode": 502, "body": json.dumps({"error": f"SageMaker 추론 실패: {e}"})}
    except s3.exceptions.NoSuchKey:
        return {"statusCode": 404, "body": json.dumps({"error": f"S3 키 없음: {xray_s3_key}"})}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
