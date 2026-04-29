# bedrock_extractor.py
# AWS Bedrock Claude 기반 임상 소견 텍스트 → HPO 코드 추출
# 기존 extractor.py (Ollama 로컬) 의 AWS 배포 대체 버전
#
# 사전 요구사항:
#   1. pip install boto3
#   2. aws configure  (Access Key / Secret / Region 입력)
#   3. AWS 콘솔 → Bedrock → Model access → claude-3-haiku 활성화

import json
import re

import boto3
from botocore.exceptions import ClientError


class BedrockHPOExtractor:
    """
    임상 소견/증상 텍스트를 Positive/Negative HPO 코드 목록으로 변환.

    AWS Bedrock Claude-3-Haiku 를 사용 (빠르고 저렴, HPO 추출에 충분).
    한국어·영어 입력 모두 처리 가능.

    Example
    -------
    >>> extractor = BedrockHPOExtractor()
    >>> result = extractor.extract_hpo("호흡곤란과 흉통이 있으며 기침은 없음")
    >>> # {"positive_hpo": ["HP:0002094", "HP:0002098"], "negative_hpo": ["HP:0012735"]}
    """

    MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

    _SYSTEM = (
        "You are a clinical NLP expert specialized in rare lung diseases. "
        "Extract HPO (Human Phenotype Ontology) terms from the clinical text. "
        "Separate symptoms the patient HAS (positive) from symptoms explicitly ABSENT (negative). "
        "Use real HPO IDs in HP:XXXXXXX format only. "
        "If uncertain about an HPO ID, omit it rather than guess. "
        "Return ONLY a JSON object with no explanation."
    )

    _USER_TMPL = (
        "Clinical text:\n{text}\n\n"
        "Return JSON:\n"
        '{{"positive_hpo": ["HP:0000001", ...], "negative_hpo": ["HP:0000002", ...]}}'
    )

    def __init__(self, region: str = "ap-northeast-2"):
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def extract_hpo(self, symptom_text: str) -> dict:
        """
        Parameters
        ----------
        symptom_text : str
            임상 소견 텍스트 (한국어 또는 영어)

        Returns
        -------
        dict
            {"positive_hpo": [...], "negative_hpo": [...]}
            실패 시 빈 리스트 반환 (예외 미발생)
        """
        if not symptom_text or not symptom_text.strip():
            return {"positive_hpo": [], "negative_hpo": []}

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": self._SYSTEM,
            "messages": [
                {"role": "user", "content": self._USER_TMPL.format(text=symptom_text)}
            ],
            "max_tokens": 512,
            "temperature": 0.0,  # 재현성을 위해 0
        }

        try:
            response = self.client.invoke_model(
                modelId=self.MODEL_ID,
                body=json.dumps(body),
            )
            raw_text = json.loads(response["body"].read())["content"][0]["text"].strip()

            # LLM 이 마크다운 코드블록으로 감싸는 경우 대비
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if not match:
                print(f"⚠️ JSON 파싱 실패. 응답 앞부분: {raw_text[:200]}")
                return {"positive_hpo": [], "negative_hpo": []}

            result = json.loads(match.group())
            result.setdefault("positive_hpo", [])
            result.setdefault("negative_hpo", [])
            return result

        except ClientError as e:
            err = e.response["Error"]
            print(f"❌ Bedrock API 오류: {err['Code']} — {err['Message']}")
            return {"positive_hpo": [], "negative_hpo": []}

        except (json.JSONDecodeError, KeyError) as e:
            print(f"❌ 응답 파싱 오류: {e}")
            return {"positive_hpo": [], "negative_hpo": []}


if __name__ == "__main__":
    extractor = BedrockHPOExtractor()

    tests = [
        "40세 여성. 3주째 지속되는 호흡곤란과 흉통을 호소합니다. 기침은 없으며 발열도 없습니다.",
        "Dyspnea on exertion, hemoptysis present. No fever, no cough.",
    ]
    for text in tests:
        print(f"\n입력: {text}")
        out = extractor.extract_hpo(text)
        print(f"  Positive HPO: {out['positive_hpo']}")
        print(f"  Negative HPO: {out['negative_hpo']}")
