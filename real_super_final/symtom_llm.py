import boto3
import json
import os
import difflib

class BedrockHPOExtractor:
    def __init__(self, region_name="ap-northeast-2", whitelist_path=None):
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )

        self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"

        self.hpo_whitelist = {}
        self.term_to_hpo_map = {}

        if whitelist_path and os.path.exists(whitelist_path):
            with open(whitelist_path, 'r', encoding='utf-8') as f:
                self.hpo_whitelist = json.load(f)
            
            # 💡 [핵심] "Fatigue" -> "HP:0012378" 형태로 찾을 수 있도록 맵핑 사전을 만듭니다 (모두 소문자로 정규화)
            self.term_to_hpo_map = {term.lower(): hpo_id for hpo_id, term in self.hpo_whitelist.items()}
            
            print(f"✅ 공식 HPO 화이트리스트 로드 및 검색 엔진 초기화 완료 (총 {len(self.hpo_whitelist)}개)")
        else:
            print("⚠️ hpo_whitelist.json 파일이 없습니다. 스크립트를 먼저 실행해주세요.")
    
    def _map_term_to_hpo(self, english_term: str):
        """
        LLM이 번역한 영어 단어와 가장 유사한 HPO 공식 코드를 찾습니다.
        """
        if not self.term_to_hpo_map:
            return None, None
            
        term_lower = english_term.lower()
        
        # 1. 완벽하게 100% 일치하는 경우 (Exact Match)
        if term_lower in self.term_to_hpo_map:
            hpo_id = self.term_to_hpo_map[term_lower]
            return hpo_id, self.hpo_whitelist[hpo_id]
            
        # 2. 스펠링이 살짝 다르거나 비슷한 경우 (Fuzzy Match) - 유사도 80% 이상만 허용
        matches = difflib.get_close_matches(term_lower, self.term_to_hpo_map.keys(), n=1, cutoff=0.8)
        
        if matches:
            best_match = matches[0]
            hpo_id = self.term_to_hpo_map[best_match]
            return hpo_id, self.hpo_whitelist[hpo_id]
            
        # 3. 매칭되는 질환이 없는 경우 (예: 외계인 납치 증후군)
        return None, None
    

    def extract_hpo_from_clinical_note(self, clinical_note: str) -> dict:
        # 🚀 [프롬프트 고도화] LLM에게 HPO 코드를 생성하지 말라고 지시합니다.
        system_prompt = """
        You are an expert clinical informatician.
        Your task is to analyze clinical notes written in Korean and extract clinical findings.
        
        Strict Rules:
        1. ATOMIC EXTRACTION: Extract ONE symptom per JSON object. If a sentence mentions multiple symptoms (e.g., "가슴 답답함이나 창백함은 없음"), you MUST split them into separate objects (e.g., "Chest pain" and "Pallor").
        2. NO NEGATION WORDS: Do NOT include negation words (like "No", "Without") in the `english_term`.
        3. PREDICT OFFICIAL HPO LABEL: Do not just casually translate. You MUST predict and use the EXACT official HPO Primary Label.
           - Rule A: Include the specific anatomical site if implied (e.g., MUST use "Hyperpigmentation of the skin" instead of just "Hyperpigmentation").
           - Rule B: Use the most standard medical terminology (e.g., use "Pallor" for 창백함, "Vertigo" or "Dizziness" correctly).
        4. Output strictly in JSON format. Do NOT generate HPO IDs, just the English term.
        
        JSON Output Schema:
        {
            "positive_findings": [{"exact_quote_from_text": "...", "english_term": "..."}],
            "negative_findings": [{"exact_quote_from_text": "...", "english_term": "..."}]
        }
        """

        user_prompt = f"""
        Analyze the following clinical note and return the JSON:
        Clinical Note: "{clinical_note}"
        """

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.0 
        })

        try:
            print("\n🧠 Bedrock (Claude 3 Haiku) 소견서 번역 및 특징 추출 중...")
            response = self.bedrock_client.invoke_model(
                body=body, modelId=self.model_id,
                accept="application/json", contentType="application/json"
            )
            
            extracted_text = json.loads(response.get("body").read())["content"][0]["text"]
            raw_findings = json.loads(extracted_text)
            
            # 최종 결과물을 담을 구조체
            final_hpo_result = {
                "positive_hpos": [],
                "negative_hpos": [],
                "unmapped_findings": [] # 매핑 실패(환각 또는 비의료어) 분류
            }
            
            print("🔍 Python Fuzzy Matching 엔진으로 공식 HPO 매핑 중...")
            
            # Positive / Negative 항목들을 순회하며 파이썬으로 HPO 매핑
            for finding_type, target_key in [("positive_findings", "positive_hpos"), ("negative_findings", "negative_hpos")]:
                for item in raw_findings.get(finding_type, []):
                    english_term = item.get("english_term", "")
                    
                    # 💡 파이썬 매핑 엔진 가동
                    hpo_id, official_term = self._map_term_to_hpo(english_term)
                    
                    if hpo_id:
                        final_hpo_result[target_key].append({
                            "exact_quote_from_text": item["exact_quote_from_text"],
                            "hpo_id": hpo_id,
                            "official_term": official_term,
                            "original_translation": english_term # LLM이 번역했던 원본 유지 (디버깅용)
                        })
                    else:
                        final_hpo_result["unmapped_findings"].append(item)

            return final_hpo_result

        except Exception as e:
            print(f"❌ Bedrock API 호출 또는 JSON 파싱 에러: {e}")
            return {"positive_hpos": [], "negative_hpos": [], "unmapped_findings": []}

# =====================================================================
# 테스트 실행
# =====================================================================
if __name__ == "__main__":
    extractor = BedrockHPOExtractor(whitelist_path="aws_say2_project/hpo_whitelist.json")
    
    # 의사가 EMR에 입력한 실제와 유사한 한국어 소견서
    sample_clinical_note = """
    45세 남성 환자, 최근 2주간 지속되는 만성 피로와 양손 검지/중지 관절통(Arthralgia)으로 내원함. 
    최근 햇빛 노출이 없었음에도 피부가 청동색으로 어두워지는 양상 보임. 
    가슴 답답함이나 창백함은 호소하지 않으며, 어지러움(빈혈 소견)은 없음. 
    과거력 상 외계인 납치 증후군(Alien Abduction Syndrome) 의심됨.
    """
    
    result = extractor.extract_hpo_from_clinical_note(sample_clinical_note)
    
    print("\n✅ [파이썬 매핑 엔진이 적용된 최종 무결점 HPO 데이터]")
    print(json.dumps(result, indent=4, ensure_ascii=False))