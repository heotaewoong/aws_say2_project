import pandas as pd
import json
import google.generativeai as genai

class PhenotypeExtractor:
    def __init__(self, api_key):
        # xAI API 설정
        genai.configure(api_key=api_key)

        # 시스템 명령 설정 (모델의 역할을 정의)
        system_instruction = (
            "You are a highly skilled Clinical Geneticist and Pulmonologist. "
            "Your task is to perform deep phenotyping by mapping clinical findings "
            "to the Human Phenotype Ontology (HPO) for rare disease diagnosis."
        )
        
        # 모델 초기화 (JSON 모드 강제)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_instruction,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # Lab 결과 판정 기준
        self.lab_rules = {
            'Platelet Count': {'low': 150, 'hpo': 'HP:0001873'},
            'Glucose': {'high': 126, 'hpo': 'HP:0000819'},
            'WBC Count': {'high': 11.0, 'hpo': 'HP:0001890'}
        }

    def extract_from_text_llm(self, text):
        prompt = f"""
        Analyze the following Discharge Summary and extract all clinical phenotypes mapped to HPO IDs.
        Focus on features relevant to both common and rare respiratory conditions.

        Note: Exclude normal findings, medications, and procedures.
        
        [Discharge Summary]
        {text}
        
        Return a JSON list of objects:
        [
          {{"finding": "...", "hpo_id": "HP:XXXXXXX"}}
        ]
        """
        
        try:
            print("✨ Gemini-2,0-Flash가 소견서를 분석 중입니다...")
            response = self.model.generate_content(prompt)
            
            # Gemini는 설정에 따라 순수 JSON 문자열을 반환합니다.
            return json.loads(response.text)
            
        except Exception as e:
            print(f"❌ Gemini API 호출 중 에러 발생: {e}")
            return []


    def extract_from_lab_data(self, lab_row):
        label = lab_row.get('label')
        value = lab_row.get('valuenum')
        
        if pd.isna(value) or not label:
            return None
            
        if label in self.lab_rules:
            rule = self.lab_rules[label]
            if 'low' in rule and value < rule['low']:
                return {'finding': f'Low {label}', 'hpo_id': rule['hpo']}
            if 'high' in rule and value > rule['high']:
                return {'finding': f'High {label}', 'hpo_id': rule['hpo']}
        return None
    
if __name__ == "__main__":
    extractor = PhenotypeExtractor()
    print("✅ Extractor 모듈 로드 완료")