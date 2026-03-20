import json
import ollama # pip install ollama

class TextPhenotypeExtractor:
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name
        self.system_instruction = (
            "You are an expert Clinical Geneticist specialized in Deep Phenotyping. "
            "Your mission is to extract EVERY SINGLE clinical phenotype mentioned in the text. "
            "Do not miss subtle signs like 'clubbing', 'cyanosis', or 'murmur'. "
            "For each finding, map it to the most accurate HPO ID. "
            "If a finding is mentioned, it MUST be included in the output list. "
            "Return ONLY a JSON list: [{'finding': '...', 'hpo_id': 'HP:XXXXXXX'}]."
        )

    def extract_from_text(self, text):
        prompt = f"Extract HPO codes from this Discharge Summary:\n\n{text}"
        
        try:
            print(f"✨ 로컬 LLM({self.model_name})이 소견서를 분석 중입니다...")
            response = ollama.generate(
                model=self.model_name,
                system=self.system_instruction,
                prompt=prompt,
                format="json", # Ollama의 JSON 모드 활성화
                options={"temperature": 0} # 일관된 결과를 위해 0 설정
            )
            
            # 응답 텍스트에서 JSON 파싱
            return json.loads(response['response'])
            
        except Exception as e:
            print(f"❌ 로컬 LLM 호출 중 에러 발생: {e}")
            return []

if __name__ == "__main__":
    # 테스트용
    extractor = TextPhenotypeExtractor()
    sample_text = "Patient has worsening abdominal distension and clubbing of fingers."
    print(extractor.extract_from_text(sample_text))