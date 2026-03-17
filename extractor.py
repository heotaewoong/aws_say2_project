import pandas as pd
import json
# import openai  # 실제 API 사용 시 활성화

class PhenotypeExtractor:
    def __init__(self):
        # 🧪 Lab 결과 판정 기준 (실제 프로젝트 시 확장 필요)
        self.lab_rules = {
            'Platelet Count': {'low': 150, 'hpo': 'HP:0001873'},
            'Glucose': {'high': 126, 'hpo': 'HP:0000819'},
            'WBC Count': {'high': 11.0, 'hpo': 'HP:0001890'}
        }

    def extract_from_text_llm(self, text):
        """
        [Phase 2-1] LLM을 사용하여 텍스트에서 HPO 추출
        프롬프트 엔지니어링이 핵심입니다.
        """
        prompt = f"""
        당신은 숙련된 호흡기 내과 전문의입니다. 
        아래의 의사 소견서에서 환자의 이상 소견(Phenotypes)을 추출하여 HPO ID와 매핑하세요.
        결과는 반드시 JSON 리스트 형식으로 답변하세요.
        
        [소견서 본문]
        {text}
        
        [응답 형식]
        [
          {{"finding": "증상명", "hpo_id": "HP:XXXXXXX"}},
          ...
        ]
        """
        # 실제 API 호출 예시 (현재는 프로토타입 결과 반환)
        # response = openai.ChatCompletion.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
        # return json.loads(response.choices[0].message.content)
        
        print("🤖 LLM 엔진이 텍스트를 분석 중입니다...")
        return [
            {"finding": "Clubbing", "hpo_id": "HP:0001217"},
            {"finding": "Dyspnea", "hpo_id": "HP:0002094"}
        ]

    def extract_from_lab_data(self, lab_row):
        """
        [Phase 2-2] 수치 데이터를 규칙 기반으로 HPO 변환
        """
        label = lab_row['label']
        value = lab_row['valuenum']
        
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