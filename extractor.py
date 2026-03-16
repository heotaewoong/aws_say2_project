class PhenotypeExtractor:
    def __init__(self):
        # 향후 실제 Lab 정상 범위 테이블로 확장 가능
        self.lab_thresholds = {
            'Platelet Count': {'low': 150, 'hpo': 'HP:0001873'},
            'Glucose': {'high': 126, 'hpo': 'HP:0000819'}
        }

    def extract_from_text(self, text):
        """
        [Phase 2 - NLP] LLM API를 호출하여 텍스트에서 HPO를 추출하는 모듈
        현재는 프로토타입용 샘플 데이터를 반환합니다.
        """
        # 실제 구현 시: openai.ChatCompletion.create(...) 호출
        sample_hpos = ['HP:0001217', 'HP:0001541'] # 곤봉지, 복수
        return sample_hpos

    def extract_from_lab(self, label, value):
        """
        [Phase 2 - Lab] 수치 데이터를 판정하여 HPO 코드로 변환
        """
        if label in self.lab_thresholds:
            rule = self.lab_thresholds[label]
            if 'low' in rule and value < rule['low']:
                return rule['hpo']
            if 'high' in rule and value > rule['high']:
                return rule['hpo']
        return None