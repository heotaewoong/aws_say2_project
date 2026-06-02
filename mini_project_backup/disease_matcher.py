import pandas as pd

class RareDiseaseMatcher:
    def __init__(self, csv_path="./data/orphadata_weighted.csv"):
        self.kb = pd.read_csv(csv_path)
        print(f"📚 {len(self.kb)}개의 지식 베이스 연결 완료")

    def match(self, patient_hpos, top_n=5):
        """
        patient_hpos: [{'hpo_id': 'HP:0002113', 'score': 0.76}, ...]
        """
        # 1. 환자의 HPO ID 리스트 추출
        print(patient_hpos)
        hpo_ids = [item['hpo_id'] for item in patient_hpos]
        # 에이전트들이 준 확신도(score) 맵핑
        conf_map = {item['hpo_id']: item['score'] for item in patient_hpos if 'score' in item}

        # 2. 지식 베이스에서 일치하는 행 필터링
        matched = self.kb[self.kb['HPO_ID'].isin(hpo_ids)].copy()
        
        # 3. 매칭 점수 계산: (Orphanet 가중치) * (에이전트 확신도)
        # 에이전트 확신도가 없는 경우(B 에이전트 등) 기본값 1.0 사용
        matched['Final_Weight'] = matched.apply(
            lambda row: row['Weight'] * conf_map.get(row['HPO_ID'], 1.0), axis=1
        )

        # 4. 질환별 점수 합산
        rankings = matched.groupby('DiseaseName').agg({
            'Final_Weight': 'sum',
            'HPO_Term': lambda x: ', '.join(x.unique())
        }).reset_index()

        rankings = rankings.sort_values(by='Final_Weight', ascending=False)
        
        return rankings.head(top_n)