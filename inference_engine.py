import pandas as pd
import numpy as np

class InferenceEngine:
    def __init__(self, knowledge_base_path='orphadata_weighted.csv'):
        self.df = pd.read_csv(knowledge_base_path)
        self.hpo_specificity = self._calculate_idf()

    def _calculate_idf(self):
        """
        증상별 희귀도(IDF) 계산: 
        많은 질환에서 공통으로 나타나는 증상(예: 경련)은 낮은 점수를, 
        특정 질환에서만 나타나는 증상(예: 곤봉지)은 높은 점수를 갖게 함.
        """
        total_diseases = self.df['OrphaCode'].nunique()
        hpo_counts = self.df.groupby('HPO_ID')['OrphaCode'].nunique()
        # IDF = log(전체 질환 수 / 해당 증상을 가진 질환 수)
        return np.log(total_diseases / hpo_counts)

    def rank_diseases(self, patient_hpo_list, top_k=5):
        """
        환자의 HPO 리스트를 입력받아 가중치 합산 점수가 높은 질환을 추천
        """
        # 지식 베이스에 존재하는 증상만 필터링
        valid_hpos = [h for h in patient_hpo_list if h in self.hpo_specificity.index]
        
        results = []
        # 질환별로 그룹화하여 점수 계산
        for (orpha_code, disease_name), group in self.df.groupby(['OrphaCode', 'DiseaseName']):
            # 환자 증상과 해당 질환의 증상이 겹치는 부분 찾기
            match = group[group['HPO_ID'].isin(valid_hpos)]
            
            if not match.empty:
                score = 0
                matched_details = []
                
                for _, row in match.iterrows():
                    # 점수 = 증상의 희귀도(IDF) * 해당 질환에서의 발생 빈도 가중치(Weight)
                    hpo_weight = row['Weight']
                    hpo_idf = self.hpo_specificity[row['HPO_ID']]
                    
                    contribution = hpo_idf * hpo_weight
                    score += contribution
                    matched_details.append(f"{row['HPO_Term']} (가중치: {contribution:.2f})")
                
                results.append({
                    'OrphaCode': orpha_code,
                    'DiseaseName': disease_name,
                    'Score': round(score, 2),
                    'Matched_Symptoms': matched_details
                })
        
        # 점수 기준 내림차순 정렬 후 Top-K 반환
        return pd.DataFrame(results).sort_values(by='Score', ascending=False).head(top_k)