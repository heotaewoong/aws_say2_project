import pandas as pd
import numpy as np
import os

class RareDiseaseInference:
    def __init__(self, kb_path=None):
        if kb_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 프로젝트 구조에 맞춰 경로 조정 (data/orphadata_weighted.csv)
            kb_path = os.path.normpath(os.path.join(current_dir, "data", "orphadata_weighted.csv"))

        if not os.path.exists(kb_path):
            raise FileNotFoundError(f"❌ 지식 베이스 파일이 없습니다: {kb_path}")
        
        self.kb_df = pd.read_csv(kb_path)
        self.hpo_specificity = self._calculate_hpo_idf()
        print(f"🧬 하이브리드 추론 엔진 로드 완료 (지식 베이스: {len(self.kb_df)} 행)")

    def _calculate_hpo_idf(self):
        """특정 증상이 나타나는 질환이 적을수록 IDF(희귀도) 점수를 높게 책정"""
        total_diseases = self.kb_df['OrphaCode'].nunique()
        hpo_counts = self.kb_df.groupby('HPO_ID')['OrphaCode'].nunique()
        # 0으로 나누기 방지 및 스무딩을 위해 +1 적용
        return np.log(total_diseases / (hpo_counts + 1)) + 1

    def rank_diseases(self, patient_hpos, top_k=5):
        """
        에이전트들의 표준 출력([{hpo_id, score, ...}])을 받아 정밀 순위 반환
        """
        if not patient_hpos:
            return pd.DataFrame()

        # 1. 환자 데이터 전처리: 동일 HPO가 여러 에이전트에서 나오면 가장 높은 점수 선택
        patient_df = pd.DataFrame(patient_hpos)
        # 키값 표준화 (hpo_id 혹은 HPO_ID)
        if 'hpo_id' in patient_df.columns:
            patient_df = patient_df.rename(columns={'hpo_id': 'HPO_ID'})
        
        # 중복 증상 중 최대 점수만 남김
        patient_scores = patient_df.groupby('HPO_ID')['score'].max().to_dict()
        valid_hpo_ids = list(patient_scores.keys())

        # 2. 지식 베이스 필터링 (환자가 가진 증상만)
        matched_kb = self.kb_df[self.kb_df['HPO_ID'].isin(valid_hpo_ids)].copy()

        if matched_kb.empty:
            return pd.DataFrame(columns=['DiseaseName', 'Score', 'Evidence'])

        # 3. 가중치 결합 계산
        # - weight: Orphanet 빈도 가중치
        # - idf: HPO 자체의 희귀도/특이도
        # - agent_score: 에이전트가 판단한 증상의 확실함
        matched_kb['idf'] = matched_kb['HPO_ID'].map(self.hpo_specificity)
        matched_kb['agent_score'] = matched_kb['HPO_ID'].map(patient_scores)
        
        # 핵심 수식: 최종 기여도 = 가중치 * 희귀도 * 확신도
        matched_kb['contribution'] = matched_kb['Weight'] * matched_kb['idf'] * matched_kb['agent_score']

        # 4. 질환별 점수 합산 및 증거 정리
        rankings = matched_kb.groupby(['OrphaCode', 'DiseaseName']).agg({
            'contribution': 'sum',
            'HPO_Term': lambda x: ", ".join([f"{term}" for term in x])
        }).reset_index()

        # 컬럼명 정리 및 정렬
        rankings = rankings.rename(columns={'contribution': 'Score', 'HPO_Term': 'Evidence'})
        rankings = rankings.sort_values(by='Score', ascending=False).head(top_k)
        
        return rankings[['DiseaseName', 'Score', 'Evidence']]