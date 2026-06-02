"""
============================================================
🧬 disease_matcher_v2.py – 개선된 질환 매칭 엔진
============================================================

🎯 이 파일이 하는 일 (초보자용 설명):
    AI가 찾은 증상(HPO 코드)들을 받아서
    "이 환자는 어떤 희귀 질환일 가능성이 높은지" 순위를 매깁니다.
    
    비유:
    - AI가 "숨이 차다(HP:0002094)", "폐가 하얗다(HP:0032183)" 를 발견
    - 이 엔진이 의학 백과사전을 뒤져서
    - "폐포단백증 85점, 폐섬유증 72점, 유육종증 65점" 순위를 매김
    
    V2 대비 개선점:
    1. [COSMIC 통합] 비암성(Orphadata) + 암성(COSMIC) 두 레이어 병렬 분석
    2. [신뢰도 가중] 에이전트별 확신도를 반영한 정밀 점수 계산
    3. [설명 생성] 왜 이 질환이 의심되는지 근거를 자동 생성
============================================================
"""

import pandas as pd
import numpy as np
import os


class DiseaseMatcherV2:
    """
    초보자 설명:
        이 클래스는 "명탐정"입니다.
        
        여러 단서(증상)를 모아서 "범인(질환)"을 추리합니다.
        단서가 희귀할수록 더 큰 가중치를 받습니다.
        (예: "기침"은 흔해서 가중치 낮음, "곤봉지"는 희귀해서 가중치 높음)
    """
    
    def __init__(self, orphadata_path=None, cosmic_path=None):
        """
        Args:
            orphadata_path: 비암성 희귀질환 DB (orphadata_weighted.csv)
            cosmic_path:    암성 질환 DB (cosmic_xray_precise.csv)
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # data/ 폴더를 여러 경로에서 탐색
        data_candidates = [
            os.path.join(current_dir, "data"),        # ./data/
            os.path.join(current_dir, "..", "data"),   # ../data/
            current_dir,                               # 현재 디렉토리
        ]
        
        # Layer 1: 비암성 희귀질환 (HPO + Orphadata)
        if orphadata_path is None:
            for d in data_candidates:
                candidate = os.path.join(d, "orphadata_weighted.csv")
                if os.path.exists(candidate):
                    orphadata_path = candidate
                    break
            if orphadata_path is None:
                orphadata_path = os.path.join(current_dir, "data", "orphadata_weighted.csv")
        
        if os.path.exists(orphadata_path):
            self.orphadata_kb = pd.read_csv(orphadata_path)
            self.orphadata_idf = self._calculate_idf(self.orphadata_kb, 'OrphaCode')
            print(f"📚 Layer 1 (비암성) 로드: {len(self.orphadata_kb)}행")
        else:
            self.orphadata_kb = None
            print(f"⚠️ Orphadata 파일 없음: {orphadata_path}")
        
        # Layer 2: 암성 질환 (COSMIC CGC)
        if cosmic_path is None:
            for d in data_candidates:
                candidate = os.path.join(d, "cosmic_xray_precise.csv")
                if os.path.exists(candidate):
                    cosmic_path = candidate
                    break
            if cosmic_path is None:
                cosmic_path = os.path.join(current_dir, "data", "cosmic_xray_precise.csv")
        
        if os.path.exists(cosmic_path):
            self.cosmic_kb = pd.read_csv(cosmic_path)
            print(f"📚 Layer 2 (암성) 로드: {len(self.cosmic_kb)}행")
        else:
            self.cosmic_kb = None
            print(f"⚠️ COSMIC 파일 없음 (암성 분석 비활성): {cosmic_path}")

    def _calculate_idf(self, kb_df, disease_col):
        """
        초보자 설명:
            IDF(역문서 빈도)는 "이 증상이 얼마나 특이한가"를 수치화합니다.
            
            - "기침" → 500개 질환에 등장 → IDF 낮음 (흔한 증상)
            - "곤봉지" → 5개 질환에만 등장 → IDF 높음 (특이한 증상!)
            
            특이한 증상일수록 진단에 결정적인 단서가 됩니다.
        """
        total_diseases = kb_df[disease_col].nunique()
        hpo_counts = kb_df.groupby('HPO_ID')[disease_col].nunique()
        return np.log(total_diseases / hpo_counts.clip(lower=1))

    def match(self, patient_hpos, top_n=5):
        """
        환자의 HPO 리스트를 받아 질환 순위를 반환합니다.
        
        Args:
            patient_hpos: 에이전트들이 찾은 HPO 리스트
                예: [{'hpo_id': 'HP:0002090', 'score': 0.85, 'finding': 'Pneumonia'}, ...]
            top_n: 상위 몇 개 질환을 반환할지
            
        Returns:
            {'layer1': DataFrame, 'layer2': DataFrame, 'combined_summary': str}
        """
        # HPO ID와 확신도 맵 생성
        hpo_ids = [item['hpo_id'] for item in patient_hpos if item.get('hpo_id')]
        conf_map = {item['hpo_id']: item.get('score', 1.0) for item in patient_hpos}
        
        results = {}
        
        # ── Layer 1: 비암성 희귀질환 분석 ──
        if self.orphadata_kb is not None:
            layer1 = self._score_layer(
                self.orphadata_kb, hpo_ids, conf_map,
                disease_name_col='DiseaseName',
                disease_id_col='OrphaCode',
                idf_scores=self.orphadata_idf,
            )
            results['layer1'] = layer1.head(top_n)
            print(f"\n🔬 Layer 1 (비암성) Top {top_n}:")
            for _, row in results['layer1'].iterrows():
                print(f"  {row['DiseaseName']:40s} | 점수: {row['FinalScore']:.2f} | 근거: {row['Evidence']}")
        
        # ── Layer 2: 암성 질환 분석 ──
        if self.cosmic_kb is not None:
            # COSMIC은 X-ray 소견 → 유전자 매핑이므로 다른 방식
            layer2_results = self._cosmic_analysis(patient_hpos)
            results['layer2'] = layer2_results
            if not layer2_results.empty:
                print(f"\n🧬 Layer 2 (암성) 유전자 추천:")
                for _, row in layer2_results.head(5).iterrows():
                    print(f"  {row.get('Gene', 'N/A'):10s} | {row.get('Cancer', 'N/A'):20s} | {row.get('Clinical', 'N/A')}")
        
        # ── 통합 요약 생성 ──
        results['summary'] = self._generate_summary(results, patient_hpos)
        
        return results

    def _score_layer(self, kb_df, hpo_ids, conf_map, disease_name_col, disease_id_col, idf_scores):
        """질환별 점수 계산 (IDF × Weight × 에이전트 확신도)"""
        valid_hpos = [h for h in hpo_ids if h in idf_scores.index]
        
        if not valid_hpos:
            return pd.DataFrame(columns=[disease_name_col, 'FinalScore', 'Evidence'])
        
        results = []
        for (did, dname), group in kb_df.groupby([disease_id_col, disease_name_col]):
            matches = group[group['HPO_ID'].isin(valid_hpos)]
            if matches.empty:
                continue
            
            score = 0
            evidence = []
            for _, row in matches.iterrows():
                hpo_id = row['HPO_ID']
                weight = row.get('Weight', 1.0)
                idf = idf_scores.get(hpo_id, 1.0)
                confidence = conf_map.get(hpo_id, 1.0)
                
                contribution = weight * idf * confidence
                score += contribution
                hpo_term = row.get('HPO_Term', hpo_id)
                evidence.append(f"{hpo_term}(+{contribution:.1f})")
            
            results.append({
                disease_name_col: dname,
                disease_id_col: did,
                'FinalScore': round(score, 2),
                'MatchCount': len(matches),
                'Evidence': ", ".join(evidence),
            })
        
        df = pd.DataFrame(results)
        if df.empty:
            return df
        return df.sort_values('FinalScore', ascending=False).reset_index(drop=True)

    def _cosmic_analysis(self, patient_hpos):
        """COSMIC Layer 2: X-ray 소견에서 유전자 추천"""
        if self.cosmic_kb is None:
            return pd.DataFrame()
        
        # 소견명으로 COSMIC 매칭
        findings = [h.get('finding', '') for h in patient_hpos]
        matched_genes = []
        
        for finding in findings:
            # 소견과 관련된 유전자 검색
            for _, row in self.cosmic_kb.iterrows():
                xray_col = 'X-Ray 소견' if 'X-Ray 소견' in self.cosmic_kb.columns else self.cosmic_kb.columns[0]
                if finding.lower() in str(row.get(xray_col, '')).lower():
                    matched_genes.append(row.to_dict())
        
        return pd.DataFrame(matched_genes).drop_duplicates() if matched_genes else pd.DataFrame()

    def _generate_summary(self, results, patient_hpos):
        """의사를 위한 텍스트 요약 생성"""
        lines = ["=" * 50, "📋 통합 진단 분석 요약", "=" * 50]
        
        # 검출된 소견
        lines.append(f"\n🔍 검출된 소견: {len(patient_hpos)}개")
        for h in patient_hpos:
            lines.append(f"  • {h.get('finding', 'Unknown')} ({h.get('hpo_id', '?')}) 확신도: {h.get('score', 0):.1%}")
        
        # Layer 1 결과
        if 'layer1' in results and not results['layer1'].empty:
            lines.append(f"\n🔬 비암성 희귀질환 후보:")
            for i, (_, row) in enumerate(results['layer1'].head(3).iterrows(), 1):
                lines.append(f"  {i}. {row['DiseaseName']} (점수: {row['FinalScore']:.1f})")
        
        # Layer 2 결과
        if 'layer2' in results and not results['layer2'].empty:
            lines.append(f"\n🧬 암성 관련 유전자 검사 추천:")
            for _, row in results['layer2'].head(3).iterrows():
                gene = row.get('Gene', row.get(results['layer2'].columns[0], 'N/A'))
                lines.append(f"  → {gene}")
        
        return "\n".join(lines)


# ============================================================
# 🧪 테스트
# ============================================================
if __name__ == "__main__":
    matcher = DiseaseMatcherV2()
    
    # 테스트: 폐렴 + 흉수가 발견된 환자
    test_hpos = [
        {'hpo_id': 'HP:0002090', 'score': 0.85, 'finding': 'Pneumonia'},
        {'hpo_id': 'HP:0002202', 'score': 0.72, 'finding': 'Pleural Effusion'},
        {'hpo_id': 'HP:0100750', 'score': 0.65, 'finding': 'Atelectasis'},
    ]
    
    results = matcher.match(test_hpos, top_n=5)
    print(results['summary'])
