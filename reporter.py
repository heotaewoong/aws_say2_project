class DiagnosticReporter:
    def __init__(self):
        self.common_diseases = ['Pneumonia', 'Atelectasis', 'Pleural Effusion', 'Congestive Heart Failure']

    def generate_report(self, subject_id, common_results, rare_results, matched_hpos):
        """
        일반 질환과 희귀 질환 결과를 종합하여 최종 리포트 생성
        """
        report = []
        report.append(f"=== [Rare-Link AI] 환자 진단 분석 리포트 (ID: {subject_id}) ===")
        
        # 1. 통합 관찰 소견 (Evidence)
        report.append(f"\n[1. 주요 관찰 소견]")
        report.append(f" - 검출된 주요 증상(HPO): {', '.join(matched_hpos)}")
        
        # 2. 일반 질환 스크리닝 결과 (Common Diseases)
        report.append(f"\n[2. 일반 폐 질환 스크리닝]")
        top_common = common_results[0] if common_results else "특이 사항 없음"
        report.append(f" - 가장 유력한 일반 소견: {top_common}")
        
        # 3. 희귀 질환 가능성 분석 (Rare Diseases)
        report.append(f"\n[3. 희귀 질환 감별 진단]")
        if not rare_results.empty:
            top_rare = rare_results.iloc[0]
            report.append(f" - 추천 후보: {top_rare['DiseaseName']} (신뢰도 점수: {top_rare['Score']})")
            report.append(f" - 판단 근거: {', '.join(top_rare['Matched_Symptoms'])}")
        else:
            report.append(f" - 특이 희귀 질환 징후가 발견되지 않았습니다.")

        # 4. 하이브리드 결론 (Clinical Logic)
        report.append(f"\n[4. 종합 결론 및 제언]")
        # 예: 곤봉지(HP:0001217)는 일반 폐 질환에서 드물기 때문에 희귀 질환 가능성 경고
        if 'HP:0001217' in matched_hpos:
            report.append("❗ 주의: 일반적인 폐 소견 외에 '곤봉지'가 관찰되었습니다. 이는 특발성 폐섬유화증 등 희귀 질환의 지표일 수 있으므로 정밀 검사를 권고합니다.")
        else:
            report.append(" - 현재 소견은 일반적인 폐 질환 범주 내에 있습니다.")

        return "\n".join(report)