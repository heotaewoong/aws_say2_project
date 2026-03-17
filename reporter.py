import datetime

class ClinicalReporter:
    def __init__(self):
        # 일반적인 폐 질환 목록 (Vision에서 주로 탐지됨)
        self.common_diseases = ["Pneumonia", "Pneumothorax", "Pleural Effusion"]

    def generate_summary(self, subject_id, nlp_findings, vision_findings, rare_rankings):
        """
        모든 소스를 통합하여 의사용 최종 리포트 작성
        """
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        
        report = []
        report.append(f"STRICTLY CONFIDENTIAL - Rare-Link AI Diagnostic Report")
        report.append(f"분석 일시: {now} | 환자 ID: {subject_id}")
        report.append("="*60)

        # 1. 멀티모달 관찰 소견
        report.append("\n[1. 통합 관찰 소견 (Clinical Evidence)]")
        # NLP 소견 정리
        nlp_terms = [f["finding"] for f in nlp_findings]
        report.append(f" - 📝 텍스트/신체 검진: {', '.join(nlp_terms) if nlp_terms else '특이 사항 없음'}")
        # Vision 소견 정리
        vis_terms = [f["finding"] for f in vision_findings]
        report.append(f" - 📸 영상 분석(CXR): {', '.join(vis_terms) if vis_terms else '정상 소견'}")

        # 2. 하이브리드 진단 결론
        report.append("\n[2. 진단 분석 결론 (Diagnostic Conclusion)]")
        
        # 일반 질환 여부 판단
        detected_common = [f for f in vis_terms if f in self.common_diseases]
        if detected_common:
            report.append(f" ⚠️ 일반 소견 감지: {', '.join(detected_common)} 가능성이 높습니다.")
        
        # 희귀 질환 경고 (가장 높은 점수의 질환 제시)
        if not rare_rankings.empty:
            top_rare = rare_rankings.iloc[0]
            report.append(f" 🧬 희귀 질환 의심: '{top_rare['DiseaseName']}' (매칭 점수: {top_rare['Score']})")
            report.append(f"    ㄴ 판단 근거: {top_rare['Evidence']}")
        
        # 3. 임상적 제언 (Clinical Suggestion)
        report.append("\n[3. 전문의 제언]")
        # 하이브리드 로직: 영상에선 일반 질환이 보이지만 텍스트에 특이 증상(예: 곤봉지)이 있는 경우
        if "HP:0001217" in [f["hpo_id"] for f in nlp_findings] and detected_common:
            report.append(" ❗ 주의: 일반 폐 소견 외에 '곤봉지'가 관찰됩니다. 단순 폐렴이 아닌 특발성 폐질환 가능성에 대한 정밀 검토가 필요합니다.")
        else:
            report.append(" - 환자의 임상 증상과 영상 소견이 일치합니다. 표준 치료 절차를 권고합니다.")

        report.append("\n" + "="*60)
        report.append("본 리포트는 AI 보조 도구에 의해 작성되었습니다. 최종 진단은 전문의의 판단이 필요합니다.")
        
        return "\n".join(report)