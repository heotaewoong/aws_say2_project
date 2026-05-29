-- mock-emr 20-145982 (김○○) IPF 시나리오 HPI + PE 노트 풍부화
-- Phase 1 LLM 이 HPO 10개+ 추출하도록 의도

\set bundle_id (SELECT bundle_id FROM rarelinkai.raw_emr_bundle WHERE patient_id='20-145982' ORDER BY fetched_at DESC LIMIT 1)

-- 기존 chief_complaint 외 hpi/pe 노트 삭제 (재실행 가능하게)
DELETE FROM rarelinkai.clinical_note
 WHERE patient_id='20-145982' AND note_type IN ('hpi','pe');

INSERT INTO rarelinkai.clinical_note (patient_id, note_type, note_text_ko, language, author_role, recorded_at, bundle_id)
SELECT '20-145982', 'hpi',
$$58세 남성, 3개월 전부터 점진적으로 진행되는 노작 시 호흡곤란을 주소로 내원하였습니다.

초기에는 계단 두 층 정도 오를 때 호흡이 가빠지는 정도였으나 최근 1개월간은 일상 활동(샤워, 옷 입기, 짧은 거리 보행) 중에도 호흡곤란이 발생합니다. 마른기침(객담 없음)이 지속적으로 동반되며 객혈은 없습니다. 지난 3개월간 의도하지 않은 4kg의 체중감소가 있었고, 만성적인 피로감을 호소합니다. 야간 발한이나 흉통, 발열은 없습니다.

흡연력: 30갑년 (15세부터 50세까지 하루 1갑, 8년 전 금연한 전 흡연자입니다.)
직업력: 사무직 종사. 분진, 석면, 새 등 항원 노출력 없음.
가족력: 폐섬유증을 포함한 폐질환 가족력 없음.
약물력: 특이 약물 복용 없음. 폐독성 약물(amiodarone, methotrexate, bleomycin 등) 사용 이력 없음.
알레르기: penicillin 알레르기 (피부 발진).

진찰 소견:
- 활력징후: BP 128/76, HR 88, RR 22, SpO2 93% (실내공기), 체온 36.6도
- 폐 청진: 양측 폐기저부에서 흡기 종말기에 미세한 Velcro 양상의 수포음(bibasilar fine inspiratory crackles)이 청진됩니다.
- 손가락: 양측 손가락 곤봉화(clubbing of fingers)가 관찰됩니다.
- 청색증, 하지 부종은 없습니다.

경과 및 평가:
1개월 전 인근 의원에서 만성 기관지염 의증으로 amoxicillin 5일 복용 후 호전 없었습니다. 지역 종합병원에서 흉부 CT 검토 후 본원 호흡기내과로 진료 의뢰되었습니다.

간질성 폐질환(ILD), 특히 특발성 폐섬유증(IPF) 의심하 다음을 평가합니다: KL-6, SP-D, ANA, RF, anti-CCP, ANCA 자가면역 panel, 폐기능검사(FVC, DLCO), 6분 보행검사, 고해상도 흉부 CT (HRCT).$$,
'ko', 'physician', TIMESTAMP WITH TIME ZONE '2026-04-23 08:32:00+09:00',
(SELECT bundle_id FROM rarelinkai.raw_emr_bundle WHERE patient_id='20-145982' ORDER BY fetched_at DESC LIMIT 1);

INSERT INTO rarelinkai.clinical_note (patient_id, note_type, note_text_ko, language, author_role, recorded_at, bundle_id)
SELECT '20-145982', 'pe',
$$폐 청진: 양측 폐기저부에서 흡기 종말기 미세 Velcro 수포음(crackles) 청진. 천명음(wheezing)은 없음.
호흡수 22/분으로 증가됨 (정상 12-20). SpO2 93% (실내공기)로 감소.
양측 손가락 곤봉화(clubbing of fingers) 양성.
경정맥 압력 정상, 청색증 없음, 하지 부종 없음.
심음 정상, 잡음 없음.$$,
'ko', 'physician', TIMESTAMP WITH TIME ZONE '2026-04-23 08:35:00+09:00',
(SELECT bundle_id FROM rarelinkai.raw_emr_bundle WHERE patient_id='20-145982' ORDER BY fetched_at DESC LIMIT 1);

INSERT INTO rarelinkai.clinical_note (patient_id, note_type, note_text_ko, language, author_role, recorded_at, bundle_id)
SELECT '20-145982', 'imaging',
$$흉부 X-ray (PA, lateral, 2026-04-23 외부의원 의뢰):
양측 폐기저부에 reticular opacity 와 honeycombing 양상의 음영 증가가 관찰됩니다.
심장 크기는 정상 범위이며 흉수는 없습니다.
판독 의견: 양측 폐기저부 reticular opacity, honeycombing 양상 — 간질성 폐질환(특히 IPF/UIP pattern) 의심. HRCT 권유.$$,
'ko', 'radiologist', TIMESTAMP WITH TIME ZONE '2026-04-23 08:40:00+09:00',
(SELECT bundle_id FROM rarelinkai.raw_emr_bundle WHERE patient_id='20-145982' ORDER BY fetched_at DESC LIMIT 1);

-- verify
SELECT note_type, length(note_text_ko) AS chars, recorded_at
  FROM rarelinkai.clinical_note
 WHERE patient_id='20-145982'
 ORDER BY recorded_at DESC;
