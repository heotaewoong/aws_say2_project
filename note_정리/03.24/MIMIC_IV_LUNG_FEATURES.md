# MIMIC-IV 폐 관련 지표 분석 — 프로젝트 활용 가능 여부

> 프로젝트: CheXNet (DenseNet-121) + HPO 기반 희귀질환 추론
> 모델 출력 14개 라벨: Atelectasis, Cardiomegaly, Consolidation, Edema,
> Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity,
> No Finding, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices

---

## ✅ 사용 가능한 지표

### 1. mimic-cxr-2.0.0-chexpert.csv — 핵심 레이블 (227,827건)
> **가장 중요한 파일. 모델 학습/평가의 정답 레이블**

| 라벨 | 양성(1) | 음성(0) | 불확실(-1) | 언급없음(NaN) | 활용 |
|------|---------|---------|-----------|-------------|------|
| Atelectasis | 45,808 | 1,531 | 10,327 | 170,161 | ✅ 학습 가능 |
| Cardiomegaly | 44,845 | 15,911 | 6,043 | 161,028 | ✅ 학습 가능 |
| Consolidation | 10,778 | 7,967 | 4,331 | 204,751 | ✅ 학습 가능 |
| Edema | 27,018 | 25,641 | 13,174 | 161,994 | ✅ 학습 가능 |
| Enlarged Cardiomediastinum | 7,179 | 5,283 | 9,375 | 205,990 | ✅ 학습 가능 |
| Fracture | 4,390 | 886 | 555 | 221,996 | ✅ (소수) |
| Lung Lesion | 6,284 | 862 | 1,141 | 219,540 | ✅ (소수) |
| Lung Opacity | 51,525 | 3,069 | 3,831 | 169,402 | ✅ 학습 가능 |
| No Finding | 75,455 | - | - | 152,372 | ✅ 정상 클래스 |
| Pleural Effusion | 54,300 | 27,158 | 5,814 | 140,555 | ✅ 학습 가능 |
| Pleural Other | 2,011 | 126 | 765 | 224,925 | ⚠️ 매우 소수 |
| Pneumonia | 16,556 | 24,338 | 18,291 | 168,642 | ✅ 학습 가능 |
| Pneumothorax | 10,358 | 42,356 | 1,134 | 173,979 | ✅ 학습 가능 |
| Support Devices | 66,558 | 3,486 | 237 | 157,546 | ✅ 학습 가능 |

**불확실(-1) 처리 전략**:
- `U-Ignore`: -1을 학습에서 제외 (가장 단순)
- `U-Zeros`: -1을 0으로 처리
- `U-Ones`: -1을 1로 처리 (CheXpert 논문 권장)
- `U-MultiClass`: 3-class 분류

---

### 2. chartevents — ICU 활력징후 (사용 가능)

| itemid | 지표명 | 단위 | 폐 관련성 | 활용 |
|--------|--------|------|----------|------|
| 220210 | Respiratory Rate | insp/min | 호흡수 | ✅ 임상 보조 지표 |
| 220277 | O2 saturation (SpO2) | % | 산소포화도 | ✅ 핵심 지표 |
| 220224 | Arterial O2 pressure (PaO2) | mmHg | 동맥혈 산소분압 | ✅ 핵심 지표 |
| 220235 | Arterial CO2 pressure (PaCO2) | mmHg | 동맥혈 이산화탄소 | ✅ 핵심 지표 |
| 223830 | PH (Arterial) | - | 동맥혈 pH | ✅ 산염기 균형 |
| 220339 | PEEP set | cmH2O | 호기말양압 (인공호흡기) | ✅ 중증도 지표 |
| 224684 | Tidal Volume (set) | mL | 1회 호흡량 | ✅ 인공호흡기 |
| 224695 | Peak Insp. Pressure | cmH2O | 최고 흡기압 | ✅ 인공호흡기 |
| 224696 | Plateau Pressure | cmH2O | 고원압 (ARDS 지표) | ✅ ARDS 진단 |
| 223835 | Inspired O2 Fraction (FiO2) | % | 흡입 산소 농도 | ✅ P/F ratio 계산 |
| 223849 | Ventilator Mode | - | 인공호흡기 모드 | ✅ 중증도 |
| 225792 | Invasive Ventilation | - | 침습적 기계환기 여부 | ✅ 중증도 |
| 223986~223989 | Lung Sounds (RUL/RLL/LUL/LLL) | - | 폐음 청진 | ✅ 임상 소견 |
| 228640 | EtCO2 | - | 호기말 이산화탄소 | ✅ 환기 모니터링 |
| 229661 | Compliance | cmH2O/L/s | 폐 순응도 | ✅ ARDS 지표 |

**P/F ratio (PaO2/FiO2) 계산 가능** → ARDS 중증도 분류에 활용

---

### 3. labevents — 혈액검사 (사용 가능)

| 검사명 | 폐 관련성 | 활용 |
|--------|----------|------|
| PaO2 (동맥혈 산소분압) | 폐 가스교환 | ✅ |
| PaCO2 (동맥혈 CO2) | 환기 기능 | ✅ |
| pH (동맥혈) | 산염기 균형 | ✅ |
| Lactate | 조직 저산소증 | ✅ |
| WBC (백혈구) | 폐렴/감염 | ✅ |
| CRP | 염증 (폐렴) | ✅ |
| Procalcitonin | 세균성 폐렴 | ✅ |
| BNP/NT-proBNP | 심인성 폐부종 | ✅ |

---

### 4. diagnoses_icd — 진단 코드 (코호트 추출용)

| ICD 코드 | 진단명 | CheXpert 라벨 연계 |
|---------|--------|------------------|
| J18.x (ICD-10) | Pneumonia | Pneumonia |
| J93.x | Pneumothorax | Pneumothorax |
| J90 | Pleural Effusion | Pleural Effusion |
| J98.1 | Pulmonary Collapse (Atelectasis) | Atelectasis |
| J81 | Pulmonary Edema | Edema |
| J80 | ARDS | Lung Opacity + Edema |
| 038.x (ICD-9) | Sepsis | - |

---

### 5. radiology.csv — 영상 판독 텍스트 (NLP 활용 가능)

- 흉부 X-ray 판독 보고서 전문 텍스트
- CheXpert 레이블과 `subject_id`로 연계 가능
- **활용**: 레이블 보완, NLP 기반 약한 지도학습

---

## ❌ 사용 불가 / 제한적 지표

| 지표 | 이유 |
|------|------|
| chartevents 전체 | 수억 건 → 직접 로드 불가, 청크 처리 필요 |
| Pleural Other 레이블 | 양성 2,011건으로 너무 적음 → 클래스 불균형 심각 |
| Support Devices 레이블 | 의료기기 존재 여부 → 폐 질환과 직접 관련 없음 |
| Fracture 레이블 | 폐 질환 아님, 골절 → 별도 모델 필요 |
| Enlarged Cardiomediastinum | 불확실(-1)이 양성(1)보다 많음 → 레이블 노이즈 |
| mimic-iv-note/discharge.csv | 텍스트 전처리 복잡, 익명화로 정보 손실 |
| ICU 전용 지표 (PEEP, Plateau 등) | ICU 환자만 해당, 일반 입원 환자 없음 |

---

## 🎯 프로젝트 권장 데이터 파이프라인

```
mimic-cxr-2.0.0-chexpert.csv  ← 레이블 (정답)
        │
        ├── subject_id + study_id로 연계
        │
        ├── patients.csv        ← 나이, 성별
        ├── admissions.csv      ← 입원 유형, 사망 여부
        ├── diagnoses_icd.csv   ← ICD 진단 코드 (코호트 필터링)
        └── radiology.csv       ← 판독 텍스트 (NLP 보조)

ICU 환자 서브셋:
        ├── icustays.csv        ← ICU 재원 기간
        └── chartevents.csv     ← SpO2, RR, FiO2, PEEP (청크 처리)
```

---

## 📊 핵심 지표 요약

| 구분 | 지표 | 출처 | 우선순위 |
|------|------|------|---------|
| 모델 레이블 | 14개 CheXpert 라벨 | mimic-cxr-chexpert.csv | ⭐⭐⭐ 필수 |
| 산소화 | SpO2, PaO2, FiO2, P/F ratio | chartevents, labevents | ⭐⭐⭐ 핵심 |
| 환기 | RR, Tidal Volume, PEEP, Plateau | chartevents | ⭐⭐ 중요 |
| 감염 | WBC, CRP, Procalcitonin | labevents | ⭐⭐ 중요 |
| 환자 기본 | 나이, 성별, 입원 유형 | patients, admissions | ⭐⭐ 중요 |
| 진단 코드 | ICD 폐 관련 코드 | diagnoses_icd | ⭐ 보조 |
| 텍스트 | 영상 판독 보고서 | radiology.csv | ⭐ 보조 |
