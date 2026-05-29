# Rare-Link AI · Frontend

**SKKU AWS SAY 2기 · 2팀** · Clinical Decision Support for Rare Pulmonary Disease
Final Phase · Week 1 (2026.04.20 ~ 04.26)

React 18 + Vite + Tailwind CSS 기반 프론트엔드 프로토타입입니다.
이 리포지토리는 `Rare-Link-AI_UIUX_설계제안서.pdf` (박성수, 2026-04-21)의
설계 원칙을 React로 실구현한 것입니다.

---

## 📦 포함 내용

| 화면 | 파일 | 상태 |
|---|---|---|
| 디자인 시스템 v0.1 | `src/DesignSystem.jsx` | ✓ 완료 |
| 로그인 | `src/LoginWorklist.jsx` | ✓ 완료 |
| 당일 외래 환자 목록 (Worklist) | `src/LoginWorklist.jsx` | ✓ 완료 |
| 환자 프리뷰 드로어 | `src/LoginWorklist.jsx` | ✓ 완료 |
| 진단 워크스페이스 (3-Panel) | _W3 예정_ | — |
| 질환 상세 (LR 막대) | _W3 예정_ | — |
| 이미지 확대 뷰어 | _W3 예정_ | — |
| 리포트 뷰어 | _W4 예정_ | — |

---

## 🚀 시작하기

### 사전 요구사항

- **Node.js 18 이상** ([nodejs.org](https://nodejs.org/ko)에서 LTS 설치)
- npm (Node.js 설치 시 자동 포함)
- 권장 에디터: VS Code + 확장 `Tailwind CSS IntelliSense`, `ES7+ React`

설치 확인:

```bash
node --version   # v18.x.x 이상이어야 함
npm --version
```

### 1. 의존성 설치

프로젝트 루트 디렉토리에서:

```bash
npm install
```

(처음 1~2분 걸림. `node_modules` 폴더가 생성됩니다.)

### 2. 개발 서버 실행

```bash
npm run dev
```

자동으로 브라우저가 열리고 **`http://localhost:5173`** 에서 앱이 뜹니다.

기본적으로 **로그인 화면**이 먼저 보입니다. 아무 값이나 둔 채로
`로그인` 또는 `EMR에서 실행` 버튼을 누르면 워크리스트로 진입합니다.

### 3. 뷰 전환 (개발용 스위처)

화면 우측 하단에 **`APP ↔ DESIGN SYSTEM`** 토글 버튼이 있습니다.
디자인 시스템 쇼케이스와 실제 앱 사이를 자유롭게 오갈 수 있습니다.

실전 배포 시에는 `src/App.jsx`의 스위처 블록을 제거하거나
환경변수로 제어하세요.

---

## 🛠 수정 방법

### 실시간 반영 (HMR)

`npm run dev` 실행 중에는 **파일 저장 → 브라우저 자동 반영** (Hot Module
Replacement)이 작동합니다. 새로고침 필요 없음.

### 자주 수정할 파일

| 수정 목적 | 파일 |
|---|---|
| **컬러/폰트/토큰 변경** | `tailwind.config.js` + 각 파일 내 `:root` CSS 변수 |
| **환자 목록 데이터 변경** | `src/LoginWorklist.jsx` 하단 `MOCK_PATIENTS` 배열 |
| **의사 정보 변경** | `src/LoginWorklist.jsx` 내 `LoginScreen` 컴포넌트의 `handleLogin` 하드코딩 |
| **새 화면 추가** | `src/` 에 새 `.jsx` 파일 생성 후 `App.jsx` 에서 import |

### 디자인 토큰 원칙

- **컬러는 `:root` CSS 변수로 관리** → `var(--rl-primary)` 형태로 사용
- Tailwind 유틸리티 클래스와 병용 (`className` + inline `style`)
- 새 컬러 추가 시 양쪽 다 수정: `tailwind.config.js` 의 `colors.rl`, 그리고 JSX 파일 내 `globalStyles` / `<style>` 의 `:root`

---

## 🏗 빌드 & 배포

### 프로덕션 빌드

```bash
npm run build
```

`dist/` 폴더에 정적 파일 생성. 이걸 그대로 어떤 정적 호스팅에라도 올릴 수 있음.

### 팀 배포 옵션

| 옵션 | 난이도 | 장점 |
|---|---|---|
| **AWS Amplify** (권장) | ★★ | 팀 AWS 계정 연결, CI/CD 내장, 커스텀 도메인 |
| **AWS S3 + CloudFront** | ★★★ | 완전 통제, 비용 최저 |
| **Vercel** | ★ | GitHub 연결 후 자동 배포, 무료 tier |
| **Netlify** | ★ | Vercel과 유사, 간편 |

AWS Amplify 배포 (빠른 가이드):

```bash
npm install -g @aws-amplify/cli
amplify init
amplify add hosting   # Hosting with Amplify Console 선택
amplify publish
```

---

## 📝 개발 로드맵 (5주)

| 주차 | 기간 | 작업 |
|---|---|---|
| **W1** (현재) | 04/20 ~ 04/26 | 디자인 시스템 · 로그인 · Worklist ✓ |
| W2 | 04/27 ~ 05/03 | 진단 워크스페이스 · SMART 샌드박스 세팅 |
| W3 | 05/04 ~ 05/10 | SageMaker 연결 · LR 상세 · 이미지 뷰어 |
| W4 | 05/11 ~ 05/17 | Heatmap · RAG · 리포트 · 폴리싱 |
| W5 | 05/18 ~ 05/24 | 시나리오 · 녹화 · 발표 연동 |
| 발표 | 05/25 ~ 05/28 | 리허설 + 최종 발표 |

---

## 📚 주요 설계 근거

전체 근거는 `Rare-Link-AI_UIUX_설계제안서.pdf` 참조.

- **3-Panel Layout**: Epic Hyperdrive · BESTCare 2.0 공통 패턴
- **LR 막대 (지지/반박)**: Robinson et al. *Am J Hum Genet* 2020;107:403-417, Fig. 2
- **"Don't miss" 플래그**: Ramnarayan et al. *BMC Med Inform Decis Mak* 2003 (Isabel)
- **HITL 배너**: EU AI Act (2024/1680/EU) Article 22
- **Explainability 기본값 ON**: Neri et al. *Radiol Med* 2023;128:755-764
- **PACS 오버레이 레이아웃**: Baltruschat et al. *Eur Radiol* 2021;31:3837-3845

---

## ⚠ Disclaimer

본 시스템은 **연구·교육용 프로토타입**이며, 아직 식약처 SaMD 또는
FDA 승인을 받지 않은 상태입니다. 실제 임상 진료에 사용할 수 없습니다.
모든 AI 출력은 주치의의 검토를 거쳐야 하며 치료 결정의 단독 근거로
사용될 수 없습니다.

---

## 🙋 팀

- **Frontend Lead**: 박성수
- **ML / Infra**: 배기태, 허태웅
- **Data**: 권미라, 양희인
- **AWS Mentor**: 이희찬

© 2026 SKKU AWS SAY 2기 · 2팀 · Rare-Link AI
