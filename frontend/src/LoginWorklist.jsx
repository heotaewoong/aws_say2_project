import React, { useState, useEffect, useRef } from 'react';
import { Stethoscope, LogIn, LogOut, User, UserPlus, Calendar, AlertTriangle, Flame, Activity, Microscope, Clock, CheckCircle2, CircleDot, Circle, Search, Filter, Bell, ChevronRight, ChevronLeft, Shield, Zap, X, ScanLine, FileText, ArrowUpRight, ListFilter, MoreHorizontal, Loader2, Inbox, Users, CalendarDays, Eye, ChevronDown, Home, Ban, FlaskConical, Settings as SettingsIcon, BookOpen, BarChart3, Languages, Database, KeyRound, HelpCircle, Mail, Volume2, Sliders, Palette, RefreshCw, Megaphone } from 'lucide-react';

/* ============================================================
   SESSION · sessionStorage 임시 토큰 (환자 정보는 절대 저장 X)
   TTL: 1시간 · key: rl-session · doctor 프로파일 + issuedAt + expiresAt
   ============================================================ */
const SESSION_KEY = 'rl-session';
const SESSION_TTL_MS = 60 * 60 * 1000; // 1 hour

function loadSession() {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (!raw) return { doctor: null, expired: false };
    const s = JSON.parse(raw);
    if (!s.expiresAt || Date.now() > s.expiresAt) {
      sessionStorage.removeItem(SESSION_KEY);
      return { doctor: null, expired: true };
    }
    return { doctor: s.doctor, expired: false, expiresAt: s.expiresAt };
  } catch {
    return { doctor: null, expired: false };
  }
}

function saveSession(doctor) {
  const now = Date.now();
  const s = { doctor, issuedAt: now, expiresAt: now + SESSION_TTL_MS };
  try { sessionStorage.setItem(SESSION_KEY, JSON.stringify(s)); } catch {}
  return s;
}

function clearSession() {
  try { sessionStorage.removeItem(SESSION_KEY); } catch {}
}

function renewSession() {
  const s = loadSession();
  if (!s.doctor) return false;
  saveSession(s.doctor);
  return true;
}

/* ----- TOPBAR · 알림 (Bell + dropdown) ----- */
const MOCK_NOTIFICATIONS = [
  // 외래 추가 등록
  { type: 'admit', date: '2026-04-23', time: '08:55', title: '신규 외래 등록', text: '원○○ · 22 F · 14:00 슬롯 (재진)',          category: 'admit', patientMrn: '23-145220',
    detail: { patient: '원○○ (23-145220)', source: '현장 접수', slot: '14:00 재진', complaint: '과호흡 · 두근거림 · 시험 기간 스트레스', referrer: 'walk-in' } },
  { type: 'admit', date: '2026-04-23', time: '08:42', title: '응급 콜인',       text: '서○○ · 56 M · 응급실 → 호흡기내과 의뢰',  category: 'admit', patientMrn: '24-002188',
    detail: { patient: '서○○ (24-002188)', source: '응급실 의뢰', slot: '11:30 추가', complaint: '급성 호흡곤란 · SpO₂ 88%', referrer: '응급의학과 김재현 전공의' } },

  // 결과 도착 (CXR / AI / Lab 각 항목)
  { type: 'cxr', date: '2026-04-23', time: '09:02', title: 'CXR 도착', text: '정○○ · LAM workup · Frontal',           category: 'result', patientMrn: '22-089433',
    detail: { patient: '정○○ (22-089433)', modality: 'CR Frontal', studyId: 'STU-2026-0423-0042', size: '448×448 (resized)', technician: '영상의학과' } },
  { type: 'ai',  date: '2026-04-23', time: '08:58', title: 'AI 분석 완료', text: '정○○ · LAM 58% · 희귀 의심',         category: 'result', patientMrn: '22-089433',
    detail: { patient: '정○○ (22-089433)', model: 'DenseNet-121 v2.3.1', latencyMs: 642, top: 'LAM 58% · PLCH 31% · idiopathic 18%', flags: '희귀' } },
  { type: 'lab', date: '2026-04-23', time: '08:48', title: 'Lab 결과 도착', text: '이○○ · CRP 0.8 (high) · CBC 정상', category: 'result', patientMrn: '21-093127',
    detail: { patient: '이○○ (21-093127)', drawnAt: '07:55', panels: 'CBC · Chem · Inflammation', abnormal: 'CRP 0.8 (↑) · ESR 38 (↑)', remark: 'Pneumonia 패턴 합치' } },
  { type: 'ai',  date: '2026-04-23', time: '08:21', title: 'AI 분석 완료', text: "김○○ · IPF 84% · Don't miss 플래그", category: 'result', patientMrn: '20-145982',
    detail: { patient: '김○○ (20-145982)', model: 'DenseNet-121 v2.3.1', latencyMs: 718, top: 'IPF 84% · Sarcoidosis 62% · HP 41%', flags: "Don't miss · 희귀 (ORPHA:2032)" } },
  { type: 'lab', date: '2026-04-23', time: '07:55', title: 'Lab 결과 도착', text: '장○○ · KL-6 1284 (critical)',      category: 'result', patientMrn: '22-145103',
    detail: { patient: '장○○ (22-145103)', drawnAt: '07:12', panels: 'CBC · Chem · ABG · Markers', abnormal: 'KL-6 1284 (↑↑) · SP-D 178 (↑) · RF 14', remark: 'RA-ILD 패턴 강력 시사' } },
  { type: 'cxr', date: '2026-04-23', time: '07:42', title: 'CXR 도착', text: '김○○ · IPF workup · Frontal',           category: 'result', patientMrn: '20-145982',
    detail: { patient: '김○○ (20-145982)', modality: 'CR Frontal', studyId: 'STU-2026-0423-0021', size: '448×448 (resized)', technician: '영상의학과' } },

  // 시스템
  { type: 'sys', date: '2026-04-23', time: '07:00', title: '모델 업데이트 배포', text: 'DenseNet-121 v2.3.1 (재학습 2026-03-15)',  category: 'system',
    detail: { component: 'DenseNet-121 SageMaker endpoint', version: 'v2.3.1', changes: '재학습 (NIH ChestX-ray14 + MIMIC-CXR-JPG · 2026-03-15) · 미세 분류 정확도 +1.8%', deployedBy: '배기태 · 허태웅' } },
  { type: 'sys', date: '2026-04-23', time: '06:30', title: 'HPO DB 갱신',         text: '2026-03-01 release · 12,847 terms',         category: 'system',
    detail: { component: 'HPO Knowledge Base', version: '2026-03-01', changes: '신규 term 142개 · 매핑 변경 38개', deployedBy: '권미라 · 양희인' } },
  { type: 'sys', date: '2026-04-23', time: '06:00', title: 'FHIR 정기 점검 완료', text: 'SMART Health IT sandbox 06:00–06:15',       category: 'system',
    detail: { component: 'SMART on FHIR sandbox', version: 'v2.2', changes: '정기 점검 · 다운타임 15분', deployedBy: 'AWS infra' } },
];

/* 전체 알림 히스토리 · 어제 ~ 며칠 전 (popup용) */
const MOCK_NOTIFICATION_HISTORY = [
  ...MOCK_NOTIFICATIONS,

  // 2026-04-22
  { type: 'admit', date: '2026-04-22', time: '13:48', title: '신규 외래 등록', text: '강○○ · 73 M · 14:00 슬롯 (재진)', category: 'admit', patientMrn: '15-228714',
    detail: { patient: '강○○ (15-228714)', source: '예약', slot: '14:00 재진', complaint: '만성 호흡곤란 · 흡연력 50 pack-year', referrer: '본인 예약' } },
  { type: 'ai',  date: '2026-04-22', time: '13:48', title: 'AI 분석 완료', text: '강○○ · COPD GOLD III 79%', category: 'result', patientMrn: '15-228714',
    detail: { patient: '강○○ (15-228714)', model: 'DenseNet-121 v2.3.1', latencyMs: 691, top: 'COPD III 79% · Bronchiectasis 34%', flags: '없음' } },
  { type: 'lab', date: '2026-04-22', time: '13:14', title: 'Lab 결과 도착', text: '강○○ · ABG 정상 · CRP 정상', category: 'result', patientMrn: '15-228714',
    detail: { patient: '강○○ (15-228714)', drawnAt: '12:48', panels: 'CBC · Chem · ABG', abnormal: '없음', remark: 'GOLD III 안정기' } },
  { type: 'sys', date: '2026-04-22', time: '11:00', title: '운영 공지',       text: '내일(04-23) 06:00–06:15 FHIR sandbox 점검 예정', category: 'system',
    detail: { component: 'FHIR sandbox', version: '—', changes: '정기 유지 보수 안내', deployedBy: 'AWS infra' } },

  // 2026-04-21
  { type: 'admit', date: '2026-04-21', time: '08:30', title: '신규 외래 등록', text: '문○○ · 38 F · 10:30 슬롯 (재진)', category: 'admit', patientMrn: '20-118245',
    detail: { patient: '문○○ (20-118245)', source: '예약', slot: '10:30 재진', complaint: '활동 시 호흡곤란 · ground-glass FU', referrer: '본인 예약' } },
  { type: 'ai',  date: '2026-04-21', time: '10:12', title: 'AI 분석 완료', text: '문○○ · NSIP 66% · 희귀', category: 'result', patientMrn: '20-118245',
    detail: { patient: '문○○ (20-118245)', model: 'DenseNet-121 v2.3.1', latencyMs: 705, top: 'NSIP 66% · HP 39%', flags: '희귀 (ORPHA:79126)' } },
  { type: 'sys', date: '2026-04-21', time: '17:30', title: 'UI 업데이트',     text: 'Worklist 사이드바 리사이즈 기능 추가', category: 'system',
    detail: { component: 'Frontend', version: 'v0.1.0-rc2', changes: '환자 목록 사이드바 200~320px 드래그 리사이즈', deployedBy: '박성수' } },

  // 2026-04-20
  { type: 'sys', date: '2026-04-20', time: '09:00', title: '버전 배포',       text: 'v0.1.0 alpha · 디자인 시스템 · 로그인 · 워크리스트', category: 'system',
    detail: { component: 'Rare-Link AI Frontend', version: 'v0.1.0-alpha', changes: 'Final Phase Week 1 시작 · 디자인 시스템 + 로그인 + 워크리스트 화면', deployedBy: '박성수' } },
  { type: 'admit', date: '2026-04-20', time: '14:00', title: '신규 외래 등록', text: '백○○ · 64 M · 13:30 슬롯', category: 'admit', patientMrn: '22-145210',
    detail: { patient: '백○○ (22-145210)', source: '예약', slot: '13:30 초진', complaint: '체중감소 · 객혈 · 림프절 종대', referrer: '1차의원 의뢰' } },

  // 2026-04-19
  { type: 'sys', date: '2026-04-19', time: '20:00', title: 'KB 갱신',         text: 'Orphadata 2026-Q1 (9,872 dx)', category: 'system',
    detail: { component: 'Orphadata KB', version: '2026-Q1', changes: '신규 희귀질환 코드 247건', deployedBy: '권미라 · 양희인' } },
  { type: 'sys', date: '2026-04-19', time: '15:00', title: '모델 평가 완료',  text: 'DenseNet-121 v2.3.1 ROC-AUC 0.92 (외부 검증)', category: 'system',
    detail: { component: 'Model evaluation', version: 'v2.3.1', changes: 'MIMIC-CXR-JPG 외부 검증 ROC-AUC 0.92 (CI 0.90-0.94)', deployedBy: '배기태' } },
];

function NotificationButton({ onOpenPatient, onOpenAnnouncement }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    const onEsc = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onEsc);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onEsc);
    };
  }, [open]);

  const total = MOCK_NOTIFICATIONS.length;

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button
        onClick={() => setOpen(o => !o)}
        className="relative p-1.5 rounded transition hover:bg-slate-100"
        style={{ color: open ? 'var(--rl-primary)' : 'var(--rl-ink-2)' }}
        title={`알림 ${total}건`}
      >
        <Bell size={16} />
        {total > 0 && (
          <div
            className="absolute top-0 right-0 rounded-full font-mono flex items-center justify-center"
            style={{
              minWidth: 14, height: 14, padding: '0 3px',
              background: 'var(--rl-critical)', color: 'white',
              fontSize: 9, fontWeight: 600,
            }}
          >
            {total}
          </div>
        )}
      </button>
      {open && <NotificationPanel onClose={() => setOpen(false)} onOpenPatient={onOpenPatient} onOpenAnnouncement={onOpenAnnouncement} />}
    </div>
  );
}

function NotificationPanel({ onClose, onOpenPatient, onOpenAnnouncement }) {
  const grouped = {
    admit:  MOCK_NOTIFICATIONS.filter(n => n.category === 'admit'),
    result: MOCK_NOTIFICATIONS.filter(n => n.category === 'result'),
    system: MOCK_NOTIFICATIONS.filter(n => n.category === 'system'),
  };
  const total = MOCK_NOTIFICATIONS.length;

  const handleClick = (n) => {
    if (n.category === 'system') {
      if (onOpenAnnouncement) onOpenAnnouncement(n);
      else openSystemAnnouncementPopup(n); // fallback
    } else if (n.patientMrn && onOpenPatient) {
      onOpenPatient(n.patientMrn);
    }
    onClose();
  };

  return (
    <div
      className="absolute bg-white rounded fade-in"
      style={{
        top: 38, right: 0, width: 360,
        maxHeight: 380,
        boxShadow: '0 12px 36px rgba(10,22,40,0.18)',
        border: '1px solid var(--rl-border)',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 100,
      }}
    >
      {/* Header */}
      <div className="px-4 py-2.5 flex items-center" style={{ borderBottom: '1px solid var(--rl-border-soft)', flexShrink: 0 }}>
        <div className="font-mono text-[10px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)' }}>
          Notifications
        </div>
        <div className="text-sm font-medium ml-2" style={{ color: 'var(--rl-ink)' }}>알림</div>
        <span className="ml-auto font-mono text-[10px]" style={{ color: 'var(--rl-amber)' }}>{total}건</span>
      </div>

      {/* Scrollable body · sticky group headers · 4-5 rows visible */}
      <div style={{ overflowY: 'auto', flex: 1, minHeight: 0 }}>
        <NotifGroup label="외래 추가 등록" mono="Admit"  items={grouped.admit}  onClick={handleClick} />
        <NotifGroup label="결과 도착"      mono="Result" items={grouped.result} onClick={handleClick} />
        <NotifGroup label="시스템"          mono="System" items={grouped.system} onClick={handleClick} />
      </div>

      {/* Footer */}
      <div className="px-3 py-2 flex items-center gap-3" style={{ borderTop: '1px solid var(--rl-border-soft)', flexShrink: 0 }}>
        <button
          className="font-mono text-[10px] uppercase tracking-widest hover:underline"
          style={{ color: 'var(--rl-primary)' }}
        >
          모두 읽음 처리
        </button>
        <button
          onClick={() => { openNotificationHistoryPopup(); onClose(); }}
          className="font-mono text-[10px] uppercase tracking-widest hover:underline flex items-center gap-1"
          style={{ color: 'var(--rl-ink-2)' }}
          title="전체 알림 히스토리 새 창"
        >
          <Clock size={10} /> 히스토리
        </button>
        <button
          onClick={onClose}
          className="ml-auto font-mono text-[10px] uppercase tracking-widest hover:underline"
          style={{ color: 'var(--rl-ink-3)' }}
        >
          닫기
        </button>
      </div>
    </div>
  );
}

function NotifGroup({ label, mono, items, onClick }) {
  if (!items || items.length === 0) return null;
  return (
    <div>
      <div
        className="px-4 py-1.5 flex items-baseline gap-2"
        style={{
          background: 'var(--rl-bg-3)',
          borderBottom: '1px solid var(--rl-border-soft)',
          position: 'sticky', top: 0, zIndex: 1,
        }}
      >
        <span className="font-mono text-[9px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)' }}>{mono}</span>
        <span className="text-[11px] font-medium" style={{ color: 'var(--rl-ink-2)' }}>{label}</span>
        <span className="ml-auto font-mono text-[10px]" style={{ color: 'var(--rl-ink-3)' }}>{items.length}건</span>
      </div>
      {items.map((n, i) => <NotifRow key={i} n={n} onClick={() => onClick && onClick(n)} />)}
    </div>
  );
}

const NOTIF_ICON = {
  admit: { icon: UserPlus,     color: 'var(--rl-primary)' },
  cxr:   { icon: ScanLine,     color: 'var(--rl-teal)' },
  ai:    { icon: Microscope,   color: 'var(--rl-amber)' },
  lab:   { icon: FlaskConical, color: 'var(--rl-teal)' },
  sys:   { icon: SettingsIcon, color: 'var(--rl-ink-3)' },
};

function NotifRow({ n, onClick }) {
  const meta = NOTIF_ICON[n.type] || NOTIF_ICON.sys;
  const Icon = meta.icon;
  return (
    <div
      onClick={onClick}
      className="px-4 py-2 flex items-start gap-2.5 transition hover:bg-slate-50 cursor-pointer"
      style={{ borderBottom: '1px solid var(--rl-border-soft)' }}
      title={n.category === 'system' ? '시스템 공지 새 창' : '환자 차트로 이동'}
    >
      <div style={{ flexShrink: 0, marginTop: 2 }}>
        <Icon size={13} style={{ color: meta.color }} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <div className="text-xs font-medium truncate" style={{ color: 'var(--rl-ink)' }}>{n.title}</div>
          <div className="font-mono text-[10px] ml-auto flex-shrink-0" style={{ color: 'var(--rl-ink-3)' }}>{n.time}</div>
        </div>
        <div className="text-[11px] truncate mt-0.5" style={{ color: 'var(--rl-ink-3)' }}>{n.text}</div>
      </div>
      <ChevronRight size={11} style={{ color: 'var(--rl-ink-4)', flexShrink: 0, marginTop: 4 }} />
    </div>
  );
}

/* 시스템 공지 popup · 단일 알림 상세 */
function openSystemAnnouncementPopup(n) {
  const w = window.open('', `sys-${n.date}-${n.time}`, 'width=720,height=620,resizable=yes,scrollbars=yes');
  if (!w) {
    alert('팝업 차단을 해제해주세요.');
    return;
  }
  const d = n.detail || {};
  w.document.write(`<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<title>시스템 공지 · ${n.title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Serif:wght@500&family=IBM+Plex+Mono&display=swap" rel="stylesheet" />
<style>
  html, body { margin: 0; padding: 0; }
  body { background: #F8FAFC; font-family: 'IBM Plex Sans KR', sans-serif; color: #0A1628; padding: 24px; -webkit-font-smoothing: antialiased; }
  .card { background: white; max-width: 640px; margin: 0 auto; border: 1px solid #E2E8F0; border-radius: 6px; padding: 28px; }
  .label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em; color: #64748B; }
  .top { display: flex; align-items: baseline; gap: 12px; padding-bottom: 12px; border-bottom: 1px solid #E2E8F0; }
  .top .badge { padding: 3px 10px; background: #F1F5F9; color: #334155; border-radius: 3px; font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em; }
  .top .when { margin-left: auto; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #64748B; }
  h1 { font-family: 'IBM Plex Serif', serif; font-size: 22px; margin: 14px 0 6px; letter-spacing: -0.01em; }
  .text { font-size: 13px; color: #334155; line-height: 1.6; margin-bottom: 16px; }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 12px 0 16px; }
  .stat { padding: 10px 12px; background: #F1F5F9; border-radius: 4px; }
  .stat .l { font-family: 'IBM Plex Mono', monospace; font-size: 9px; text-transform: uppercase; letter-spacing: 0.15em; color: #64748B; }
  .stat .v { font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: #0A1628; margin-top: 3px; }
  .changes {
    padding: 14px 16px; background: #EFF4FB; border-left: 3px solid #0C447C; border-radius: 4px;
    font-size: 12px; line-height: 1.6;
  }
  .changes .l { font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em; color: #0C447C; margin-bottom: 6px; }
  .footer { margin-top: 18px; padding-top: 12px; border-top: 1px solid #E2E8F0; display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 9px; text-transform: uppercase; letter-spacing: 0.15em; color: #94A3B8; }
</style>
</head>
<body>
  <div class="card">
    <div class="top">
      <span class="badge">System</span>
      <span class="label">${d.component || ''}</span>
      <span class="when">${n.date} ${n.time} KST</span>
    </div>
    <h1>${n.title}</h1>
    <div class="text">${n.text}</div>
    <div class="stats">
      ${d.version    ? `<div class="stat"><div class="l">Version</div><div class="v">${d.version}</div></div>` : ''}
      ${d.deployedBy ? `<div class="stat"><div class="l">Deployed by</div><div class="v">${d.deployedBy}</div></div>` : ''}
    </div>
    ${d.changes ? `<div class="changes"><div class="l">Changes · 변경 내역</div>${d.changes}</div>` : ''}
    <div class="footer">
      <span>Rare-Link AI · System announcement</span>
      <span>EU AI Act Art. 22</span>
    </div>
  </div>
</body>
</html>`);
  w.document.close();
}

/* 알림 히스토리 popup · 전체 알림 리스트 + details/summary 클릭 expand */
function openNotificationHistoryPopup() {
  const w = window.open('', 'notif-history', 'width=820,height=900,resizable=yes,scrollbars=yes');
  if (!w) {
    alert('팝업 차단을 해제해주세요.');
    return;
  }

  const CAT_LABEL = { admit: '외래 추가 등록', result: '결과 도착', system: '시스템' };
  const TYPE_LABEL = {
    admit: { label: '외래 등록', color: '#0C447C', bg: '#EFF4FB' },
    cxr:   { label: 'CXR',       color: '#0E8574', bg: '#E6F5F2' },
    ai:    { label: 'AI',        color: '#B45309', bg: '#FEF3C7' },
    lab:   { label: 'Lab',       color: '#0E8574', bg: '#E6F5F2' },
    sys:   { label: '시스템',    color: '#64748B', bg: '#F1F5F9' },
  };

  const all = MOCK_NOTIFICATION_HISTORY.slice().sort((a, b) => {
    const da = `${a.date} ${a.time}`;
    const db = `${b.date} ${b.time}`;
    return db.localeCompare(da);
  });

  const counts = {
    total:  all.length,
    admit:  all.filter(n => n.category === 'admit').length,
    result: all.filter(n => n.category === 'result').length,
    system: all.filter(n => n.category === 'system').length,
  };

  const renderDetail = (n) => {
    const d = n.detail || {};
    const rows = Object.entries(d).map(([k, v]) => `
      <tr>
        <td class="dlabel">${k}</td>
        <td class="dval">${v}</td>
      </tr>`).join('');
    return `<table class="detail">${rows}</table>`;
  };

  const items = all.map((n, i) => {
    const t = TYPE_LABEL[n.type] || TYPE_LABEL.sys;
    return `
    <details class="item" data-cat="${n.category}">
      <summary>
        <span class="chip" style="background:${t.bg};color:${t.color}">${t.label}</span>
        <span class="title">${n.title}</span>
        <span class="text">${n.text}</span>
        <span class="when">${n.date} ${n.time}</span>
      </summary>
      <div class="body">
        <div class="cat mono small muted">CATEGORY · ${CAT_LABEL[n.category]}</div>
        ${renderDetail(n)}
      </div>
    </details>`;
  }).join('');

  w.document.write(`<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<title>알림 히스토리 · Rare-Link AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Serif:wght@500&family=IBM+Plex+Mono&display=swap" rel="stylesheet" />
<style>
  html, body { margin: 0; padding: 0; }
  body { background: #F8FAFC; font-family: 'IBM Plex Sans KR', sans-serif; color: #0A1628; padding: 24px; -webkit-font-smoothing: antialiased; }
  .card { background: white; max-width: 720px; margin: 0 auto; border: 1px solid #E2E8F0; border-radius: 6px; overflow: hidden; }
  header { padding: 18px 24px; border-bottom: 1px solid #E2E8F0; display: flex; align-items: baseline; gap: 12px; }
  header h1 { font-family: 'IBM Plex Serif', serif; font-size: 22px; margin: 0; letter-spacing: -0.01em; }
  header .label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em; color: #0C447C; }
  header .total { margin-left: auto; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #B45309; }
  .filters {
    padding: 10px 24px; border-bottom: 1px solid #E2E8F0; background: #F8FAFC;
    display: flex; align-items: center; gap: 6px;
  }
  .filters .lab { font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em; color: #64748B; margin-right: 4px; }
  .filters button {
    padding: 4px 10px; border-radius: 4px; border: 1px solid #CBD5E1; background: white; cursor: pointer;
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em;
    color: #334155;
  }
  .filters button.active { background: #0C447C; color: white; border-color: #0C447C; }
  .filters button:hover:not(.active) { background: #F1F5F9; }
  .list { padding: 4px 0; }
  details.item { border-bottom: 1px solid #E2E8F0; }
  details.item[hidden] { display: none; }
  details.item summary {
    padding: 10px 24px; cursor: pointer; display: flex; align-items: center; gap: 10px;
    list-style: none; font-size: 12px;
  }
  details.item summary::-webkit-details-marker { display: none; }
  details.item summary:hover { background: #F8FAFC; }
  details.item[open] summary { background: #EFF4FB; }
  .chip {
    display: inline-block; padding: 2px 8px; border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace; font-size: 9px;
    text-transform: uppercase; letter-spacing: 0.1em; flex-shrink: 0;
    min-width: 56px; text-align: center;
  }
  .title { font-weight: 500; flex-shrink: 0; }
  .text { color: #64748B; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .when { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #94A3B8; flex-shrink: 0; }
  .body { padding: 10px 24px 14px 90px; background: #F8FAFC; border-top: 1px solid #E2E8F0; }
  .cat { margin-bottom: 6px; }
  .small { font-size: 10px; }
  .muted { color: #64748B; }
  .mono { font-family: 'IBM Plex Mono', monospace; }
  table.detail { width: 100%; border-collapse: collapse; }
  table.detail td { padding: 4px 0; vertical-align: top; }
  td.dlabel {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.1em; color: #64748B;
    width: 110px; padding-right: 12px;
  }
  td.dval { font-size: 11px; color: #0A1628; }
  footer {
    padding: 10px 24px; border-top: 1px solid #E2E8F0;
    font-family: 'IBM Plex Mono', monospace; font-size: 9px; text-transform: uppercase; letter-spacing: 0.15em; color: #94A3B8;
    display: flex; justify-content: space-between;
  }
</style>
</head>
<body>
  <div class="card">
    <header>
      <span class="label">Notification History</span>
      <h1>알림 히스토리</h1>
      <span class="total">전체 ${counts.total}건</span>
    </header>
    <div class="filters">
      <span class="lab">Filter</span>
      <button class="active" data-filter="all">전체 ${counts.total}</button>
      <button data-filter="admit">외래 등록 ${counts.admit}</button>
      <button data-filter="result">결과 도착 ${counts.result}</button>
      <button data-filter="system">시스템 ${counts.system}</button>
    </div>
    <div class="list">
      ${items}
    </div>
    <footer>
      <span>Rare-Link AI · 최근 5일</span>
      <span>EU AI Act Art. 22</span>
    </footer>
  </div>
  <script>
    (function() {
      var btns = document.querySelectorAll('.filters button');
      var items = document.querySelectorAll('details.item');
      btns.forEach(function(b) {
        b.addEventListener('click', function() {
          btns.forEach(function(x) { x.classList.remove('active'); });
          b.classList.add('active');
          var f = b.getAttribute('data-filter');
          items.forEach(function(it) {
            if (f === 'all' || it.getAttribute('data-cat') === f) {
              it.hidden = false;
            } else {
              it.hidden = true;
              it.removeAttribute('open');
            }
          });
        });
      });
    })();
  </script>
</body>
</html>`);
  w.document.close();
}

/* ----- TOPBAR · 세션 카운트다운 (클릭으로 연장) ----- */
function SessionCountdown() {
  const [remaining, setRemaining] = useState(() => {
    const s = loadSession();
    return s.expiresAt ? Math.max(0, s.expiresAt - Date.now()) : 0;
  });
  const [justRenewed, setJustRenewed] = useState(false);

  useEffect(() => {
    const tick = () => {
      const s = loadSession();
      setRemaining(s.expiresAt ? Math.max(0, s.expiresAt - Date.now()) : 0);
    };
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  if (remaining <= 0) return null;

  const min = Math.floor(remaining / 60000);
  const sec = Math.floor((remaining % 60000) / 1000);
  const formatted = `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;

  const level =
    min < 5  ? { c: 'var(--rl-critical)', bg: 'var(--rl-critical-soft)' } :
    min < 10 ? { c: 'var(--rl-amber)',    bg: 'var(--rl-amber-soft)' } :
               { c: 'var(--rl-ink-2)',    bg: 'var(--rl-bg-3)' };

  const extend = () => {
    if (renewSession()) {
      setRemaining(SESSION_TTL_MS);
      setJustRenewed(true);
      setTimeout(() => setJustRenewed(false), 900);
    }
  };

  return (
    <div className="flex items-center gap-1" title={`세션 잔여 ${min}분 ${sec}초 · TTL 1h`}>
      <div
        className="flex items-center gap-1.5 px-2 py-1 rounded font-mono text-[10px]"
        style={{ background: level.bg, color: level.c, transition: 'background 0.3s, color 0.3s' }}
      >
        <Clock size={11} />
        <span>{formatted}</span>
      </div>
      <button
        onClick={extend}
        className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-mono uppercase tracking-widest hairline-strong hover:bg-slate-50 transition"
        style={{
          color: justRenewed ? 'var(--rl-teal)' : 'var(--rl-primary)',
          borderColor: justRenewed ? 'var(--rl-teal)' : undefined,
        }}
        title="세션 1시간 연장"
      >
        <RefreshCw size={10} className={justRenewed ? '' : ''} />
        {justRenewed ? '연장됨' : '연장'}
      </button>
    </div>
  );
}

export default function RareLinkApp() {
  const initial = loadSession();
  const [screen, setScreen] = useState(initial.doctor ? 'worklist' : 'login');
  const [doctor, setDoctor] = useState(initial.doctor);
  const [sessionExpired, setSessionExpired] = useState(initial.expired);
  const [pendingPatientMrn, setPendingPatientMrn] = useState(null);

  // 만료 감시 · 1분마다 확인 (탭 재활성화 시 즉시 확인)
  useEffect(() => {
    if (screen === 'login') return;
    const check = () => {
      const s = loadSession();
      if (!s.doctor) {
        clearSession();
        setDoctor(null);
        setScreen('login');
        setSessionExpired(true);
      }
    };
    const id = setInterval(check, 60 * 1000);
    const onFocus = () => check();
    window.addEventListener('focus', onFocus);
    return () => { clearInterval(id); window.removeEventListener('focus', onFocus); };
  }, [screen]);

  const logout = () => {
    clearSession();
    setDoctor(null);
    setScreen('login');
  };

  // 알림 클릭 → 환자 차트로 이동 (다른 screen에 있으면 worklist로 navigate)
  const [pendingAnnouncement, setPendingAnnouncement] = useState(null);

  const openPatientByMrn = (mrn) => {
    if (!mrn) return;
    setPendingPatientMrn(mrn);
    setScreen('worklist');
  };
  const clearPendingPatient = () => setPendingPatientMrn(null);

  const openAnnouncement = (n) => {
    setPendingAnnouncement(n);
    setScreen('announcement');
  };

  const common = {
    doctor, onLogout: logout, onNavigate: setScreen,
    onOpenPatient: openPatientByMrn,
    onOpenAnnouncement: openAnnouncement,
  };

  return (
    <div className="min-h-screen" style={{ fontFamily: "'IBM Plex Sans KR', 'IBM Plex Sans', sans-serif", background: 'var(--rl-bg-2)' }}>
      <style>{globalStyles}</style>
      {screen === 'login' && (
        <LoginScreen
          sessionExpired={sessionExpired}
          onLogin={(d) => {
            saveSession(d);
            setDoctor(d);
            setSessionExpired(false);
            setScreen('worklist');
          }}
        />
      )}
      {screen === 'worklist'     && <WorklistScreen     {...common} pendingPatientMrn={pendingPatientMrn} onClearPendingPatient={clearPendingPatient} />}
      {screen === 'settings'     && <SettingsScreen     {...common} />}
      {screen === 'dashboard'    && <ComingSoonScreen   {...common} screenKey="dashboard" />}
      {screen === 'knowledge'    && <ComingSoonScreen   {...common} screenKey="knowledge" />}
      {screen === 'announcement' && <AnnouncementScreen {...common} initialNotif={pendingAnnouncement} />}
    </div>
  );
}

/* ============================================================
   GLOBAL DESIGN TOKENS (from Design System v0.1)
   ============================================================ */
const globalStyles = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@300;400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,500;0,600;1,400;1,500&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

  :root {
    --rl-ink:           #0A1628;
    --rl-ink-2:         #334155;
    --rl-ink-3:         #64748B;
    --rl-ink-4:         #94A3B8;
    --rl-border:        #CBD5E1;
    --rl-border-soft:   #E2E8F0;
    --rl-bg:            #FFFFFF;
    --rl-bg-2:          #F8FAFC;
    --rl-bg-3:          #F1F5F9;
    --rl-primary:       #0C447C;
    --rl-primary-dark:  #083158;
    --rl-primary-2:     #1D5FAB;
    --rl-primary-soft:  #EFF4FB;
    --rl-teal:          #0E8574;
    --rl-teal-soft:     #E6F5F2;
    --rl-amber:         #B45309;
    --rl-amber-soft:    #FEF3C7;
    --rl-critical:      #A32D2D;
    --rl-critical-soft: #FEE4E4;
    --rl-rare:          #6B21A8;
    --rl-rare-soft:     #F3E8FF;
  }

  .font-serif { font-family: 'IBM Plex Serif', Georgia, serif; }
  .font-mono  { font-family: 'IBM Plex Mono', monospace; }

  .hairline { border: 1px solid var(--rl-border-soft); }
  .hairline-strong { border: 1px solid var(--rl-border); }

  .chip {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 500; letter-spacing: 0.02em;
    white-space: nowrap;
  }

  @keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.4; }
  }
  .pulse-dot { animation: pulse-dot 2s ease-in-out infinite; }

  @keyframes sweep {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }

  @keyframes graph-pulse {
    0%, 100% { opacity: 0.7; }
    50%      { opacity: 0.25; }
  }

  @keyframes fade-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .fade-in { animation: fade-in 0.5s ease-out; }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  .spin { animation: spin 1s linear infinite; }

  .row-hover:hover { background: var(--rl-primary-soft); cursor: pointer; }

  .drawer-bg {
    animation: fade-in 0.25s ease-out;
  }
`;

/* ============================================================
   SCREEN 01 · LOGIN
   ============================================================ */
function LoginScreen({ onLogin, sessionExpired }) {
  const [institution, setInstitution] = useState('skku');
  const [doctorId, setDoctorId] = useState('jeong.ms');
  const [password, setPassword] = useState('••••••••••');
  const [loading, setLoading] = useState(false);
  const [phase, setPhase] = useState(null); // null | 'oauth' | 'fhir' | 'synthea'

  const phases = [
    { key: 'oauth',   label: 'OAuth 2.0 토큰 교환 중',  d: 500 },
    { key: 'fhir',    label: 'FHIR R4 서버 핸드셰이크', d: 700 },
    { key: 'synthea', label: 'Synthea 환자 코호트 로드', d: 600 },
  ];

  const handleLogin = () => {
    setLoading(true);
    let acc = 0;
    phases.forEach((p, i) => {
      acc += p.d;
      setTimeout(() => setPhase(p.key), acc - p.d);
      if (i === phases.length - 1) {
        setTimeout(() => onLogin({
          id: doctorId,
          name: '정민수',
          role: '호흡기내과 과장',
          institution: '성균관대학교병원',
          department: '호흡기내과',
        }), acc);
      }
    });
  };

  return (
    <div className="min-h-screen flex">
      {/* ============== LEFT: BRAND PANEL ============== */}
      <div className="flex-1 relative overflow-hidden hidden lg:flex flex-col justify-between p-12" style={{ background: 'var(--rl-primary-dark)', color: 'white' }}>
        {/* Grid backdrop */}
        <div className="absolute inset-0 opacity-30" style={{
          backgroundImage: 'linear-gradient(rgba(255,255,255,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.08) 1px, transparent 1px)',
          backgroundSize: '32px 32px',
        }} />

        {/* Sweeping light */}
        <div className="absolute top-1/3 left-0 right-0 h-32 opacity-15 pointer-events-none" style={{
          background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)',
          animation: 'sweep 8s ease-in-out infinite',
        }} />

        {/* Header */}
        <div className="relative z-10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded flex items-center justify-center" style={{ background: 'white' }}>
              <Stethoscope size={20} style={{ color: 'var(--rl-primary-dark)' }} strokeWidth={2.5} />
            </div>
            <div>
              <div className="font-serif text-xl leading-none" style={{ letterSpacing: '-0.01em' }}>
                Rare-Link <span style={{ fontStyle: 'italic', fontWeight: 500 }}>AI</span>
              </div>
              <div className="font-mono text-[10px] mt-1 uppercase tracking-widest opacity-60">
                Pulmonary Differential · SKKU AWS SAY 2기 · 2팀
              </div>
            </div>
          </div>
        </div>

        {/* Hero */}
        <div className="relative z-10">
          <div className="font-mono text-[10px] uppercase tracking-widest opacity-60 mb-4">
            Clinical Decision Support for Rare Pulmonary Disease
          </div>
          <h1 className="font-serif leading-tight mb-6" style={{ fontSize: '3.2rem', letterSpacing: '-0.02em' }}>
            의사가 <span style={{ fontStyle: 'italic' }}>이미 아는</span><br />
            언어로, 희귀질환을<br />
            놓치지 않게
          </h1>
          <p className="text-base leading-relaxed max-w-md opacity-80">
            DenseNet-121 흉부 X-선 모델과 HPO 기반 Likelihood Ratio 엔진이 <span className="font-serif italic">528</span>개 폐질환 중 임상 증거로 가장 뒷받침되는 감별진단을 제시합니다.
          </p>
        </div>

        {/* HPO Graph visual */}
        <div className="relative z-10 mt-auto">
          <HPOGraph />

          {/* Compliance strip */}
          <div className="flex items-center gap-3 mt-8 flex-wrap">
            {[
              { icon: <Shield size={12} />, label: 'EU AI Act · Art. 22' },
              { icon: <Shield size={12} />, label: 'FDA SaMD Framework' },
              { icon: <Zap size={12} />,    label: 'SMART on FHIR v2.2' },
              { icon: <Shield size={12} />, label: 'HIPAA · 개인정보보호법' },
            ].map(b => (
              <div key={b.label} className="flex items-center gap-1.5 px-2.5 py-1 rounded-sm" style={{ background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.12)' }}>
                {b.icon}
                <span className="font-mono text-[10px] uppercase tracking-wider opacity-80">{b.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ============== RIGHT: FORM PANEL ============== */}
      <div className="w-full lg:w-[480px] flex flex-col justify-center p-8 lg:p-12 bg-white">
        <div className="w-full max-w-sm mx-auto">
          <div className="font-mono text-[10px] uppercase tracking-widest mb-2" style={{ color: 'var(--rl-primary)' }}>
            Clinician Login · 2026.04.23 (Thu)
          </div>
          <h2 className="font-serif text-3xl mb-1" style={{ color: 'var(--rl-ink)', letterSpacing: '-0.01em' }}>로그인</h2>
          <p className="text-sm mb-5" style={{ color: 'var(--rl-ink-3)' }}>
            의사 계정 또는 EMR SSO로 접속하세요.
          </p>

          {sessionExpired && (
            <div
              className="rounded px-3 py-2 mb-5 flex items-start gap-2 text-xs"
              style={{ background: 'var(--rl-amber-soft)', border: '1px solid var(--rl-amber)' }}
            >
              <Clock size={13} style={{ color: 'var(--rl-amber)', marginTop: 1, flexShrink: 0 }} />
              <div style={{ color: 'var(--rl-ink-2)' }}>
                <span className="font-medium" style={{ color: 'var(--rl-amber)' }}>세션이 만료되어 로그아웃되었습니다.</span>{' '}
                <span className="font-mono text-[10px]" style={{ color: 'var(--rl-ink-3)' }}>TTL 1h</span> · 다시 로그인해주세요.
              </div>
            </div>
          )}

          {/* Institution */}
          <Field label="소속 기관" icon={<Activity size={14} />}>
            <select
              value={institution}
              onChange={e => setInstitution(e.target.value)}
              className="w-full px-3 py-2.5 rounded bg-white text-sm outline-none hairline-strong focus:border-[color:var(--rl-primary)]"
              style={{ color: 'var(--rl-ink)' }}
              disabled={loading}
            >
              <option value="skku">성균관대학교병원 · 호흡기내과</option>
              <option value="demo">AWS SAY 데모 병원</option>
              <option value="sandbox">SMART Health IT Sandbox</option>
            </select>
          </Field>

          {/* Doctor ID */}
          <Field label="의사 ID" icon={<User size={14} />}>
            <input
              value={doctorId}
              onChange={e => setDoctorId(e.target.value)}
              className="w-full px-3 py-2.5 rounded text-sm outline-none hairline-strong focus:border-[color:var(--rl-primary)]"
              style={{ color: 'var(--rl-ink)' }}
              disabled={loading}
            />
          </Field>

          {/* Password */}
          <Field label="비밀번호" icon={<Shield size={14} />}>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              className="w-full px-3 py-2.5 rounded text-sm outline-none hairline-strong focus:border-[color:var(--rl-primary)] font-mono"
              style={{ color: 'var(--rl-ink)' }}
              disabled={loading}
            />
          </Field>

          {/* Primary Login */}
          <button
            onClick={handleLogin}
            disabled={loading}
            className="w-full py-3 rounded mt-2 text-sm font-medium flex items-center justify-center gap-2 transition hover:opacity-90"
            style={{ background: 'var(--rl-primary)', color: 'white', opacity: loading ? 0.9 : 1 }}
          >
            {loading ? <Loader2 size={16} className="spin" /> : <LogIn size={16} />}
            {loading ? '연결 중' : '로그인'}
          </button>

          {/* Divider */}
          <div className="flex items-center gap-3 my-5">
            <div className="flex-1 h-px" style={{ background: 'var(--rl-border-soft)' }} />
            <span className="font-mono text-[10px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)' }}>또는</span>
            <div className="flex-1 h-px" style={{ background: 'var(--rl-border-soft)' }} />
          </div>

          {/* SMART SSO */}
          <button
            onClick={handleLogin}
            disabled={loading}
            className="w-full py-3 rounded hairline-strong text-sm font-medium flex items-center justify-center gap-2 transition hover:bg-slate-50"
            style={{ color: 'var(--rl-primary)' }}
          >
            <Zap size={15} />
            EMR에서 실행 · SMART on FHIR SSO
          </button>

          {/* Loading phases */}
          {loading && (
            <div className="mt-6 space-y-2 fade-in">
              {phases.map(p => {
                const done = phases.findIndex(x => x.key === phase) > phases.findIndex(x => x.key === p.key);
                const active = phase === p.key;
                return (
                  <div key={p.key} className="flex items-center gap-2 text-xs">
                    {done ? (
                      <CheckCircle2 size={14} style={{ color: 'var(--rl-teal)' }} />
                    ) : active ? (
                      <Loader2 size={14} className="spin" style={{ color: 'var(--rl-primary)' }} />
                    ) : (
                      <Circle size={14} style={{ color: 'var(--rl-border)' }} />
                    )}
                    <span style={{ color: active || done ? 'var(--rl-ink)' : 'var(--rl-ink-3)' }}>{p.label}</span>
                  </div>
                );
              })}
            </div>
          )}

          {/* Footer disclaimer */}
          <div className="mt-10 pt-5 text-[11px] leading-relaxed" style={{ borderTop: '1px solid var(--rl-border-soft)', color: 'var(--rl-ink-3)' }}>
            <div className="font-mono uppercase tracking-widest text-[10px] mb-1" style={{ color: 'var(--rl-amber)' }}>
              ⚠ Research / Educational Prototype
            </div>
            본 시스템은 SKKU AWS SAY 2기 2팀의 프로젝트이며 현재 SaMD 허가 전 연구·교육 목적의 프로토타입입니다. 모든 AI 출력은 주치의의 검토를 거쳐야 하며 치료 결정의 단독 근거로 사용될 수 없습니다.
          </div>
        </div>
      </div>
    </div>
  );
}

function Field({ label, icon, children }) {
  return (
    <div className="mb-3">
      <label className="flex items-center gap-1.5 text-xs font-medium mb-1.5" style={{ color: 'var(--rl-ink-2)' }}>
        <span style={{ color: 'var(--rl-ink-3)' }}>{icon}</span>
        {label}
      </label>
      {children}
    </div>
  );
}

function HPOGraph() {
  // Disease-symptom abstract network. Each dot = HPO term or disease node.
  const nodes = [
    { x: 60,  y: 30,  r: 5, type: 'disease' },
    { x: 140, y: 50,  r: 3, type: 'hpo' },
    { x: 100, y: 90,  r: 3, type: 'hpo' },
    { x: 200, y: 30,  r: 4, type: 'disease' },
    { x: 260, y: 80,  r: 3, type: 'hpo' },
    { x: 320, y: 40,  r: 5, type: 'disease' },
    { x: 180, y: 100, r: 3, type: 'hpo' },
    { x: 380, y: 80,  r: 3, type: 'hpo' },
    { x: 240, y: 130, r: 4, type: 'disease' },
    { x: 40,  y: 110, r: 3, type: 'hpo' },
    { x: 340, y: 130, r: 3, type: 'hpo' },
  ];
  const edges = [
    [0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [4, 5], [2, 6], [6, 8], [4, 7], [5, 7], [6, 2], [9, 0], [10, 5], [8, 10], [7, 8],
  ];
  return (
    <svg viewBox="0 0 440 170" className="w-full" style={{ maxHeight: 170 }}>
      {edges.map(([a, b], i) => (
        <line
          key={i}
          x1={nodes[a].x} y1={nodes[a].y}
          x2={nodes[b].x} y2={nodes[b].y}
          stroke="rgba(255,255,255,0.25)"
          strokeWidth="0.8"
          style={{ animation: `graph-pulse 3.5s ease-in-out ${i * 0.15}s infinite` }}
        />
      ))}
      {nodes.map((n, i) => (
        <g key={i}>
          <circle
            cx={n.x} cy={n.y} r={n.r + 3}
            fill="none"
            stroke={n.type === 'disease' ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.15)'}
            strokeWidth="0.8"
            style={{ animation: `graph-pulse 3s ease-in-out ${i * 0.2}s infinite` }}
          />
          <circle
            cx={n.x} cy={n.y} r={n.r}
            fill={n.type === 'disease' ? '#4DD4F5' : 'rgba(255,255,255,0.7)'}
          />
        </g>
      ))}
    </svg>
  );
}

/* ============================================================
   SCREEN 02 · HOME (Hub: 당일 외래 / 환자 검색 / 미확인 결과)
   ============================================================ */
function WorklistScreen({ doctor, onLogout, onNavigate, onOpenPatient, onOpenAnnouncement, pendingPatientMrn, onClearPendingPatient }) {
  const [section, setSection] = useState('today'); // 'today' | 'search' | 'unread'
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [contextList, setContextList] = useState([]);
  const [contextLabel, setContextLabel] = useState('');
  const [patients, setPatients] = useState(MOCK_PATIENTS);
  const [history] = useState(MOCK_PATIENT_HISTORY);

  const acknowledge = (mrn) =>
    setPatients(ps => ps.map(p => (p.mrn === mrn ? { ...p, acknowledged: true } : p)));

  const unreadCount = patients.filter(p => p.status === 'ready' && !p.acknowledged).length;

  const openPatient = (patient, list, label) => {
    setContextList(list);
    setContextLabel(label);
    setSelectedPatient(patient);
  };

  // 알림에서 진입한 환자 mrn → 환자 풀에서 찾아서 차트 열기
  const handleOpenPatient = (mrn) => {
    if (!mrn) return;
    const all = [...patients, ...history];
    const p = all.find(x => x.mrn === mrn);
    if (!p) {
      alert(`환자 정보를 찾을 수 없습니다 · MRN ${mrn}`);
      return;
    }
    const isToday = patients.some(x => x.mrn === mrn);
    const list  = isToday ? patients : [p];
    const label = isToday ? `당일 외래 · ${patients.length}명` : `알림에서 진입 · ${p.name}`;
    openPatient(p, list, label);
  };

  // 다른 screen에서 navigate 후 진입한 경우 (Settings·Dashboard 등에서 알림 클릭)
  useEffect(() => {
    if (!pendingPatientMrn) return;
    handleOpenPatient(pendingPatientMrn);
    onClearPendingPatient && onClearPendingPatient();
  }, [pendingPatientMrn]);

  // 환자 선택 시: EMR 차트 레이아웃 (좌 사이드바 + 메인 차트)
  if (selectedPatient) {
    return (
      <div className="min-h-screen flex flex-col" style={{ background: 'var(--rl-bg-2)' }}>
        <TopBar doctor={doctor} onLogout={onLogout} activeScreen="worklist" onNavigate={onNavigate} onOpenPatient={handleOpenPatient} onOpenAnnouncement={onOpenAnnouncement} />
        <ChartLayout
          patient={selectedPatient}
          list={contextList}
          contextLabel={contextLabel}
          onSelect={setSelectedPatient}
          onHome={() => setSelectedPatient(null)}
          onAcknowledge={acknowledge}
        />
      </div>
    );
  }

  // 미선택 시: 허브 (3 섹션)
  return (
    <div className="min-h-screen flex flex-col" style={{ background: 'var(--rl-bg-2)' }}>
      <TopBar doctor={doctor} onLogout={onLogout} activeScreen="worklist" onNavigate={onNavigate} onOpenPatient={handleOpenPatient} onOpenAnnouncement={onOpenAnnouncement} />

      <main className="flex-1 max-w-[1440px] w-full mx-auto px-8 py-6">
        <HomeHeader doctor={doctor} unreadCount={unreadCount} />

        <SectionNav
          active={section}
          onChange={setSection}
          counts={{
            today: patients.length,
            unread: unreadCount,
          }}
        />

        {section === 'today' && (
          <TodaySection
            patients={patients}
            onSelect={(p) => openPatient(p, patients, `당일 외래 · ${patients.length}명`)}
          />
        )}
        {section === 'search' && (
          <SearchSection
            allPatients={[...patients, ...history]}
            onSelect={(p, list) => openPatient(p, list, `검색 결과 · ${list.length}명`)}
          />
        )}
        {section === 'unread' && (
          <UnreadSection
            patients={patients}
            onSelect={(p, list) => openPatient(p, list, `미확인 결과 · ${list.length}건`)}
            onAcknowledge={acknowledge}
          />
        )}

        {/* HITL footer reminder · 모든 섹션 공통 */}
        <div className="mt-6 rounded px-4 py-3 text-xs flex items-start gap-2" style={{ background: 'var(--rl-amber-soft)', border: '1px solid var(--rl-amber)' }}>
          <AlertTriangle size={14} style={{ color: 'var(--rl-amber)', marginTop: 2, flexShrink: 0 }} />
          <div style={{ color: 'var(--rl-ink-2)' }}>
            <span className="font-medium" style={{ color: 'var(--rl-amber)' }}>본 시스템의 모든 AI 분석 결과는 진단 보조용입니다.</span>{' '}
            환자에 대한 최종 진단 및 치료 결정은 반드시 주치의의 임상적 판단에 따라야 합니다.
            <span className="font-mono ml-2" style={{ color: 'var(--rl-ink-3)' }}>[EU AI Act Art. 22]</span>
          </div>
        </div>
      </main>
    </div>
  );
}

/* ----------- HOME HEADER ----------- */
function HomeHeader({ doctor, unreadCount }) {
  return (
    <div className="flex items-baseline gap-4 mb-4">
      <div>
        <div className="font-mono text-[10px] uppercase tracking-widest mb-1" style={{ color: 'var(--rl-ink-3)' }}>
          Home · 2026.04.23 (Thu) · KST
        </div>
        <h1 className="font-serif text-3xl" style={{ color: 'var(--rl-ink)', letterSpacing: '-0.01em' }}>
          {doctor.name} 과장님, 안녕하세요.
        </h1>
      </div>
      <div className="ml-auto flex items-center gap-4 text-xs" style={{ color: 'var(--rl-ink-3)' }}>
        {unreadCount > 0 && (
          <span className="flex items-center gap-1.5 px-2 py-1 rounded" style={{ background: 'var(--rl-amber-soft)', color: 'var(--rl-amber)' }}>
            <Inbox size={12} />
            <span className="font-medium">미확인 {unreadCount}건</span>
          </span>
        )}
        <span className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full pulse-dot" style={{ background: 'var(--rl-teal)' }} />
          <span className="font-mono">FHIR sync · 08:32</span>
        </span>
      </div>
    </div>
  );
}

/* ----------- SECTION NAV (sub-tabs) ----------- */
function SectionNav({ active, onChange, counts }) {
  const tabs = [
    { k: 'today',  label: '당일 외래',       icon: <CalendarDays size={14} />, count: counts.today,   accent: 'primary' },
    { k: 'search', label: '환자 검색',       icon: <Search size={14} />,       count: null,           accent: 'ink' },
    { k: 'unread', label: '미확인 환자결과', icon: <Inbox size={14} />,        count: counts.unread,  accent: 'amber', alert: counts.unread > 0 },
  ];
  return (
    <div className="hairline rounded bg-white p-1 mb-4 flex items-center gap-1">
      {tabs.map(t => {
        const isActive = active === t.k;
        return (
          <button
            key={t.k}
            onClick={() => onChange(t.k)}
            className="flex-1 px-4 py-2.5 rounded text-sm font-medium transition flex items-center justify-center gap-2"
            style={{
              background: isActive ? 'var(--rl-primary)' : 'transparent',
              color: isActive ? 'white' : 'var(--rl-ink-2)',
            }}
          >
            <span style={{ color: isActive ? 'white' : `var(--rl-${t.accent === 'ink' ? 'ink-3' : t.accent})` }}>
              {t.icon}
            </span>
            {t.label}
            {t.count !== null && (
              <span
                className="font-mono text-[11px] px-1.5 py-0.5 rounded"
                style={{
                  background: isActive ? 'rgba(255,255,255,0.18)' : 'var(--rl-bg-3)',
                  color: isActive ? 'white' : 'var(--rl-ink-2)',
                }}
              >
                {t.count}
              </span>
            )}
            {t.alert && !isActive && (
              <span className="w-1.5 h-1.5 rounded-full pulse-dot" style={{ background: 'var(--rl-amber)' }} />
            )}
          </button>
        );
      })}
    </div>
  );
}

/* ============================================================
   SECTION A · 당일 외래
   ============================================================ */
function TodaySection({ patients, onSelect }) {
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');

  const filtered = patients.filter(p => {
    if (search && !p.name.includes(search) && !p.mrn.includes(search)) return false;
    if (filter === 'all') return true;
    if (filter === 'rare') return p.rare || p.dontMiss;
    return p.status === filter;
  });

  const stats = {
    total: patients.length,
    analyzed: patients.filter(p => p.status === 'ready').length,
    analyzing: patients.filter(p => p.status === 'analyzing').length,
    rare: patients.filter(p => p.rare).length,
  };

  return (
    <div className="fade-in">
      {/* Stats · 카드 클릭 시 필터 적용 (같은 카드 재클릭 시 전체로) */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        <StatCard
          label="오늘 예약 환자"
          value={stats.total}     unit="명"
          icon={<User size={14} />}            accent="primary"
          active={filter === 'all'}
          onClick={() => setFilter('all')}
        />
        <StatCard
          label="AI 분석 완료"
          value={stats.analyzed}  unit="/ " totalUnit={stats.total + '명'}
          icon={<CheckCircle2 size={14} />}    accent="teal"
          active={filter === 'ready'}
          onClick={() => setFilter(filter === 'ready' ? 'all' : 'ready')}
        />
        <StatCard
          label="분석 중"
          value={stats.analyzing} unit="명"
          icon={<Loader2 size={14} className="spin" />} accent="ink"
          active={filter === 'analyzing'}
          onClick={() => setFilter(filter === 'analyzing' ? 'all' : 'analyzing')}
        />
        <StatCard
          label="희귀질환 의심"
          value={stats.rare}      unit="건"
          icon={<Flame size={14} />}           accent="rare"
          active={filter === 'rare'}
          onClick={() => setFilter(filter === 'rare' ? 'all' : 'rare')}
        />
      </div>

      {/* Filter Bar */}
      <div className="hairline rounded bg-white p-2 mb-3 flex items-center gap-1">
        <ListFilter size={14} style={{ color: 'var(--rl-ink-3)', margin: '0 8px' }} />
        {[
          { k: 'all',       label: '전체',           n: patients.length },
          { k: 'pending',   label: '대기',           n: patients.filter(p => p.status === 'pending').length },
          { k: 'analyzing', label: '분석 중',        n: patients.filter(p => p.status === 'analyzing').length },
          { k: 'ready',     label: '결과 대기 확인', n: patients.filter(p => p.status === 'ready').length },
          { k: 'rare',      label: '희귀 플래그',    n: patients.filter(p => p.rare || p.dontMiss).length, flag: true },
        ].map(f => (
          <button
            key={f.k}
            onClick={() => setFilter(f.k)}
            className="px-3 py-1.5 rounded text-xs font-medium transition flex items-center gap-1.5"
            style={{
              background: filter === f.k ? 'var(--rl-primary)' : 'transparent',
              color: filter === f.k ? 'white' : 'var(--rl-ink-2)',
            }}
          >
            {f.flag && <Flame size={11} style={{ color: filter === f.k ? 'white' : 'var(--rl-rare)' }} />}
            {f.label}
            <span className="font-mono" style={{ opacity: 0.7 }}>{f.n}</span>
          </button>
        ))}

        <div className="ml-auto flex items-center gap-2 pr-2">
          <div className="relative">
            <Search size={13} className="absolute left-2 top-1/2 -translate-y-1/2" style={{ color: 'var(--rl-ink-3)' }} />
            <input
              placeholder="이름 또는 MRN"
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="pl-7 pr-3 py-1.5 rounded text-xs hairline-strong outline-none w-48 focus:border-[color:var(--rl-primary)]"
            />
          </div>
        </div>
      </div>

      {/* Worklist Table */}
      <div className="hairline rounded bg-white overflow-hidden">
        <div className="grid px-4 py-2.5 font-mono text-[10px] uppercase tracking-widest" style={{
          gridTemplateColumns: '60px 1.3fr 2fr 140px 1fr 90px',
          color: 'var(--rl-ink-3)',
          background: 'var(--rl-bg-3)',
          borderBottom: '1px solid var(--rl-border-soft)',
        }}>
          <div>예약</div>
          <div>환자 정보</div>
          <div>주호소 · 소견</div>
          <div>CXR · AI · Lab</div>
          <div>플래그</div>
          <div style={{ textAlign: 'right' }}>액션</div>
        </div>

        {filtered.map((p, i) => (
          <PatientRow
            key={p.mrn}
            p={p}
            onClick={() => onSelect(p)}
            isLast={i === filtered.length - 1}
          />
        ))}

        {filtered.length === 0 && (
          <div className="p-10 text-center text-sm" style={{ color: 'var(--rl-ink-3)' }}>
            조건에 맞는 환자가 없습니다.
          </div>
        )}
      </div>
    </div>
  );
}

/* ============================================================
   SECTION B · 환자 검색
   ============================================================ */
function SearchSection({ allPatients, onSelect }) {
  const [query, setQuery] = useState('');
  const [range, setRange] = useState('1m'); // 'today' | '1w' | '1m' | '3m' | 'all'
  const [flagFilter, setFlagFilter] = useState('all'); // 'all' | 'rare' | 'dontMiss' | 'allergy'

  const today = new Date('2026-04-23');
  const cutoffDays = { today: 0, '1w': 7, '1m': 30, '3m': 90, all: 99999 }[range];

  const filtered = allPatients.filter(p => {
    if (range !== 'all') {
      const d = p.visitDate ? new Date(p.visitDate) : today;
      const diff = (today - d) / (1000 * 60 * 60 * 24);
      if (diff > cutoffDays) return false;
    }
    if (flagFilter === 'rare'     && !p.rare)     return false;
    if (flagFilter === 'dontMiss' && !p.dontMiss) return false;
    if (flagFilter === 'allergy'  && !p.allergy)  return false;
    if (query) {
      const q = query.toLowerCase();
      const inName = p.name.includes(query);
      const inMrn  = p.mrn.toLowerCase().includes(q);
      const inDx   = (p.preview || []).some(d => d.name.toLowerCase().includes(q));
      const inComplaint = (p.complaint || '').toLowerCase().includes(q);
      if (!inName && !inMrn && !inDx && !inComplaint) return false;
    }
    return true;
  });

  const ranges = [
    { k: 'today', label: '오늘' },
    { k: '1w',    label: '최근 1주' },
    { k: '1m',    label: '최근 1개월' },
    { k: '3m',    label: '최근 3개월' },
    { k: 'all',   label: '전체' },
  ];
  const flags = [
    { k: 'all',      label: '모든 플래그',      icon: null },
    { k: 'rare',     label: '희귀',             icon: <Flame size={11} />,          color: 'var(--rl-rare)' },
    { k: 'dontMiss', label: "Don't miss",       icon: <AlertTriangle size={11} />, color: 'var(--rl-amber)' },
    { k: 'allergy',  label: '알러지',           icon: <Ban size={11} />,           color: 'var(--rl-critical)' },
  ];

  return (
    <div className="fade-in">
      {/* Search input · large */}
      <div className="hairline rounded bg-white p-4 mb-3">
        <div className="flex items-center gap-3 mb-3">
          <Search size={18} style={{ color: 'var(--rl-primary)' }} />
          <input
            autoFocus
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="환자명 · MRN · 주호소 · 진단명 으로 검색"
            className="flex-1 outline-none text-base"
            style={{ color: 'var(--rl-ink)' }}
          />
          {query && (
            <button onClick={() => setQuery('')} className="p-1 rounded hover:bg-slate-100" style={{ color: 'var(--rl-ink-3)' }}>
              <X size={14} />
            </button>
          )}
          <span className="font-mono text-[11px]" style={{ color: 'var(--rl-ink-3)' }}>
            {filtered.length} / {allPatients.length}
          </span>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-1">
            <CalendarDays size={12} style={{ color: 'var(--rl-ink-3)', marginRight: 4 }} />
            {ranges.map(r => (
              <button
                key={r.k}
                onClick={() => setRange(r.k)}
                className="px-2.5 py-1 rounded text-[11px] font-medium transition"
                style={{
                  background: range === r.k ? 'var(--rl-primary-soft)' : 'transparent',
                  color: range === r.k ? 'var(--rl-primary)' : 'var(--rl-ink-3)',
                }}
              >
                {r.label}
              </button>
            ))}
          </div>

          <div className="h-4 w-px" style={{ background: 'var(--rl-border-soft)' }} />

          <div className="flex items-center gap-1">
            <Filter size={12} style={{ color: 'var(--rl-ink-3)', marginRight: 4 }} />
            {flags.map(f => (
              <button
                key={f.k}
                onClick={() => setFlagFilter(f.k)}
                className="px-2.5 py-1 rounded text-[11px] font-medium transition flex items-center gap-1"
                style={{
                  background: flagFilter === f.k ? 'var(--rl-primary-soft)' : 'transparent',
                  color: flagFilter === f.k ? 'var(--rl-primary)' : (f.color || 'var(--rl-ink-3)'),
                }}
              >
                {f.icon}
                {f.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Result Table */}
      <div className="hairline rounded bg-white overflow-hidden">
        <div className="grid px-4 py-2.5 font-mono text-[10px] uppercase tracking-widest" style={{
          gridTemplateColumns: '110px 1.3fr 2fr 140px 1fr 90px',
          color: 'var(--rl-ink-3)',
          background: 'var(--rl-bg-3)',
          borderBottom: '1px solid var(--rl-border-soft)',
        }}>
          <div>방문일</div>
          <div>환자 정보</div>
          <div>주호소 · 소견</div>
          <div>CXR · AI · Lab</div>
          <div>플래그</div>
          <div style={{ textAlign: 'right' }}>액션</div>
        </div>

        {filtered.map((p, i) => (
          <SearchResultRow
            key={p.mrn + (p.visitDate || '')}
            p={p}
            onClick={() => onSelect(p, filtered)}
            isLast={i === filtered.length - 1}
          />
        ))}

        {filtered.length === 0 && (
          <div className="p-10 text-center text-sm" style={{ color: 'var(--rl-ink-3)' }}>
            검색 조건에 맞는 환자가 없습니다.
          </div>
        )}
      </div>
    </div>
  );
}

function SearchResultRow({ p, onClick, isLast }) {
  const dateLabel = p.visitDate
    ? p.visitDate.replace(/^2026-/, '').replace('-', '/')
    : '오늘';
  return (
    <div
      onClick={onClick}
      className="grid px-4 py-3 row-hover transition"
      style={{
        gridTemplateColumns: '110px 1.3fr 2fr 140px 1fr 90px',
        borderBottom: isLast ? 'none' : '1px solid var(--rl-border-soft)',
        alignItems: 'center',
      }}
    >
      <div>
        <div className="font-mono text-sm font-medium" style={{ color: 'var(--rl-ink)' }}>{dateLabel}</div>
        <div className="font-mono text-[10px] uppercase" style={{ color: 'var(--rl-ink-3)' }}>{p.visit} · {p.time}</div>
      </div>

      <div className="flex items-center gap-2.5 min-w-0">
        <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: 'var(--rl-primary-soft)', color: 'var(--rl-primary)' }}>
          <User size={12} />
        </div>
        <div className="min-w-0">
          <div className="text-sm font-medium truncate" style={{ color: 'var(--rl-ink)' }}>
            {p.name} <span className="font-normal" style={{ color: 'var(--rl-ink-2)' }}>· {p.sex}/{p.age}</span>
          </div>
          <div className="font-mono text-[11px]" style={{ color: 'var(--rl-ink-3)' }}>{p.mrn}</div>
        </div>
      </div>

      <div className="min-w-0 pr-3">
        <div className="text-sm truncate" style={{ color: 'var(--rl-ink)' }}>{p.complaint}</div>
        {p.preview && p.preview[0] && (
          <div className="text-[11px] truncate" style={{ color: 'var(--rl-ink-3)' }}>
            Top dx: {p.preview[0].name} · {(p.preview[0].prob * 100).toFixed(0)}%
          </div>
        )}
      </div>

      <StatusTriCell cxr={p.cxr} ai={p.status} lab={getLabStatus(p)} />

      <div className="flex items-center gap-1.5 flex-wrap">
        {p.rare && (
          <span className="chip" style={{ background: 'var(--rl-rare-soft)', color: 'var(--rl-rare)' }}>
            <Flame size={10} /> 희귀
          </span>
        )}
        {p.dontMiss && (
          <span className="chip" style={{ background: 'var(--rl-amber-soft)', color: 'var(--rl-amber)' }}>
            <AlertTriangle size={10} /> Don't miss
          </span>
        )}
        {p.allergy && (
          <span className="chip" style={{ background: 'var(--rl-critical-soft)', color: 'var(--rl-critical)' }}>
            <Ban size={10} /> 알러지
          </span>
        )}
      </div>

      <div style={{ textAlign: 'right' }}>
        <button className="inline-flex items-center gap-1 text-xs" style={{ color: 'var(--rl-primary)', fontWeight: 500 }}>
          열기 <ChevronRight size={12} />
        </button>
      </div>
    </div>
  );
}

/* ============================================================
   SECTION C · 미확인 환자결과
   ============================================================ */
function UnreadSection({ patients, onSelect, onAcknowledge }) {
  const unread = patients
    .filter(p => p.status === 'ready' && !p.acknowledged)
    .sort((a, b) => {
      const sev = (x) => (x.dontMiss ? 2 : 0) + (x.rare ? 1 : 0);
      const s = sev(b) - sev(a);
      if (s !== 0) return s;
      return (a.resultAt || '').localeCompare(b.resultAt || '');
    });

  const counts = {
    total: unread.length,
    dontMiss: unread.filter(p => p.dontMiss).length,
    rare: unread.filter(p => p.rare).length,
  };

  if (unread.length === 0) {
    return (
      <div className="fade-in hairline rounded bg-white p-12 text-center">
        <CheckCircle2 size={28} style={{ color: 'var(--rl-teal)', margin: '0 auto 10px' }} />
        <div className="text-sm font-medium" style={{ color: 'var(--rl-ink)' }}>모든 결과를 확인하셨습니다.</div>
        <div className="text-xs mt-1" style={{ color: 'var(--rl-ink-3)' }}>새로운 AI 분석 결과가 도착하면 이곳에 표시됩니다.</div>
      </div>
    );
  }

  return (
    <div className="fade-in">
      {/* Summary strip */}
      <div className="hairline rounded bg-white px-4 py-3 mb-3 flex items-center gap-5">
        <div className="flex items-center gap-2">
          <Inbox size={16} style={{ color: 'var(--rl-amber)' }} />
          <span className="text-sm font-medium" style={{ color: 'var(--rl-ink)' }}>
            확인이 필요한 결과 <span className="font-serif text-lg" style={{ color: 'var(--rl-amber)' }}>{counts.total}</span> 건
          </span>
        </div>
        <div className="h-4 w-px" style={{ background: 'var(--rl-border-soft)' }} />
        {counts.dontMiss > 0 && (
          <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--rl-amber)' }}>
            <AlertTriangle size={12} />
            Don't miss <span className="font-mono font-medium">{counts.dontMiss}</span>
          </span>
        )}
        {counts.rare > 0 && (
          <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--rl-rare)' }}>
            <Flame size={12} />
            희귀 의심 <span className="font-mono font-medium">{counts.rare}</span>
          </span>
        )}
        <span className="ml-auto font-mono text-[10px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)' }}>
          정렬 · Don't miss → 희귀 → 도착 순
        </span>
      </div>

      <div className="space-y-2">
        {unread.map(p => (
          <UnreadCard
            key={p.mrn}
            p={p}
            onSelect={(pp) => onSelect(pp, unread)}
            onAcknowledge={onAcknowledge}
          />
        ))}
      </div>
    </div>
  );
}

function UnreadCard({ p, onSelect, onAcknowledge }) {
  const top = p.preview && p.preview[0];
  const accent = p.dontMiss
    ? { c: 'var(--rl-amber)', bg: 'var(--rl-amber-soft)' }
    : p.rare
      ? { c: 'var(--rl-rare)', bg: 'var(--rl-rare-soft)' }
      : { c: 'var(--rl-primary)', bg: 'var(--rl-primary-soft)' };

  return (
    <div
      className="hairline rounded bg-white overflow-hidden flex"
      style={{ borderLeft: `3px solid ${accent.c}` }}
    >
      <div onClick={() => onSelect(p)} className="flex-1 flex items-center gap-4 px-4 py-3 cursor-pointer row-hover">
        {/* Patient identity */}
        <div className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: accent.bg, color: accent.c }}>
          <User size={14} />
        </div>
        <div className="min-w-0 w-44">
          <div className="text-sm font-medium truncate" style={{ color: 'var(--rl-ink)' }}>
            {p.name} <span className="font-normal" style={{ color: 'var(--rl-ink-2)' }}>· {p.sex}/{p.age}</span>
          </div>
          <div className="font-mono text-[11px]" style={{ color: 'var(--rl-ink-3)' }}>{p.mrn} · {p.time}</div>
        </div>

        {/* Top dx */}
        {top && (
          <div className="flex-1 min-w-0">
            <div className="font-mono text-[10px] uppercase tracking-widest mb-0.5" style={{ color: 'var(--rl-ink-3)' }}>
              Top differential
            </div>
            <div className="flex items-baseline gap-2">
              <div className="text-sm font-medium truncate" style={{ color: 'var(--rl-ink)' }}>{top.name}</div>
              <div className="font-serif text-base leading-none" style={{ color: accent.c }}>
                {(top.prob * 100).toFixed(0)}<span className="text-[10px]">%</span>
              </div>
              {top.orpha && (
                <span className="font-mono text-[10px]" style={{ color: 'var(--rl-ink-3)' }}>{top.orpha}</span>
              )}
            </div>
          </div>
        )}

        {/* Flags */}
        <div className="flex items-center gap-1.5 flex-wrap" style={{ minWidth: 140 }}>
          {p.dontMiss && (
            <span className="chip" style={{ background: 'var(--rl-amber-soft)', color: 'var(--rl-amber)' }}>
              <AlertTriangle size={10} /> Don't miss
            </span>
          )}
          {p.rare && (
            <span className="chip" style={{ background: 'var(--rl-rare-soft)', color: 'var(--rl-rare)' }}>
              <Flame size={10} /> 희귀
            </span>
          )}
        </div>

        {/* Arrived ago */}
        <div className="text-right" style={{ minWidth: 90 }}>
          <div className="font-mono text-[10px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)' }}>도착</div>
          <div className="font-mono text-xs" style={{ color: 'var(--rl-ink-2)' }}>{p.resultAt || p.time}</div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-stretch border-l" style={{ borderColor: 'var(--rl-border-soft)' }}>
        <button
          onClick={() => onSelect(p)}
          className="px-4 text-xs font-medium flex items-center gap-1.5 hover:bg-slate-50 transition"
          style={{ color: 'var(--rl-primary)' }}
        >
          <Eye size={13} /> 결과 열기
        </button>
        <button
          onClick={() => onAcknowledge(p.mrn)}
          className="px-4 text-xs font-medium flex items-center gap-1.5 hover:bg-slate-50 transition border-l"
          style={{ color: 'var(--rl-teal)', borderColor: 'var(--rl-border-soft)' }}
          title="확인 처리"
        >
          <CheckCircle2 size={13} /> 확인
        </button>
      </div>
    </div>
  );
}

function TopBar({ doctor, onLogout, activeScreen = 'worklist', onNavigate, onOpenPatient, onOpenAnnouncement }) {
  const navs = [
    { k: 'worklist',  label: '환자 목록' },
    { k: 'dashboard', label: '분석 대시보드' },
    { k: 'knowledge', label: '지식 베이스' },
    { k: 'settings',  label: '설정' },
  ];
  return (
    <div className="hairline bg-white sticky top-0 z-40" style={{ borderTop: 'none', borderLeft: 'none', borderRight: 'none' }}>
      <div className="max-w-[1440px] mx-auto px-8 py-3 flex items-center gap-4">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded flex items-center justify-center" style={{ background: 'var(--rl-primary)' }}>
            <Stethoscope size={15} color="white" strokeWidth={2.5} />
          </div>
          <div>
            <div className="font-serif text-base leading-none" style={{ color: 'var(--rl-ink)', letterSpacing: '-0.01em' }}>
              Rare-Link <span style={{ fontStyle: 'italic', fontWeight: 500 }}>AI</span>
            </div>
            <div className="font-mono text-[10px] mt-0.5 uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)' }}>
              Clinical CDSS · v0.1.0
            </div>
          </div>
        </div>

        <div className="mx-4 h-6 w-px" style={{ background: 'var(--rl-border)' }} />

        <nav className="flex items-center gap-5 text-sm">
          {navs.map(n => {
            const active = activeScreen === n.k;
            return (
              <button
                key={n.k}
                onClick={() => onNavigate && onNavigate(n.k)}
                style={{
                  color: active ? 'var(--rl-primary)' : 'var(--rl-ink-3)',
                  fontWeight: active ? 600 : 400,
                  borderBottom: active ? '2px solid var(--rl-primary)' : '2px solid transparent',
                  paddingBottom: '12px',
                  marginBottom: '-12px',
                  background: 'none',
                  cursor: 'pointer',
                  fontSize: 'inherit',
                  fontFamily: 'inherit',
                }}
              >
                {n.label}
              </button>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-4">
          <NotificationButton onOpenPatient={onOpenPatient} onOpenAnnouncement={onOpenAnnouncement} />

          <div className="flex items-center gap-2.5 pl-3" style={{ borderLeft: '1px solid var(--rl-border)' }}>
            <div className="w-8 h-8 rounded-full flex items-center justify-center" style={{ background: 'var(--rl-primary-soft)', color: 'var(--rl-primary)' }}>
              <User size={14} />
            </div>
            <div>
              <div className="text-sm leading-none font-medium" style={{ color: 'var(--rl-ink)' }}>{doctor.name} 과장</div>
              <div className="text-[10px] mt-0.5" style={{ color: 'var(--rl-ink-3)' }}>
                {doctor.institution} · {doctor.department}
              </div>
            </div>
          </div>

          <SessionCountdown />

          <button
            onClick={onLogout}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs hairline-strong transition hover:bg-slate-50"
            style={{ color: 'var(--rl-ink-2)' }}
          >
            <LogOut size={12} /> 로그아웃
          </button>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, unit, totalUnit, icon, accent, onClick, active }) {
  const colors = {
    primary: { c: 'var(--rl-primary)', bg: 'var(--rl-primary-soft)' },
    teal:    { c: 'var(--rl-teal)',    bg: 'var(--rl-teal-soft)' },
    rare:    { c: 'var(--rl-rare)',    bg: 'var(--rl-rare-soft)' },
    ink:     { c: 'var(--rl-ink-2)',   bg: 'var(--rl-bg-3)' },
  }[accent];

  const clickable = !!onClick;
  return (
    <div
      onClick={onClick}
      className={`rounded bg-white p-4 flex items-center gap-4 transition ${clickable ? 'cursor-pointer hover:bg-slate-50' : ''}`}
      style={{
        border: `1px solid ${active ? colors.c : 'var(--rl-border-soft)'}`,
        borderWidth: active ? '1px' : '1px',
        boxShadow: active ? `inset 0 0 0 1px ${colors.c}` : 'none',
      }}
      title={clickable ? `${label} 필터 적용` : undefined}
    >
      <div className="w-9 h-9 rounded flex items-center justify-center" style={{ background: colors.bg, color: colors.c }}>
        {icon}
      </div>
      <div>
        <div className="text-[11px] mb-0.5 flex items-center gap-1" style={{ color: 'var(--rl-ink-3)' }}>
          {label}
          {active && <span className="font-mono text-[9px]" style={{ color: colors.c }}>· 활성</span>}
        </div>
        <div className="flex items-baseline gap-1">
          <span className="font-serif text-2xl leading-none" style={{ color: active ? colors.c : 'var(--rl-ink)' }}>{value}</span>
          <span className="text-xs" style={{ color: 'var(--rl-ink-3)' }}>{unit}{totalUnit || ''}</span>
        </div>
      </div>
    </div>
  );
}

function PatientRow({ p, onClick, isLast }) {
  return (
    <div
      onClick={onClick}
      className="grid px-4 py-3 row-hover transition"
      style={{
        gridTemplateColumns: '60px 1.3fr 2fr 140px 1fr 90px',
        borderBottom: isLast ? 'none' : '1px solid var(--rl-border-soft)',
        alignItems: 'center',
      }}
    >
      {/* Time */}
      <div>
        <div className="font-mono text-sm font-medium" style={{ color: 'var(--rl-ink)' }}>{p.time}</div>
        <div className="font-mono text-[10px] uppercase" style={{ color: 'var(--rl-ink-3)' }}>{p.visit}</div>
      </div>

      {/* Patient */}
      <div className="flex items-center gap-2.5 min-w-0">
        <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: 'var(--rl-primary-soft)', color: 'var(--rl-primary)' }}>
          <User size={12} />
        </div>
        <div className="min-w-0">
          <div className="text-sm font-medium truncate" style={{ color: 'var(--rl-ink)' }}>
            {p.name} <span className="font-normal" style={{ color: 'var(--rl-ink-2)' }}>· {p.sex}/{p.age}</span>
          </div>
          <div className="font-mono text-[11px]" style={{ color: 'var(--rl-ink-3)' }}>{p.mrn}</div>
        </div>
      </div>

      {/* Chief complaint */}
      <div className="min-w-0 pr-3">
        <div className="text-sm truncate" style={{ color: 'var(--rl-ink)' }}>{p.complaint}</div>
        {p.allergy && (
          <div className="flex items-center gap-1 mt-0.5">
            <Ban size={10} style={{ color: 'var(--rl-critical)' }} />
            <span className="text-[11px]" style={{ color: 'var(--rl-critical)' }}>알러지 · {p.allergy}</span>
          </div>
        )}
      </div>

      {/* CXR + AI + Lab status */}
      <StatusTriCell cxr={p.cxr} ai={p.status} lab={getLabStatus(p)} />

      {/* Flags */}
      <div className="flex items-center gap-1.5 flex-wrap">
        {p.rare && (
          <span className="chip" style={{ background: 'var(--rl-rare-soft)', color: 'var(--rl-rare)' }}>
            <Flame size={10} /> 희귀
          </span>
        )}
        {p.dontMiss && (
          <span className="chip" style={{ background: 'var(--rl-amber-soft)', color: 'var(--rl-amber)' }}>
            <AlertTriangle size={10} /> Don't miss
          </span>
        )}
        {p.topDx && !p.rare && !p.dontMiss && (
          <span className="chip" style={{ background: 'var(--rl-teal-soft)', color: 'var(--rl-teal)' }}>
            Top: {p.topDx}
          </span>
        )}
      </div>

      {/* Action */}
      <div style={{ textAlign: 'right' }}>
        <button className="inline-flex items-center gap-1 text-xs" style={{ color: 'var(--rl-primary)', fontWeight: 500 }}>
          열기 <ChevronRight size={12} />
        </button>
      </div>
    </div>
  );
}

function StatusTriCell({ cxr, ai, lab }) {
  const cxrMap = {
    arrived:  { icon: <CheckCircle2 size={12} />, label: '촬영 완료', c: 'var(--rl-teal)' },
    pending:  { icon: <Circle size={12} />,       label: '촬영 대기', c: 'var(--rl-ink-3)' },
  };
  const aiMap = {
    pending:   { icon: <Circle size={12} />,                  label: '대기',     c: 'var(--rl-ink-3)' },
    analyzing: { icon: <Loader2 size={12} className="spin" />, label: '분석 중',  c: 'var(--rl-primary)' },
    ready:     { icon: <CheckCircle2 size={12} />,            label: '결과 대기', c: 'var(--rl-teal)' },
  };
  const labMap = {
    none:    { icon: <Circle size={12} />,                  label: '결과 없음', c: 'var(--rl-ink-3)' },
    pending: { icon: <Loader2 size={12} className="spin" />, label: '결과 대기', c: 'var(--rl-amber)' },
    ready:   { icon: <CheckCircle2 size={12} />,            label: '결과 도착', c: 'var(--rl-teal)' },
  };
  const c = cxrMap[cxr];
  const a = aiMap[ai];
  const l = labMap[lab];
  return (
    <div className="flex flex-col gap-0.5 text-[11px]">
      <div className="flex items-center gap-1.5" style={{ color: c.c }}>
        <ScanLine size={11} />
        <span className="font-mono">CXR</span>
        {c.icon}
        <span>{c.label}</span>
      </div>
      <div className="flex items-center gap-1.5" style={{ color: a.c }}>
        <Microscope size={11} />
        <span className="font-mono">AI</span>
        {a.icon}
        <span>{a.label}</span>
      </div>
      <div className="flex items-center gap-1.5" style={{ color: l.c }}>
        <FlaskConical size={11} />
        <span className="font-mono">Lab</span>
        {l.icon}
        <span>{l.label}</span>
      </div>
    </div>
  );
}

/* ============================================================
   CHART LAYOUT · 환자 선택 시 EMR 풀스크린 차트
   ============================================================ */
function ChartLayout({ patient, list, contextLabel, onSelect, onHome, onAcknowledge }) {
  return (
    <div className="flex-1 flex" style={{ minHeight: 0 }}>
      <PatientSidebar
        list={list}
        contextLabel={contextLabel}
        selectedMrn={patient.mrn}
        onSelect={onSelect}
        onHome={onHome}
      />
      <PatientChart
        patient={patient}
        onHome={onHome}
        onAcknowledge={onAcknowledge}
      />
    </div>
  );
}

/* ----------- PATIENT SIDEBAR (resizable: 200 ~ 320 px, default 250) ----------- */
const SIDEBAR_MIN = 200;
const SIDEBAR_MAX = 320;
const SIDEBAR_DEFAULT = 250;

function PatientSidebar({ list, contextLabel, selectedMrn, onSelect, onHome }) {
  const [q, setQ] = useState('');
  const [width, setWidth] = useState(SIDEBAR_DEFAULT);
  const [dragging, setDragging] = useState(false);

  const filtered = list.filter(p =>
    !q || p.name.includes(q) || p.mrn.toLowerCase().includes(q.toLowerCase())
  );

  // Drag-to-resize handlers · attach on window so cursor can leave the handle
  useEffect(() => {
    if (!dragging) return;
    const onMove = (e) => {
      const w = Math.max(SIDEBAR_MIN, Math.min(SIDEBAR_MAX, e.clientX));
      setWidth(w);
    };
    const onUp = () => setDragging(false);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [dragging]);

  const compact = width < 250;

  return (
    <aside
      className="flex flex-col bg-white relative"
      style={{
        width,
        flexShrink: 0,
        borderRight: '1px solid var(--rl-border)',
        height: 'calc(100vh - 57px)',
        position: 'sticky',
        top: 57,
      }}
    >
      {/* Header */}
      <div className="px-4 py-3" style={{ borderBottom: '1px solid var(--rl-border-soft)' }}>
        <button
          onClick={onHome}
          className="flex items-center gap-1.5 text-[11px] font-medium mb-2 hover:underline"
          style={{ color: 'var(--rl-primary)' }}
        >
          <Home size={12} /> 홈으로
        </button>
        <div className="font-mono text-[10px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)' }}>
          Patient List
        </div>
        <div className="text-sm font-medium mt-0.5 truncate" style={{ color: 'var(--rl-ink)' }}>
          {contextLabel}
        </div>
      </div>

      {/* Search */}
      <div className="px-3 py-2" style={{ borderBottom: '1px solid var(--rl-border-soft)' }}>
        <div className="relative">
          <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2" style={{ color: 'var(--rl-ink-3)' }} />
          <input
            value={q}
            onChange={e => setQ(e.target.value)}
            placeholder="이름 · MRN"
            className="pl-7 pr-3 py-1.5 rounded text-xs hairline-strong outline-none w-full focus:border-[color:var(--rl-primary)]"
          />
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto">
        {filtered.map(p => (
          <SidebarPatientRow
            key={p.mrn + (p.visitDate || '')}
            p={p}
            selected={p.mrn === selectedMrn}
            compact={compact}
            onClick={() => onSelect(p)}
          />
        ))}
        {filtered.length === 0 && (
          <div className="p-6 text-center text-xs" style={{ color: 'var(--rl-ink-3)' }}>
            결과 없음
          </div>
        )}
      </div>

      {/* Width readout (drag-only feedback) */}
      {dragging && (
        <div
          className="font-mono text-[10px] px-1.5 py-0.5 rounded"
          style={{
            position: 'absolute',
            right: 8,
            bottom: 8,
            background: 'var(--rl-primary)',
            color: 'white',
            zIndex: 60,
          }}
        >
          {width}px
        </div>
      )}

      {/* Drag handle · sits on the right edge */}
      <div
        onMouseDown={(e) => { e.preventDefault(); setDragging(true); }}
        onDoubleClick={() => setWidth(SIDEBAR_DEFAULT)}
        title="드래그로 너비 조절 · 더블클릭으로 기본값(250px)"
        style={{
          position: 'absolute',
          top: 0,
          bottom: 0,
          right: -3,
          width: 6,
          cursor: 'col-resize',
          zIndex: 50,
          background: dragging ? 'var(--rl-primary)' : 'transparent',
          transition: dragging ? 'none' : 'background 0.15s',
        }}
        onMouseEnter={(e) => { if (!dragging) e.currentTarget.style.background = 'var(--rl-primary-soft)'; }}
        onMouseLeave={(e) => { if (!dragging) e.currentTarget.style.background = 'transparent'; }}
      />
    </aside>
  );
}

function SidebarPatientRow({ p, selected, compact, onClick }) {
  const stripe = p.dontMiss
    ? 'var(--rl-amber)'
    : p.rare
      ? 'var(--rl-rare)'
      : p.allergy
        ? 'var(--rl-critical)'
        : 'transparent';

  return (
    <div
      onClick={onClick}
      className="px-3 py-2.5 cursor-pointer transition"
      style={{
        background: selected ? 'var(--rl-primary-soft)' : 'transparent',
        borderLeft: `3px solid ${selected ? 'var(--rl-primary)' : stripe}`,
        borderBottom: '1px solid var(--rl-border-soft)',
      }}
      onMouseEnter={e => { if (!selected) e.currentTarget.style.background = 'var(--rl-bg-2)'; }}
      onMouseLeave={e => { if (!selected) e.currentTarget.style.background = 'transparent'; }}
    >
      {/* Row 1 · 시간 + 이름 · 성별/나이 + 미확인 dot (최소 폭에서도 보장) */}
      <div className="flex items-baseline gap-2 min-w-0">
        <div className="font-mono text-[10px] flex-shrink-0" style={{ color: 'var(--rl-ink-3)' }}>
          {p.visitDate ? p.visitDate.replace(/^2026-/, '').replace('-', '/') : p.time}
        </div>
        <div className="text-sm font-medium truncate flex-1 min-w-0" style={{ color: 'var(--rl-ink)' }}>
          {p.name} <span className="font-normal" style={{ color: 'var(--rl-ink-2)' }}>· {p.sex}/{p.age}</span>
        </div>
        {p.status === 'ready' && !p.acknowledged && (
          <span className="w-1.5 h-1.5 rounded-full pulse-dot flex-shrink-0" style={{ background: 'var(--rl-amber)' }} />
        )}
      </div>

      {/* Row 2 · MRN + flag icons */}
      <div className="flex items-center gap-2 mt-0.5 min-w-0">
        <div className="font-mono text-[10px] truncate" style={{ color: 'var(--rl-ink-3)' }}>{p.mrn}</div>
        <div className="flex items-center gap-1 flex-shrink-0">
          {p.rare && <Flame size={9} style={{ color: 'var(--rl-rare)' }} />}
          {p.dontMiss && <AlertTriangle size={9} style={{ color: 'var(--rl-amber)' }} />}
          {p.allergy && <Ban size={9} style={{ color: 'var(--rl-critical)' }} />}
        </div>
      </div>

      {/* Row 3 · complaint (compact 모드에선 숨김) */}
      {!compact && (
        <div className="text-[11px] truncate mt-0.5" style={{ color: 'var(--rl-ink-3)' }}>
          {p.complaint}
        </div>
      )}
    </div>
  );
}

/* ----------- PATIENT CHART (main area · 스크롤 없이 fit) ----------- */
function PatientChart({ patient, onHome, onAcknowledge }) {
  const [tab, setTab] = useState('overview');

  return (
    <div
      className="flex-1 flex flex-col"
      style={{ minWidth: 0, height: 'calc(100vh - 57px)', overflow: 'hidden' }}
    >
      <PatientBanner patient={patient} onHome={onHome} onAcknowledge={onAcknowledge} />
      <ChartTabs active={tab} onChange={setTab} patient={patient} />

      {/* Tab content · viewport에 맞게 fit */}
      <div className="flex-1 px-6 py-4" style={{ minHeight: 0, overflow: 'hidden' }}>
        {tab === 'overview'  && <ChartOverview patient={patient} />}
        {tab === 'cxr'       && <ChartCXR patient={patient} />}
        {tab === 'workspace' && <ChartWorkspacePlaceholder />}
        {tab === 'report'    && <ChartReport patient={patient} />}
        {tab === 'history'   && <ChartHistory patient={patient} />}
      </div>

      {/* HITL footer · 항상 하단 고정 */}
      <div className="px-6 py-2 text-[11px] flex items-start gap-2 flex-shrink-0" style={{ background: 'var(--rl-amber-soft)', borderTop: '1px solid var(--rl-amber)' }}>
        <AlertTriangle size={12} style={{ color: 'var(--rl-amber)', marginTop: 1, flexShrink: 0 }} />
        <div style={{ color: 'var(--rl-ink-2)' }}>
          <span className="font-medium" style={{ color: 'var(--rl-amber)' }}>본 시스템의 모든 AI 분석 결과는 진단 보조용입니다.</span>{' '}
          환자에 대한 최종 진단 및 치료 결정은 반드시 주치의의 임상적 판단에 따라야 합니다.
          <span className="font-mono ml-2" style={{ color: 'var(--rl-ink-3)' }}>[EU AI Act Art. 22]</span>
        </div>
      </div>
    </div>
  );
}

/* ----------- PATIENT BANNER (flex 자식, 항상 상단 고정) ----------- */
function PatientBanner({ patient, onHome, onAcknowledge }) {
  const isUnread = patient.status === 'ready' && !patient.acknowledged;
  return (
    <div
      className="bg-white"
      style={{ borderBottom: '1px solid var(--rl-border)', flexShrink: 0 }}
    >
      <div className="px-6 py-2.5 flex items-center gap-3 flex-wrap">
        <div className="w-11 h-11 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: 'var(--rl-primary)', color: 'white' }}>
          <User size={18} />
        </div>

        <div className="min-w-0">
          <div className="flex items-baseline gap-2">
            <div className="font-serif text-xl leading-none" style={{ color: 'var(--rl-ink)', letterSpacing: '-0.01em' }}>
              {patient.name}
            </div>
            <div className="text-sm" style={{ color: 'var(--rl-ink-2)' }}>
              {patient.sex === 'M' ? '남' : '여'} · {patient.age}세
            </div>
            <div className="font-mono text-xs" style={{ color: 'var(--rl-ink-3)' }}>· {patient.mrn}</div>
          </div>
          <div className="flex items-center gap-3 mt-1 text-[11px]" style={{ color: 'var(--rl-ink-3)' }}>
            <span className="flex items-center gap-1"><CalendarDays size={11} /> {patient.visitDate || '오늘 (2026.04.23)'} · {patient.time}</span>
            <span>· {patient.visit}</span>
          </div>
        </div>

        {/* Flags strip */}
        <div className="flex items-center gap-1.5 ml-4 flex-wrap">
          {patient.allergy && (
            <span className="chip" style={{ background: 'var(--rl-critical-soft)', color: 'var(--rl-critical)' }}>
              <Ban size={11} /> 알러지 · {patient.allergy}
            </span>
          )}
          {patient.dontMiss && (
            <span className="chip" style={{ background: 'var(--rl-amber-soft)', color: 'var(--rl-amber)' }}>
              <AlertTriangle size={11} /> Don't miss
            </span>
          )}
          {patient.rare && (
            <span className="chip" style={{ background: 'var(--rl-rare-soft)', color: 'var(--rl-rare)' }}>
              <Flame size={11} /> 희귀질환 의심
            </span>
          )}
        </div>

        <div className="ml-auto flex items-center gap-2">
          {isUnread && (
            <button
              onClick={() => onAcknowledge(patient.mrn)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium hover:opacity-90"
              style={{ background: 'var(--rl-teal)', color: 'white' }}
            >
              <CheckCircle2 size={12} /> 결과 확인 처리
            </button>
          )}
          <button
            onClick={onHome}
            className="p-1.5 rounded hover:bg-slate-100"
            style={{ color: 'var(--rl-ink-2)' }}
            title="홈으로"
          >
            <X size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}

/* ----------- CHART TABS ----------- */
function ChartTabs({ active, onChange, patient }) {
  const tabs = [
    { k: 'overview',  label: 'Overview',           icon: <Eye size={13} /> },
    { k: 'cxr',       label: 'CXR · AI',           icon: <ScanLine size={13} /> },
    { k: 'workspace', label: '진단 워크스페이스',  icon: <Microscope size={13} />, badge: 'W3' },
    { k: 'report',    label: '리포트',             icon: <FileText size={13} /> },
    { k: 'history',   label: '히스토리',           icon: <Clock size={13} /> },
  ];
  return (
    <div className="bg-white px-6" style={{
      borderBottom: '1px solid var(--rl-border-soft)',
      flexShrink: 0,
    }}>
      <div className="flex items-center gap-1">
        {tabs.map(t => {
          const isActive = active === t.k;
          return (
            <button
              key={t.k}
              onClick={() => onChange(t.k)}
              className="px-4 py-3 text-xs font-medium flex items-center gap-1.5 transition"
              style={{
                color: isActive ? 'var(--rl-primary)' : 'var(--rl-ink-2)',
                borderBottom: `2px solid ${isActive ? 'var(--rl-primary)' : 'transparent'}`,
                marginBottom: '-1px',
              }}
            >
              {t.icon}
              {t.label}
              {t.badge && (
                <span className="font-mono text-[9px] px-1 py-0.5 rounded" style={{ background: 'var(--rl-bg-3)', color: 'var(--rl-ink-3)' }}>
                  {t.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/* ----------- CHART OVERVIEW · 객관적 환자 정보 + Lab + CXR + Top 3 ----------- */
function ChartOverview({ patient }) {
  const top3 = (patient.preview || []).slice(0, 3);
  const [cxrView, setCxrView] = useState('original');

  return (
    <div
      className="grid gap-3 fade-in h-full"
      style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.25fr) minmax(0, 1fr)' }}
    >
      {/* 좌: 객관적 환자 정보 + 바이탈 + Lab */}
      <Panel title="환자 정보" mono="Demographics" fill>
        <div className="grid grid-cols-2 gap-2">
          <InfoCell label="나이 · 성별" value={`${patient.sex === 'M' ? '남' : '여'} · ${patient.age}세`} compact />
          <InfoCell label="MRN"          value={patient.mrn} mono compact />
          <InfoCell label="방문 유형"    value={patient.visit} compact />
          <InfoCell label="예약 시간"    value={patient.time} mono compact />
          <InfoCell label="방문 일자"    value={patient.visitDate || '오늘 (2026-04-23)'} mono compact />
          <InfoCell label="알러지"       value={patient.allergy || '—'} compact />
        </div>
        <div className="mt-2.5 pt-2.5" style={{ borderTop: '1px solid var(--rl-border-soft)' }}>
          <div className="font-mono text-[10px] uppercase tracking-widest mb-1" style={{ color: 'var(--rl-ink-3)' }}>주호소 · Chief complaint</div>
          <div className="text-sm leading-relaxed" style={{ color: 'var(--rl-ink)' }}>{patient.complaint}</div>
        </div>
        <div className="mt-2.5 pt-2.5" style={{ borderTop: '1px solid var(--rl-border-soft)' }}>
          <div className="font-mono text-[10px] uppercase tracking-widest mb-1" style={{ color: 'var(--rl-ink-3)' }}>바이탈 · Vitals</div>
          <div className="text-sm font-mono leading-relaxed" style={{ color: 'var(--rl-ink)' }}>{patient.vitals || DEFAULT_VITALS}</div>
        </div>
        <div className="mt-2.5 pt-2.5" style={{ borderTop: '1px solid var(--rl-border-soft)' }}>
          <LabSection patient={patient} />
        </div>
      </Panel>

      {/* 중: CXR */}
      <Panel
        title="CXR · Chest X-ray"
        mono="Frontal"
        fill
        right={
          <div className="flex items-center gap-2">
            <CxrViewToggle view={cxrView} onChange={setCxrView} />
            <button
              onClick={() => openCxrPopup(patient)}
              className="text-[11px] font-medium flex items-center gap-1 hover:underline"
              style={{ color: 'var(--rl-primary)' }}
            >
              확대 보기 <ArrowUpRight size={11} />
            </button>
          </div>
        }
      >
        <div
          className="hairline-strong rounded"
          style={{ background: '#0A1628', height: '100%', position: 'relative', overflow: 'hidden' }}
        >
          {patient.cxr === 'arrived' ? (
            <CXRMock heatmap={cxrView === 'heatmap'} />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <ScanLine size={32} style={{ color: 'rgba(255,255,255,0.3)', margin: '0 auto 8px' }} />
                <div className="font-mono text-[11px] uppercase tracking-widest" style={{ color: 'rgba(255,255,255,0.5)' }}>촬영 대기 중</div>
              </div>
            </div>
          )}
        </div>
      </Panel>

      {/* 우: Top 3 Differential */}
      <Panel
        title="Top 3 Differential"
        mono="528 → HPO → Top 3"
        fill
      >
        {patient.status === 'ready' && top3.length > 0 ? (
          <div className="space-y-2">
            {top3.map((dx, i) => (
              <DxPreviewRow
                key={i}
                rank={i + 1}
                {...dx}
                onClick={() => openDxEvidencePopup(patient, dx, i + 1)}
              />
            ))}
          </div>
        ) : patient.status === 'analyzing' ? (
          <div className="rounded h-full flex items-center justify-center text-sm gap-2" style={{ color: 'var(--rl-primary)', background: 'var(--rl-primary-soft)' }}>
            <Loader2 size={14} className="spin" />
            DenseNet-121 추론 진행 중...
          </div>
        ) : (
          <div className="rounded h-full flex items-center justify-center text-sm" style={{ color: 'var(--rl-ink-3)', background: 'var(--rl-bg-2)' }}>
            {patient.cxr === 'pending' ? 'CXR 촬영 대기 중' : 'AI 분석 대기 중'}
          </div>
        )}
      </Panel>
    </div>
  );
}

/* ----------- LAB SECTION · 카테고리 탭 + 결과 테이블 ----------- */
const DEFAULT_VITALS = 'BP 130/80 · HR 84 · RR 18 · SpO₂ 96% (RA) · T 36.7°C';

const DEFAULT_LABS = {
  cbc: [
    { name: 'WBC',          value: '7.4',  unit: '×10⁹/L', range: '4.0–10.0', flag: null },
    { name: 'Hb',           value: '13.8', unit: 'g/dL',   range: '13.0–17.0', flag: null },
    { name: 'Hct',          value: '41.2', unit: '%',      range: '40–52',    flag: null },
    { name: 'Plt',          value: '245',  unit: '×10⁹/L', range: '150–400',  flag: null },
    { name: 'Lymphocyte',   value: '1.8',  unit: '×10⁹/L', range: '1.0–4.0',  flag: null },
    { name: 'Neutrophil%',  value: '62.1', unit: '%',      range: '40–75',    flag: null },
    { name: 'Eosinophil%',  value: '2.4',  unit: '%',      range: '0–7',      flag: null },
  ],
  chem: [
    { name: 'BUN',     value: '14',    unit: 'mg/dL',  range: '8–20',     flag: null },
    { name: 'Cr',      value: '0.92',  unit: 'mg/dL',  range: '0.7–1.3',  flag: null },
    { name: 'eGFR',    value: '88',    unit: 'mL/min', range: '≥60',      flag: null },
    { name: 'Na',      value: '139',   unit: 'mmol/L', range: '136–145',  flag: null },
    { name: 'K',       value: '4.2',   unit: 'mmol/L', range: '3.5–5.0',  flag: null },
    { name: 'AST',     value: '24',    unit: 'U/L',    range: '10–40',    flag: null },
    { name: 'ALT',     value: '21',    unit: 'U/L',    range: '10–40',    flag: null },
    { name: 'Glucose', value: '102',   unit: 'mg/dL',  range: '70–110',   flag: null },
    { name: 'BNP',     value: '38',    unit: 'pg/mL',  range: '<100',     flag: null },
  ],
  abg: [
    { name: 'pH',           value: '7.41', unit: '',       range: '7.35–7.45', flag: null },
    { name: 'PaO₂',         value: '78',   unit: 'mmHg',   range: '80–100',    flag: 'low' },
    { name: 'PaCO₂',        value: '38',   unit: 'mmHg',   range: '35–45',     flag: null },
    { name: 'HCO₃⁻',        value: '24',   unit: 'mmol/L', range: '22–26',     flag: null },
    { name: 'SaO₂',         value: '94',   unit: '%',      range: '95–100',    flag: 'low' },
    { name: 'A-a gradient', value: '22',   unit: 'mmHg',   range: '<15',       flag: 'high' },
    { name: 'Lactate',      value: '1.2',  unit: 'mmol/L', range: '<2.0',      flag: null },
  ],
  inflam: [
    { name: 'CRP',           value: '0.8',  unit: 'mg/dL', range: '<0.5',  flag: 'high' },
    { name: 'ESR',           value: '38',   unit: 'mm/hr', range: '<20',   flag: 'high' },
    { name: 'Procalcitonin', value: '0.05', unit: 'ng/mL', range: '<0.05', flag: null },
    { name: 'KL-6',          value: '1284', unit: 'U/mL',  range: '<500',  flag: 'critical' },
    { name: 'SP-D',          value: '178',  unit: 'ng/mL', range: '<110',  flag: 'high' },
    { name: 'ANA',           value: '1:80', unit: '',      range: '<1:80', flag: null },
    { name: 'RF',            value: '14',   unit: 'IU/mL', range: '<14',   flag: null },
    { name: 'Anti-CCP',      value: '6',    unit: 'U/mL',  range: '<7',    flag: null },
  ],
};

/* Lab status 결정 · 환자 status 기반 (override: patient.labStatus) */
function getLabStatus(patient) {
  if (patient.labStatus) return patient.labStatus;
  if (patient.status === 'ready')     return 'ready';
  if (patient.status === 'analyzing') return 'pending';
  return 'none';
}

function LabSection({ patient }) {
  const labStatus = getLabStatus(patient);

  // 공통 헤더
  const Header = ({ trailing }) => (
    <div className="flex items-baseline gap-2 mb-2" style={{ whiteSpace: 'nowrap', minWidth: 0 }}>
      <div className="font-mono text-[10px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)', flexShrink: 0 }}>
        검사 결과 · Labs
      </div>
      <span className="ml-auto font-mono text-[10px] truncate" style={{ color: 'var(--rl-ink-4)', minWidth: 0 }}>
        {trailing}
      </span>
    </div>
  );

  if (labStatus === 'none') {
    return (
      <div>
        <Header trailing="미오더" />
        <LabNoneState />
      </div>
    );
  }

  if (labStatus === 'pending') {
    return (
      <div>
        <Header trailing="검체 채취 · 결과 대기" />
        <LabPendingState />
      </div>
    );
  }

  // ready
  return <LabReadyState patient={patient} />;
}

function LabNoneState() {
  return (
    <div
      className="rounded p-3 text-center text-xs flex flex-col items-center gap-1.5"
      style={{ background: 'var(--rl-bg-2)', border: '1px dashed var(--rl-border)' }}
    >
      <Microscope size={18} style={{ color: 'var(--rl-ink-4)' }} />
      <div style={{ color: 'var(--rl-ink-3)' }}>처방된 검사가 없습니다</div>
      <button
        className="font-mono text-[10px] uppercase tracking-widest hover:underline mt-0.5"
        style={{ color: 'var(--rl-primary)' }}
      >
        + 검사 처방
      </button>
    </div>
  );
}

function LabPendingState() {
  return (
    <div
      className="rounded p-3 text-xs flex flex-col items-center gap-1.5"
      style={{ background: 'var(--rl-amber-soft)', border: '1px solid var(--rl-amber)' }}
    >
      <Loader2 size={18} className="spin" style={{ color: 'var(--rl-amber)' }} />
      <div className="font-medium text-center" style={{ color: 'var(--rl-amber)' }}>
        검체 채취 완료 · 결과 분석 중
      </div>
      <div className="font-mono text-[10px] text-center" style={{ color: 'var(--rl-ink-3)' }}>
        채혈 07:42 KST · 예상 결과 09:15
      </div>
      <div className="grid grid-cols-2 gap-1 w-full mt-1">
        {['CBC', 'Chem', 'ABG', 'Markers'].map(k => (
          <div
            key={k}
            className="flex items-center gap-1.5 px-2 py-1 rounded"
            style={{ background: 'rgba(255,255,255,0.6)' }}
          >
            <Loader2 size={9} className="spin" style={{ color: 'var(--rl-amber)' }} />
            <span className="font-mono text-[10px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-2)' }}>{k}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function LabReadyState({ patient }) {
  const [tab, setTab] = useState('cbc');
  const labs = patient.labs || DEFAULT_LABS;
  const tabs = [
    { k: 'cbc',    label: 'CBC' },
    { k: 'chem',   label: 'Chem' },
    { k: 'abg',    label: 'ABG' },
    { k: 'inflam', label: 'Markers' },
  ];
  const rows = labs[tab] || [];
  const totalAbnormal = ['cbc','chem','abg','inflam'].reduce(
    (s, k) => s + (labs[k] || []).filter(r => r.flag).length, 0
  );

  return (
    <div>
      <div className="flex items-baseline gap-2 mb-2" style={{ whiteSpace: 'nowrap', minWidth: 0 }}>
        <div className="font-mono text-[10px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)', flexShrink: 0 }}>
          검사 결과 · Labs
        </div>
        {totalAbnormal > 0 && (
          <span className="font-mono text-[10px]" style={{ color: 'var(--rl-amber)', flexShrink: 0 }}>
            · {totalAbnormal}건 이상치
          </span>
        )}
        <span className="ml-auto font-mono text-[10px] truncate" style={{ color: 'var(--rl-ink-4)', minWidth: 0 }}>
          {patient.visitDate || '오늘'} 채혈
        </span>
      </div>

      {/* Lab category tabs */}
      <div className="flex items-center gap-1 mb-2" style={{ whiteSpace: 'nowrap' }}>
        {tabs.map(t => {
          const active = tab === t.k;
          const ab = (labs[t.k] || []).filter(r => r.flag).length;
          return (
            <button
              key={t.k}
              onClick={() => setTab(t.k)}
              className="px-2 py-0.5 text-[10px] font-mono font-medium uppercase tracking-wider rounded transition flex items-center gap-1"
              style={{
                background: active ? 'var(--rl-primary)' : 'var(--rl-bg-3)',
                color: active ? 'white' : 'var(--rl-ink-2)',
              }}
            >
              {t.label}
              {ab > 0 && (
                <span
                  className="font-mono text-[9px] px-1 rounded"
                  style={{
                    background: active ? 'rgba(255,255,255,0.22)' : 'var(--rl-amber-soft)',
                    color: active ? 'white' : 'var(--rl-amber)',
                  }}
                >
                  {ab}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Lab table · header + rows · 4-column 고정 grid */}
      <div className="grid items-baseline font-mono text-[9px] uppercase tracking-widest pb-1" style={{
        gridTemplateColumns: LAB_GRID,
        columnGap: 6,
        color: 'var(--rl-ink-4)',
        borderBottom: '1px solid var(--rl-border-soft)',
      }}>
        <div>검사</div>
        <div style={{ textAlign: 'right' }}>값</div>
        <div>단위</div>
        <div style={{ textAlign: 'right' }}>정상범위</div>
      </div>
      <div>
        {rows.map((r, i) => <LabRow key={i} {...r} isLast={i === rows.length - 1} />)}
      </div>
    </div>
  );
}

const LAB_GRID = 'minmax(0, 1fr) 56px 60px 72px';

function LabRow({ name, value, unit, range, flag, isLast }) {
  const flagColor =
    flag === 'critical' ? 'var(--rl-critical)' :
    flag === 'high'     ? 'var(--rl-critical)' :
    flag === 'low'      ? 'var(--rl-amber)' :
    'var(--rl-ink)';
  const flagSymbol =
    flag === 'critical' ? '↑↑' :
    flag === 'high'     ? '↑' :
    flag === 'low'      ? '↓' :
    '';
  return (
    <div
      className="grid items-baseline py-1"
      style={{
        gridTemplateColumns: LAB_GRID,
        columnGap: 6,
        borderBottom: isLast ? 'none' : '1px solid var(--rl-border-soft)',
      }}
    >
      <div className="text-[11px] truncate" style={{ color: 'var(--rl-ink-2)' }}>{name}</div>
      <div
        className="font-mono text-[11px] flex items-baseline gap-0.5 justify-end"
        style={{ color: flagColor, fontWeight: flag ? 600 : 400 }}
      >
        <span>{value}</span>
        {flagSymbol && <span className="text-[10px]">{flagSymbol}</span>}
      </div>
      <div className="font-mono text-[10px] truncate" style={{ color: 'var(--rl-ink-3)' }}>{unit}</div>
      <div className="font-mono text-[10px] text-right" style={{ color: 'var(--rl-ink-4)' }}>{range}</div>
    </div>
  );
}

function Panel({ title, mono, right, children, fill }) {
  return (
    <div
      className="hairline rounded bg-white"
      style={fill ? { display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden', minWidth: 0 } : { minWidth: 0 }}
    >
      <div
        className="px-3 py-2 flex items-center gap-2"
        style={{ borderBottom: '1px solid var(--rl-border-soft)', flexShrink: 0, whiteSpace: 'nowrap', minWidth: 0 }}
      >
        <div className="font-mono text-[10px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)', flexShrink: 0 }}>
          {mono}
        </div>
        <div className="text-sm font-medium truncate" style={{ color: 'var(--rl-ink)', minWidth: 0 }}>{title}</div>
        {right && <div className="ml-auto" style={{ flexShrink: 0, whiteSpace: 'nowrap' }}>{right}</div>}
      </div>
      <div
        className="p-3"
        style={fill ? { flex: 1, minHeight: 0, overflow: 'auto' } : undefined}
      >
        {children}
      </div>
    </div>
  );
}

/* ----------- TAB · CXR (좌 원본 / 우 AI Heatmap 비교) ----------- */
function ChartCXR({ patient }) {
  return (
    <div className="h-full fade-in">
      <Panel
        title="CXR · 비교 뷰어"
        mono="Original ↔ AI Heatmap"
        fill
        right={
          <button
            onClick={() => openCxrPopup(patient)}
            className="text-[11px] font-medium flex items-center gap-1 hover:underline"
            style={{ color: 'var(--rl-primary)' }}
          >
            새 창에서 보기 <ArrowUpRight size={11} />
          </button>
        }
      >
        <div className="h-full flex flex-col gap-2" style={{ minHeight: 0 }}>
          <div className="flex gap-4 flex-1 justify-center items-stretch" style={{ minHeight: 0 }}>
            <CxrFrame patient={patient} heatmap={false} caption="원본 · Original" />
            <CxrFrame patient={patient} heatmap={true}  caption="AI Heatmap · Grad-CAM" />
          </div>
          <div className="text-[11px] text-center flex-shrink-0" style={{ color: 'var(--rl-ink-3)' }}>
            PACS 통합 + Window/Level + 측정 도구는 <span className="font-mono">W3 · 5/4~5/10</span> 구현 예정
          </div>
        </div>
      </Panel>
    </div>
  );
}

function CxrFrame({ patient, heatmap, caption }) {
  const captionColor = heatmap ? 'var(--rl-amber)' : 'var(--rl-ink-2)';
  return (
    <div className="flex flex-col items-center" style={{ minHeight: 0, minWidth: 0 }}>
      <div
        className="font-mono text-[10px] uppercase tracking-widest mb-1.5 flex items-center gap-1.5"
        style={{ color: captionColor }}
      >
        <span className="w-1.5 h-1.5 rounded-full" style={{ background: captionColor }} />
        {caption}
      </div>
      {/* Square frame · height 기준 1:1, 양 옆 흰 여백은 부모(Panel) 흰 배경 */}
      <div
        className="hairline-strong rounded"
        style={{
          background: '#0A1628',
          position: 'relative',
          overflow: 'hidden',
          borderColor: heatmap ? 'rgba(180,83,9,0.4)' : undefined,
          flex: '1 1 0',
          minHeight: 0,
          aspectRatio: '1 / 1',
          maxWidth: '100%',
        }}
      >
        {patient.cxr === 'arrived'
          ? <CXRMock heatmap={heatmap} />
          : <div className="absolute inset-0 flex items-center justify-center text-white opacity-50 font-mono text-xs">촬영 대기 중</div>
        }
      </div>
    </div>
  );
}

/* ----------- CXR view toggle (원본 ↔ AI Heatmap) ----------- */
function CxrViewToggle({ view, onChange }) {
  const tabs = [
    { k: 'original', label: '원본' },
    { k: 'heatmap',  label: 'Heatmap' },
  ];
  return (
    <div className="flex items-center" style={{ background: 'var(--rl-bg-3)', padding: 2, borderRadius: 4 }}>
      {tabs.map(t => {
        const active = view === t.k;
        return (
          <button
            key={t.k}
            onClick={() => onChange(t.k)}
            className="px-2 py-0.5 text-[10px] font-mono font-medium uppercase tracking-wider rounded transition"
            style={{
              background: active ? 'var(--rl-primary)' : 'transparent',
              color: active ? 'white' : 'var(--rl-ink-2)',
            }}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}

/* ----------- TAB · WORKSPACE placeholder ----------- */
function ChartWorkspacePlaceholder() {
  return (
    <div className="fade-in hairline rounded bg-white p-12 text-center">
      <Microscope size={32} style={{ color: 'var(--rl-primary)', margin: '0 auto 12px' }} />
      <div className="font-serif text-xl mb-1" style={{ color: 'var(--rl-ink)' }}>3-Panel 진단 워크스페이스</div>
      <div className="text-sm mb-4" style={{ color: 'var(--rl-ink-3)' }}>
        좌(증상·HPO 입력) · 중(CXR + 모델) · 우(528 감별진단 + LR 막대)
      </div>
      <div className="font-mono text-[11px] uppercase tracking-widest" style={{ color: 'var(--rl-amber)' }}>
        Week 3 · 5/4 ~ 5/10 구현 예정
      </div>
      <div className="text-[11px] mt-2" style={{ color: 'var(--rl-ink-3)' }}>
        근거: PDF §4.1 3-Panel Layout · Robinson 2020 LR 막대
      </div>
    </div>
  );
}

/* ----------- TAB · REPORT (PDF mock 미리보기 + 새 창 전체화면) ----------- */
function ChartReport({ patient }) {
  return (
    <div className="h-full fade-in">
      <Panel
        title="진단 리포트 · Preview"
        mono="Report viewer · #06"
        fill
        right={
          <button
            onClick={() => openReportPopup(patient)}
            className="text-[11px] font-medium flex items-center gap-1 hover:underline"
            style={{ color: 'var(--rl-primary)' }}
          >
            전체화면 보기 <ArrowUpRight size={11} />
          </button>
        }
      >
        <div
          className="h-full"
          style={{ overflow: 'auto', background: 'var(--rl-bg-3)', padding: 16, minHeight: 0 }}
        >
          <ReportPage patient={patient} pageNo={1} totalPages={2} />
          <ReportPage patient={patient} pageNo={2} totalPages={2} />
          {patient.finalReport && <AIRagReport finalReport={patient.finalReport} />}
        </div>
      </Panel>
    </div>
  );
}

function ReportPage({ patient, pageNo, totalPages }) {
  return (
    <div
      className="bg-white mx-auto mb-3"
      style={{
        aspectRatio: '210 / 297',
        maxWidth: 540,
        boxShadow: '0 4px 16px rgba(10,22,40,0.12)',
        border: '1px solid var(--rl-border-soft)',
        padding: 28,
        display: 'flex',
        flexDirection: 'column',
        fontSize: 11,
        color: 'var(--rl-ink)',
      }}
    >
      {/* Letterhead */}
      <div className="flex items-baseline gap-3 pb-3" style={{ borderBottom: '2px solid var(--rl-primary)' }}>
        <div className="font-serif text-base leading-none" style={{ color: 'var(--rl-primary-dark)' }}>
          성균관대학교병원
        </div>
        <div className="text-[10px]" style={{ color: 'var(--rl-ink-3)' }}>호흡기내과 · Pulmonary Division</div>
        <div className="ml-auto font-mono text-[9px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)' }}>
          AI-Assisted Diagnostic Report
        </div>
      </div>

      <div className="flex justify-between items-baseline mt-3 mb-3 font-mono text-[9px]" style={{ color: 'var(--rl-ink-3)' }}>
        <div>Report ID · RPT-2026-0423-{patient.mrn.replace('-', '')}</div>
        <div>Page {pageNo} / {totalPages} · 발행 2026-04-23 09:14 KST</div>
      </div>

      {pageNo === 1 ? (
        <ReportPage1 patient={patient} />
      ) : (
        <ReportPage2 patient={patient} />
      )}

      {/* Footer disclaimer · 모든 페이지 공통 */}
      <div className="mt-auto pt-3" style={{ borderTop: '1px solid var(--rl-border-soft)' }}>
        <div className="font-mono text-[8px] uppercase tracking-widest" style={{ color: 'var(--rl-amber)' }}>
          ⚠ AI-Assisted · Final diagnosis requires physician review
        </div>
        <div className="text-[9px] mt-1" style={{ color: 'var(--rl-ink-3)' }}>
          본 리포트의 AI 분석 결과는 진단 보조용이며 최종 진단 및 치료 결정은 주치의의 임상적 판단에 따릅니다. [EU AI Act Art. 22]
        </div>
      </div>
    </div>
  );
}

function ReportPage1({ patient }) {
  const top = (patient.preview || [])[0];
  return (
    <>
      {/* Patient block */}
      <div className="grid grid-cols-3 gap-3 mb-3 p-2.5" style={{ background: 'var(--rl-bg-2)', border: '1px solid var(--rl-border-soft)' }}>
        <ReportField label="환자명" value={patient.name} />
        <ReportField label="MRN" value={patient.mrn} mono />
        <ReportField label="나이 · 성별" value={`${patient.sex === 'M' ? '남' : '여'} · ${patient.age}세`} />
        <ReportField label="방문 일자" value={patient.visitDate || '2026-04-23'} mono />
        <ReportField label="방문 유형" value={patient.visit} />
        <ReportField label="알러지" value={patient.allergy || '없음'} />
      </div>

      {/* Chief complaint */}
      <ReportSection title="1. 주호소 · Chief Complaint">
        <div className="text-[11px] leading-relaxed">{patient.complaint}</div>
      </ReportSection>

      {/* AI Differential */}
      <ReportSection title="2. AI 감별진단 · Top 3 Differential (DenseNet-121 + HPO-LR)">
        <table className="w-full text-[10px]" style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--rl-border)' }}>
              <th className="text-left py-1 font-mono text-[9px] uppercase" style={{ color: 'var(--rl-ink-3)' }}>#</th>
              <th className="text-left py-1 font-mono text-[9px] uppercase" style={{ color: 'var(--rl-ink-3)' }}>진단명</th>
              <th className="text-right py-1 font-mono text-[9px] uppercase" style={{ color: 'var(--rl-ink-3)' }}>확률</th>
              <th className="text-left py-1 font-mono text-[9px] uppercase pl-2" style={{ color: 'var(--rl-ink-3)' }}>플래그</th>
            </tr>
          </thead>
          <tbody>
            {(patient.preview || []).slice(0, 3).map((dx, i) => (
              <tr key={i} style={{ borderBottom: '1px solid var(--rl-border-soft)' }}>
                <td className="py-1.5 font-mono">{i + 1}</td>
                <td className="py-1.5">
                  {dx.name}
                  {dx.orpha && <span className="font-mono ml-1.5 text-[9px]" style={{ color: 'var(--rl-ink-3)' }}>{dx.orpha}</span>}
                </td>
                <td className="py-1.5 text-right font-serif">{(dx.prob * 100).toFixed(0)}%</td>
                <td className="py-1.5 pl-2">
                  {dx.dontMiss && <span className="font-mono text-[9px]" style={{ color: 'var(--rl-amber)' }}>Don't miss</span>}
                  {dx.dontMiss && dx.rare && ' · '}
                  {dx.rare && <span className="font-mono text-[9px]" style={{ color: 'var(--rl-rare)' }}>희귀</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </ReportSection>

      {/* Recommendation */}
      <ReportSection title="3. 권고 · Recommendation">
        <ul className="text-[10px] leading-relaxed pl-4" style={{ listStyle: 'disc' }}>
          {top && top.dontMiss && <li><b>{top.name}</b> 의심 — HRCT 및 폐기능검사(PFT) 우선 권고</li>}
          {top && top.rare && <li>희귀질환 가능성 → 호흡기내과 + 영상의학과 multidisciplinary discussion(MDT) 권고</li>}
          <li>주치의 검토 후 진단 확정 및 치료 방향 결정 필요</li>
          <li>관련 lab 추가 권고: BAL fluid 분석, 자가항체 패널 (ANA, RF, Anti-CCP)</li>
        </ul>
      </ReportSection>
    </>
  );
}

function ReportPage2({ patient }) {
  return (
    <>
      <ReportSection title="4. CXR · Frontal + AI Heatmap (Grad-CAM)">
        <div className="grid grid-cols-2 gap-2 mb-2">
          <div>
            <div className="font-mono text-[8px] uppercase tracking-widest mb-1" style={{ color: 'var(--rl-ink-3)' }}>원본 · Original</div>
            <div style={{ background: '#0A1628', aspectRatio: '1 / 1', overflow: 'hidden', border: '1px solid var(--rl-border)' }}>
              <CXRMock heatmap={false} />
            </div>
          </div>
          <div>
            <div className="font-mono text-[8px] uppercase tracking-widest mb-1" style={{ color: 'var(--rl-amber)' }}>AI Heatmap · Grad-CAM</div>
            <div style={{ background: '#0A1628', aspectRatio: '1 / 1', overflow: 'hidden', border: '1px solid rgba(180,83,9,0.4)' }}>
              <CXRMock heatmap={true} />
            </div>
          </div>
        </div>
        <div className="text-[9px]" style={{ color: 'var(--rl-ink-3)' }}>
          AI 모델: DenseNet-121 · 입력 448×448 · Heatmap 활성 영역: 양측 폐 하부 reticular pattern (HRCT 권고)
        </div>
      </ReportSection>

      <ReportSection title="5. 임상 소견 · Clinical Notes">
        <div className="text-[10px] leading-relaxed">
          본 환자는 <b>{patient.complaint}</b>로 내원한 {patient.age}세 {patient.sex === 'M' ? '남성' : '여성'}으로,
          AI 보조 분석 결과 상위 감별진단 중 <b>{(patient.preview || [{ name: '—' }])[0].name}</b>이 가장 확률 높게 제시되었습니다.
          단, AI 결과는 진단 보조이며 최종 진단은 주치의의 임상 판단과 추가 검사 결과를 종합하여 확정합니다.
        </div>
      </ReportSection>

      <ReportSection title="6. 의사 서명 · Physician Sign-off">
        <div className="grid grid-cols-2 gap-3 mt-3">
          <div>
            <div className="font-mono text-[9px] uppercase tracking-widest mb-2" style={{ color: 'var(--rl-ink-3)' }}>주치의</div>
            <div className="font-serif text-sm" style={{ borderBottom: '1px solid var(--rl-ink)', paddingBottom: 2 }}>
              정민수 과장
            </div>
            <div className="font-mono text-[9px] mt-1" style={{ color: 'var(--rl-ink-3)' }}>호흡기내과 · 면허 #12345</div>
          </div>
          <div>
            <div className="font-mono text-[9px] uppercase tracking-widest mb-2" style={{ color: 'var(--rl-ink-3)' }}>전자 서명일</div>
            <div className="font-mono text-sm" style={{ borderBottom: '1px solid var(--rl-ink)', paddingBottom: 2 }}>
              2026-04-23 09:14 KST
            </div>
            <div className="font-mono text-[9px] mt-1" style={{ color: 'var(--rl-ink-3)' }}>HASH · SHA256:a4f9…7c2b</div>
          </div>
        </div>
      </ReportSection>
    </>
  );
}

function ReportField({ label, value, mono }) {
  return (
    <div>
      <div className="font-mono text-[8px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)' }}>{label}</div>
      <div className={`text-[11px] ${mono ? 'font-mono' : ''}`} style={{ color: 'var(--rl-ink)' }}>{value}</div>
    </div>
  );
}

function ReportSection({ title, children }) {
  return (
    <div className="mb-3">
      <div className="font-mono text-[10px] uppercase tracking-widest mb-1.5" style={{ color: 'var(--rl-primary)' }}>
        {title}
      </div>
      {children}
    </div>
  );
}

/* AI RAG 리포트 팝업 페이지 HTML 생성 헬퍼 */
function buildAiPageHtml(fr) {
  if (!fr) return '';
  const data = fr.diagnosis_json || fr;
  const rec = data.recommendation || {};
  const notes = data.clinical_notes || {};
  const conf = data.confidence_metrics || {};
  const score = conf.overall_confidence_score || 0;
  const scoreColor = score >= 0.8 ? '#0E8574' : score >= 0.6 ? '#B45309' : '#A32D2D';
  const scoreLabel = score >= 0.8 ? 'High' : score >= 0.6 ? 'Medium' : 'Low';
  const listHtml = (items) => (items || []).map(i => `<li>${i}</li>`).join('');
  const suf = conf.data_sufficiency || {};
  const sufColor = (v) => v === 'High' ? '#0E8574' : v === 'Medium' ? '#B45309' : '#64748B';
  const sufRow = Object.entries(suf).map(([k, v]) => {
    const label = k === 'genomic_evidence' ? '유전체' : k === 'clinical_case_match' ? '임상케이스' : '임상시험';
    return `<span style="margin-right:12px;font-size:10px;"><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${sufColor(v)};margin-right:3px;vertical-align:middle;"></span>${label}: <b style="color:${sufColor(v)}">${v}</b></span>`;
  }).join('');

  return `
  <div class="page">
    <div class="letterhead">
      <div class="name" style="font-size:15px;">AI 진단 보조 리포트</div>
      <div class="div">Rare-Link RAG Phase 5</div>
      <div class="label">${(fr.llm_model || 'Claude 3.5 Sonnet').replace('anthropic.', '').slice(0, 35)}</div>
    </div>
    <div class="meta">
      <span>신뢰도 · <b style="color:${scoreColor}">${(score * 100).toFixed(0)}% (${scoreLabel})</b> &nbsp; ${sufRow}</span>
      <span style="font-size:9px;">${fr.generated_at ? new Date(fr.generated_at).toLocaleString('ko-KR', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : ''}</span>
    </div>
    ${conf.rationale ? `<div style="font-size:10px;color:#64748B;margin-bottom:10px;padding:6px 8px;background:#F8FAFC;border:1px solid #E2E8F0;">${conf.rationale}</div>` : ''}
    ${notes.summary ? `<div class="section"><div class="title">임상 종합 소견</div><div style="margin-bottom:5px;">${notes.summary}</div>${notes.top1_reasoning ? `<div style="margin-top:5px;"><span class="mono muted">TOP 1 근거 ·</span> ${notes.top1_reasoning}</div>` : ''}${notes.differential_note ? `<div style="margin-top:5px;"><span class="mono muted">감별 진단 ·</span> ${notes.differential_note}</div>` : ''}${notes.epidemiology_note ? `<div style="margin-top:5px;"><span class="rare mono" style="font-size:9px;">희귀질환 역학 ·</span> <span style="color:#6B21A8;">${notes.epidemiology_note}</span></div>` : ''}</div>` : ''}
    ${rec.immediate_workup?.length ? `<div class="section"><div class="title">즉각 검사 권고</div><ul>${listHtml(rec.immediate_workup)}</ul></div>` : ''}
    ${rec.specialist_referral?.length ? `<div class="section"><div class="title">전문과 의뢰 <span class="amber">(MDT)</span></div><ul>${listHtml(rec.specialist_referral)}</ul></div>` : ''}
    ${rec.treatment_guideline?.length ? `<div class="section"><div class="title">치료 가이드라인</div><ul>${listHtml(rec.treatment_guideline)}</ul></div>` : ''}
    ${rec.genetic_test?.length ? `<div class="section"><div class="title" style="color:#6B21A8;">유전자 검사 (희귀질환)</div><ul>${listHtml(rec.genetic_test)}</ul></div>` : ''}
    ${rec.additional_lab?.length ? `<div class="section"><div class="title">추가 Lab 권고</div><ul>${listHtml(rec.additional_lab)}</ul></div>` : ''}
    ${rec.clinical_trial_info?.length ? `<div class="section"><div class="title">임상시험 정보</div><ul>${listHtml(rec.clinical_trial_info)}</ul></div>` : ''}
    ${notes.rag_evidence ? `<div class="section"><div class="title">RAG 근거 (DB·API 교차검증)</div><div>${notes.rag_evidence}</div></div>` : ''}
    ${notes.case_comparison ? `<div class="section"><div class="title">PubMed 케이스 비교</div><div>${notes.case_comparison}</div></div>` : ''}
    <div class="footer">
      <div class="warn">⚠ AI-Assisted RAG · Rare-Link Phase 5 · Final diagnosis requires physician review</div>
      <div class="disc">${notes.disclaimer || 'AI 결과는 진단 보조이며 최종 진단은 주치의의 임상 판단과 추가 검사 결과를 종합하여 확정합니다.'} [EU AI Act Art. 22]</div>
    </div>
  </div>`;
}

/* PDF 전체화면 popup · 같은 리포트 콘텐츠를 새 창에 풀스크린으로 */
function openReportPopup(patient) {
  const w = window.open('', `rpt-${patient.mrn}`, 'width=900,height=1100,resizable=yes,scrollbars=yes');
  if (!w) {
    alert('팝업 차단을 해제해주세요.');
    return;
  }
  const sex = patient.sex === 'M' ? '남' : '여';
  const date = patient.visitDate || '2026-04-23';
  const top3 = (patient.preview || []).slice(0, 3);
  const top = top3[0] || { name: '—' };

  const dxRows = top3.map((dx, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${dx.name}${dx.orpha ? ` <span class="mono muted">${dx.orpha}</span>` : ''}</td>
      <td class="right serif">${(dx.prob * 100).toFixed(0)}%</td>
      <td>${dx.dontMiss ? '<span class="amber">Don\'t miss</span>' : ''}${dx.dontMiss && dx.rare ? ' · ' : ''}${dx.rare ? '<span class="rare">희귀</span>' : ''}</td>
    </tr>`).join('');

  const recs = [];
  if (top.dontMiss) recs.push(`<b>${top.name}</b> 의심 — HRCT 및 폐기능검사(PFT) 우선 권고`);
  if (top.rare) recs.push('희귀질환 가능성 → 호흡기내과 + 영상의학과 MDT 권고');
  recs.push('주치의 검토 후 진단 확정 및 치료 방향 결정 필요');
  recs.push('관련 lab 추가 권고: BAL fluid 분석, 자가항체 패널 (ANA, RF, Anti-CCP)');

  w.document.write(`<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<title>리포트 · ${patient.name} · ${patient.mrn}</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Serif:wght@500&family=IBM+Plex+Mono&display=swap" rel="stylesheet" />
<style>
  html, body { margin: 0; padding: 0; }
  body { background: #F1F5F9; font-family: 'IBM Plex Sans KR', sans-serif; color: #0A1628; padding: 24px; -webkit-font-smoothing: antialiased; }
  .page {
    background: white; max-width: 794px; margin: 0 auto 16px;
    box-shadow: 0 4px 16px rgba(10,22,40,0.12);
    border: 1px solid #E2E8F0;
    padding: 48px;
    aspect-ratio: 210 / 297;
    display: flex; flex-direction: column;
    font-size: 12px;
  }
  .letterhead { display: flex; align-items: baseline; gap: 12px; padding-bottom: 8px; border-bottom: 2px solid #0C447C; }
  .letterhead .name { font-family: 'IBM Plex Serif', serif; font-size: 18px; color: #083158; }
  .letterhead .div { font-size: 11px; color: #64748B; }
  .letterhead .label { margin-left: auto; font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em; color: #64748B; }
  .meta { display: flex; justify-content: space-between; margin: 12px 0; font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #64748B; }
  .patient { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; padding: 12px; background: #F8FAFC; border: 1px solid #E2E8F0; margin-bottom: 14px; }
  .patient .label { font-family: 'IBM Plex Mono', monospace; font-size: 9px; text-transform: uppercase; letter-spacing: 0.15em; color: #64748B; }
  .patient .val { font-size: 12px; }
  .section { margin-bottom: 14px; }
  .section .title { font-family: 'IBM Plex Mono', monospace; font-size: 11px; text-transform: uppercase; letter-spacing: 0.15em; color: #0C447C; margin-bottom: 6px; }
  table { width: 100%; border-collapse: collapse; font-size: 11px; }
  table th { text-align: left; padding: 4px 0; border-bottom: 1px solid #CBD5E1; font-family: 'IBM Plex Mono', monospace; font-size: 9px; text-transform: uppercase; color: #64748B; }
  table td { padding: 6px 0; border-bottom: 1px solid #E2E8F0; }
  table .right { text-align: right; }
  .serif { font-family: 'IBM Plex Serif', serif; }
  .mono { font-family: 'IBM Plex Mono', monospace; }
  .muted { color: #64748B; font-size: 10px; }
  .amber { color: #B45309; font-family: 'IBM Plex Mono', monospace; font-size: 10px; }
  .rare { color: #6B21A8; font-family: 'IBM Plex Mono', monospace; font-size: 10px; }
  ul { font-size: 11px; line-height: 1.6; padding-left: 18px; margin: 0; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .img { background: #0A1628; aspect-ratio: 1 / 1; overflow: hidden; border: 1px solid #CBD5E1; }
  .img.heat { border-color: rgba(180,83,9,0.4); }
  .img-label { font-family: 'IBM Plex Mono', monospace; font-size: 9px; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 4px; color: #334155; }
  .img-label.heat { color: #B45309; }
  .signoff { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }
  .signoff .label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em; color: #64748B; margin-bottom: 8px; }
  .signoff .line { border-bottom: 1px solid #0A1628; padding-bottom: 2px; }
  .signoff .name-sig { font-family: 'IBM Plex Serif', serif; font-size: 14px; }
  .signoff .date-sig { font-family: 'IBM Plex Mono', monospace; font-size: 14px; }
  .signoff .hash { font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: #64748B; margin-top: 4px; }
  .footer { margin-top: auto; padding-top: 10px; border-top: 1px solid #E2E8F0; }
  .footer .warn { font-family: 'IBM Plex Mono', monospace; font-size: 9px; text-transform: uppercase; letter-spacing: 0.15em; color: #B45309; }
  .footer .disc { font-size: 10px; color: #64748B; margin-top: 3px; }
</style>
</head>
<body>
  <div class="page">
    <div class="letterhead">
      <div class="name">성균관대학교병원</div>
      <div class="div">호흡기내과 · Pulmonary Division</div>
      <div class="label">AI-Assisted Diagnostic Report</div>
    </div>
    <div class="meta">
      <span>Report ID · RPT-2026-0423-${patient.mrn.replace('-', '')}</span>
      <span>Page 1 / 2 · 발행 2026-04-23 09:14 KST</span>
    </div>
    <div class="patient">
      <div><div class="label">환자명</div><div class="val">${patient.name}</div></div>
      <div><div class="label">MRN</div><div class="val mono">${patient.mrn}</div></div>
      <div><div class="label">나이 · 성별</div><div class="val">${sex} · ${patient.age}세</div></div>
      <div><div class="label">방문 일자</div><div class="val mono">${date}</div></div>
      <div><div class="label">방문 유형</div><div class="val">${patient.visit}</div></div>
      <div><div class="label">알러지</div><div class="val">${patient.allergy || '없음'}</div></div>
    </div>
    <div class="section">
      <div class="title">1. 주호소 · Chief Complaint</div>
      <div>${patient.complaint}</div>
    </div>
    <div class="section">
      <div class="title">2. AI 감별진단 · Top 3 Differential (DenseNet-121 + HPO-LR)</div>
      <table>
        <thead><tr><th>#</th><th>진단명</th><th class="right">확률</th><th>플래그</th></tr></thead>
        <tbody>${dxRows}</tbody>
      </table>
    </div>
    <div class="section">
      <div class="title">3. 권고 · Recommendation</div>
      <ul>${recs.map(r => `<li>${r}</li>`).join('')}</ul>
    </div>
    <div class="footer">
      <div class="warn">⚠ AI-Assisted · Final diagnosis requires physician review</div>
      <div class="disc">본 리포트의 AI 분석 결과는 진단 보조용이며 최종 진단 및 치료 결정은 주치의의 임상적 판단에 따릅니다. [EU AI Act Art. 22]</div>
    </div>
  </div>

  <div class="page">
    <div class="letterhead">
      <div class="name">성균관대학교병원</div>
      <div class="div">호흡기내과 · Pulmonary Division</div>
      <div class="label">AI-Assisted Diagnostic Report</div>
    </div>
    <div class="meta">
      <span>Report ID · RPT-2026-0423-${patient.mrn.replace('-', '')}</span>
      <span>Page 2 / 2 · 발행 2026-04-23 09:14 KST</span>
    </div>
    <div class="section">
      <div class="title">4. CXR · Frontal + AI Heatmap (Grad-CAM)</div>
      <div class="grid2">
        <div>
          <div class="img-label">원본 · Original</div>
          <div class="img">${buildCxrSvg({ heatmap: false })}</div>
        </div>
        <div>
          <div class="img-label heat">AI Heatmap · Grad-CAM</div>
          <div class="img heat">${buildCxrSvg({ heatmap: true })}</div>
        </div>
      </div>
      <div class="muted" style="margin-top:6px;">AI 모델: DenseNet-121 · 입력 448×448 · Heatmap 활성 영역: 양측 폐 하부 reticular pattern (HRCT 권고)</div>
    </div>
    <div class="section">
      <div class="title">5. 임상 소견 · Clinical Notes</div>
      <div>본 환자는 <b>${patient.complaint}</b>로 내원한 ${patient.age}세 ${sex === '남' ? '남성' : '여성'}으로, AI 보조 분석 결과 상위 감별진단 중 <b>${top.name}</b>이 가장 확률 높게 제시되었습니다. 단, AI 결과는 진단 보조이며 최종 진단은 주치의의 임상 판단과 추가 검사 결과를 종합하여 확정합니다.</div>
    </div>
    <div class="section">
      <div class="title">6. 의사 서명 · Physician Sign-off</div>
      <div class="signoff">
        <div>
          <div class="label">주치의</div>
          <div class="line name-sig">정민수 과장</div>
          <div class="hash">호흡기내과 · 면허 #12345</div>
        </div>
        <div>
          <div class="label">전자 서명일</div>
          <div class="line date-sig">2026-04-23 09:14 KST</div>
          <div class="hash">HASH · SHA256:a4f9…7c2b</div>
        </div>
      </div>
    </div>
    <div class="footer">
      <div class="warn">⚠ AI-Assisted · Final diagnosis requires physician review</div>
      <div class="disc">본 리포트의 AI 분석 결과는 진단 보조용이며 최종 진단 및 치료 결정은 주치의의 임상적 판단에 따릅니다. [EU AI Act Art. 22]</div>
    </div>
  </div>
${patient.finalReport ? buildAiPageHtml(patient.finalReport) : ''}
</body>
</html>`);
  w.document.close();
}

/* ---- AIRagReport · Bedrock Phase 5 결과 렌더링 (React 인라인 컴포넌트) ---- */
function AIRagReport({ finalReport }) {
  if (!finalReport) return null;
  const data = finalReport.diagnosis_json || finalReport;
  const rec = data.recommendation || {};
  const notes = data.clinical_notes || {};
  const conf = data.confidence_metrics || {};
  const score = conf.overall_confidence_score || 0;
  const scoreColor = score >= 0.8 ? 'var(--rl-teal)' : score >= 0.6 ? 'var(--rl-amber)' : 'var(--rl-critical)';
  const scoreLabel = score >= 0.8 ? 'High' : score >= 0.6 ? 'Medium' : 'Low';
  const suf = conf.data_sufficiency || {};
  const sufColor = (v) => v === 'High' ? 'var(--rl-teal)' : v === 'Medium' ? 'var(--rl-amber)' : 'var(--rl-ink-3)';
  const sufLabel = { genomic_evidence: '유전체 근거', clinical_case_match: '임상케이스', trial_availability: '임상시험' };

  // MDT 필요 여부: specialist_referral에 MDT 키워드 있으면 true
  const hasMDT = (rec.specialist_referral || []).some(s => /MDT|multidisciplinary|다학제/i.test(s));

  // PubCaseFinder 실패 여부
  const pcfFailed = finalReport.self_check?.pcf_failed
    || (notes.rag_evidence || '').includes('PubCaseFinder 수집 실패');

  // PMID 텍스트에서 추출해 PubMed 링크로 변환
  const linkifyPMID = (text) => {
    if (!text) return text;
    const parts = text.split(/(PMID[:\s]*\d{7,9})/g);
    return parts.map((part, i) => {
      const m = part.match(/PMID[:\s]*(\d{7,9})/);
      if (m) {
        return (
          <a key={i} href={`https://pubmed.ncbi.nlm.nih.gov/${m[1]}/`} target="_blank" rel="noreferrer"
            style={{ color: 'var(--rl-primary)', fontFamily: 'monospace', fontSize: 9,
              background: 'rgba(12,68,124,0.07)', padding: '1px 4px', borderRadius: 3,
              textDecoration: 'none', marginLeft: 2 }}>
            PMID:{m[1]}
          </a>
        );
      }
      return part;
    });
  };

  // NCT ID를 ClinicalTrials.gov 링크로 변환
  const linkifyNCT = (text) => {
    if (!text) return text;
    const parts = text.split(/(NCT\d{8})/g);
    return parts.map((part, i) => {
      if (/^NCT\d{8}$/.test(part)) {
        return (
          <a key={i} href={`https://clinicaltrials.gov/study/${part}`} target="_blank" rel="noreferrer"
            style={{ color: 'var(--rl-teal)', fontFamily: 'monospace', fontSize: 9,
              background: 'rgba(14,133,116,0.08)', padding: '1px 4px', borderRadius: 3,
              textDecoration: 'none', marginLeft: 2 }}>
            {part}
          </a>
        );
      }
      return part;
    });
  };

  return (
    <div className="mx-auto mb-3" style={{ maxWidth: 540, background: 'white', boxShadow: '0 4px 16px rgba(10,22,40,0.12)', border: '1px solid var(--rl-border-soft)', fontSize: 11, color: 'var(--rl-ink)' }}>

      {/* ── 헤더 ── */}
      <div style={{ background: '#0A1628', color: 'white', padding: '8px 14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div className="font-mono text-[8px] uppercase tracking-widest" style={{ opacity: 0.5, marginBottom: 2 }}>Rare-Link · RAG Phase 5 · Bedrock</div>
          <div className="font-serif" style={{ fontSize: 13 }}>AI 진단 보조 리포트</div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div className="font-mono text-[8px]" style={{ opacity: 0.5 }}>
            {(finalReport.llm_model || 'Claude 3.5 Sonnet').replace('anthropic.', '').replace(/-v\d+:\d+$/, '')}
          </div>
          {finalReport.generated_at && (
            <div className="font-mono text-[8px]" style={{ opacity: 0.5, marginTop: 1 }}>
              {new Date(finalReport.generated_at).toLocaleString('ko-KR', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })} KST
            </div>
          )}
        </div>
      </div>

      {/* ── MDT 필요 배너 ── */}
      {hasMDT && (
        <div style={{ background: '#FFF7ED', borderBottom: '2px solid #B45309', padding: '6px 14px', display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 14 }}>⚠️</span>
          <div>
            <div className="font-mono text-[9px] font-semibold uppercase tracking-widest" style={{ color: '#B45309' }}>MDT 의뢰 필요</div>
            <div className="text-[10px]" style={{ color: '#92400E' }}>희귀질환 후보 포함 — 다학제팀 협진 권고 (Orphanet 지침 기반)</div>
          </div>
        </div>
      )}

      {/* ── PubCaseFinder 실패 경고 ── */}
      {pcfFailed && (
        <div style={{ background: '#FEF3C7', borderBottom: '1px solid #FCD34D', padding: '5px 14px', display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 12, flexShrink: 0 }}>⚠</span>
          <div className="text-[10px]" style={{ color: '#92400E' }}>
            <b>PubCaseFinder 데이터 수집 실패</b> — 글로벌 HPO 매칭 교차검증 미수행.
            Orphanet·Monarch·PubMed·ClinicalTrials 데이터로만 분석됨.
          </div>
        </div>
      )}

      <div style={{ padding: '10px 14px' }}>

        {/* ── 신뢰도 + 데이터 충분도 ── */}
        <div style={{ display: 'flex', gap: 10, marginBottom: 10, padding: '8px 10px', background: 'var(--rl-bg-2)', border: '1px solid var(--rl-border-soft)', borderRadius: 4 }}>
          <div style={{ flex: 1 }}>
            <div className="font-mono text-[8px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)', marginBottom: 5 }}>종합 신뢰도</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <div style={{ flex: 1, height: 6, background: 'var(--rl-border)', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{ width: `${score * 100}%`, height: '100%', background: scoreColor, borderRadius: 3 }} />
              </div>
              <span style={{ fontFamily: 'monospace', fontSize: 12, fontWeight: 700, color: scoreColor }}>{(score * 100).toFixed(0)}%</span>
              <span className="font-mono text-[9px]" style={{ color: scoreColor, background: `${scoreColor}18`, padding: '1px 5px', borderRadius: 10 }}>{scoreLabel}</span>
            </div>
            {conf.rationale && (
              <div className="text-[10px] leading-relaxed" style={{ color: 'var(--rl-ink-3)', marginTop: 4, fontStyle: 'italic' }}>{conf.rationale}</div>
            )}
          </div>
          {Object.keys(suf).length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5, flexShrink: 0, justifyContent: 'center' }}>
              {Object.entries(suf).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <div style={{ width: 7, height: 7, borderRadius: '50%', background: sufColor(v), flexShrink: 0 }} />
                  <span className="font-mono text-[9px]" style={{ color: 'var(--rl-ink-3)', minWidth: 56 }}>{sufLabel[k] || k}</span>
                  <span className="font-mono text-[9px] font-semibold" style={{ color: sufColor(v) }}>{v}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── 임상 종합 소견 ── */}
        {(notes.summary || notes.top1_reasoning || notes.differential_note || notes.epidemiology_note) && (
          <RagSection title="임상 종합 소견" color="var(--rl-primary)">
            {notes.summary && (
              <RagBlock label="요약">
                <div className="text-[11px] leading-relaxed">{notes.summary}</div>
              </RagBlock>
            )}
            {notes.top1_reasoning && (
              <RagBlock label="Top 1 근거">
                <div className="text-[10px] leading-relaxed">{linkifyPMID(notes.top1_reasoning)}</div>
              </RagBlock>
            )}
            {notes.differential_note && (
              <RagBlock label="감별 진단 (Top 2·3 배제)">
                <div className="text-[10px] leading-relaxed" style={{ color: 'var(--rl-ink-2)' }}>{linkifyPMID(notes.differential_note)}</div>
              </RagBlock>
            )}
            {notes.epidemiology_note && (
              <RagBlock label="희귀질환 역학" labelColor="var(--rl-rare)">
                <div className="text-[10px] leading-relaxed" style={{ color: '#6B21A8' }}>{notes.epidemiology_note}</div>
              </RagBlock>
            )}
          </RagSection>
        )}

        {/* ── 즉각 검사 권고 ── */}
        {rec.immediate_workup?.length > 0 && (
          <RagSection title="즉각 검사 권고" color="var(--rl-primary)">
            <RagChecklist items={rec.immediate_workup} />
          </RagSection>
        )}

        {/* ── 전문과 의뢰 / MDT ── */}
        {rec.specialist_referral?.length > 0 && (
          <RagSection title={hasMDT ? '전문과 의뢰 · MDT 필수' : '전문과 의뢰'} color="var(--rl-amber)">
            <RagChecklist items={rec.specialist_referral} color="#B45309" bullet="→" />
          </RagSection>
        )}

        {/* ── 치료 가이드라인 ── */}
        {rec.treatment_guideline?.length > 0 && (
          <RagSection title="치료 가이드라인" color="var(--rl-primary)">
            <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none' }}>
              {rec.treatment_guideline.map((item, i) => (
                <li key={i} className="text-[10px] leading-relaxed" style={{ marginBottom: 5, paddingLeft: 12, position: 'relative' }}>
                  <span style={{ position: 'absolute', left: 0, color: 'var(--rl-primary)', fontWeight: 700 }}>·</span>
                  {linkifyPMID(item)}
                </li>
              ))}
            </ul>
          </RagSection>
        )}

        {/* ── 유전자 검사 ── */}
        {rec.genetic_test?.length > 0 && (
          <RagSection title="유전자 검사 권고" color="var(--rl-rare)">
            <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none' }}>
              {rec.genetic_test.map((item, i) => (
                <li key={i} className="text-[10px] leading-relaxed" style={{ marginBottom: 5, paddingLeft: 12, position: 'relative', color: '#6B21A8' }}>
                  <span style={{ position: 'absolute', left: 0, fontWeight: 700 }}>·</span>
                  {item}
                </li>
              ))}
            </ul>
          </RagSection>
        )}

        {/* ── 추가 Lab ── */}
        {rec.additional_lab?.length > 0 && (
          <RagSection title="추가 Lab 권고" color="var(--rl-ink-3)">
            <RagChecklist items={rec.additional_lab} color="var(--rl-ink-2)" />
          </RagSection>
        )}

        {/* ── 임상시험 ── */}
        {rec.clinical_trial_info?.length > 0 && (
          <RagSection title="모집 중 임상시험" color="var(--rl-teal)">
            <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none' }}>
              {rec.clinical_trial_info.map((item, i) => (
                <li key={i} className="text-[10px] leading-relaxed" style={{ marginBottom: 5, paddingLeft: 12, position: 'relative' }}>
                  <span style={{ position: 'absolute', left: 0, color: 'var(--rl-teal)', fontWeight: 700 }}>·</span>
                  {linkifyNCT(item)}
                </li>
              ))}
            </ul>
          </RagSection>
        )}

        {/* ── RAG 근거 분석 ── */}
        {(notes.rag_evidence || notes.case_comparison) && (
          <RagSection title="RAG 근거 분석 (DB · API 교차검증)" color="var(--rl-ink-3)">
            {notes.rag_evidence && (
              <RagBlock label="DB·API 교차검증">
                <div className="text-[10px] leading-relaxed" style={{ color: 'var(--rl-ink-2)' }}>{linkifyPMID(notes.rag_evidence)}</div>
              </RagBlock>
            )}
            {notes.case_comparison && (
              <RagBlock label="PubMed 케이스 비교">
                <div className="text-[10px] leading-relaxed" style={{ color: 'var(--rl-ink-2)' }}>{linkifyPMID(notes.case_comparison)}</div>
              </RagBlock>
            )}
          </RagSection>
        )}

        {/* ── 면책 고지 ── */}
        <div style={{ marginTop: 8, padding: '7px 10px', background: 'rgba(180,83,9,0.05)', border: '1px solid rgba(180,83,9,0.2)', borderRadius: 4 }}>
          <div className="font-mono text-[8px] uppercase tracking-widest" style={{ color: 'var(--rl-amber)', marginBottom: 3 }}>
            ⚠ AI-Assisted · Human-in-the-Loop Required · EU AI Act Art. 22
          </div>
          <div className="text-[10px] leading-relaxed" style={{ color: 'var(--rl-ink-3)' }}>
            {notes.disclaimer || 'AI 결과는 진단 보조이며 최종 진단은 주치의의 임상 판단과 추가 검사 결과를 종합하여 확정합니다.'}
          </div>
          {finalReport.rag_citations && (
            <div className="font-mono text-[8px] mt-2" style={{ color: 'var(--rl-ink-3)' }}>
              유효 PMID {finalReport.rag_citations.pmid_valid?.length || 0}건
              {finalReport.rag_citations.pmid_invalid?.length > 0 && (
                <span style={{ color: 'var(--rl-critical)', marginLeft: 6 }}>
                  · 환각 PMID {finalReport.rag_citations.pmid_invalid.length}건 검출
                </span>
              )}
              {' · '}RAG APIs: {(finalReport.rag_apis_used || []).join(', ')}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* AIRagReport 내부 서브 컴포넌트 */
function RagSection({ title, color, children }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div className="font-mono text-[9px] uppercase tracking-widest"
        style={{ color: color || 'var(--rl-primary)', borderBottom: `1px solid ${color || 'var(--rl-primary)'}30`,
          paddingBottom: 3, marginBottom: 6, letterSpacing: '0.08em' }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function RagBlock({ label, labelColor, children }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div className="font-mono text-[8px] uppercase tracking-widest"
        style={{ color: labelColor || 'var(--rl-ink-3)', marginBottom: 2 }}>
        {label}
      </div>
      {children}
    </div>
  );
}

function RagChecklist({ items, color, bullet = '·' }) {
  if (!items?.length) return null;
  return (
    <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none' }}>
      {items.map((item, i) => (
        <li key={i} className="text-[10px] leading-relaxed"
          style={{ marginBottom: 4, paddingLeft: 12, position: 'relative', color: color || 'var(--rl-ink)' }}>
          <span style={{ position: 'absolute', left: 0, fontWeight: 700, color: color || 'var(--rl-primary)' }}>{bullet}</span>
          {item}
        </li>
      ))}
    </ul>
  );
}

/* 하위 호환 — 기존 코드가 참조하는 경우 대비 */
function AIRagSection({ title, children }) {
  return <RagSection title={title}>{children}</RagSection>;
}
function AIRagItem({ label, value, accent }) {
  return <RagBlock label={label} labelColor={accent}><div className="text-[10px] leading-relaxed">{value}</div></RagBlock>;
}
function AIRagList({ label, items, accent }) {
  if (!items?.length) return null;
  return <RagBlock label={label} labelColor={accent}><RagChecklist items={items} color={accent} /></RagBlock>;
}

/* ----------- TAB · HISTORY (과거 방문 + 클릭 새 창 상세) ----------- */
const DEFAULT_HISTORY = [
  {
    date: '2026-04-09', visit: '재진',
    complaint: '기침 악화 · 운동 시 호흡곤란 (mMRC 2)',
    dx: '특발성 폐섬유증 (IPF) · stable',
    tx: 'Pirfenidone 801mg tid 유지 · NAC 600mg bid',
    physician: '정민수 과장',
    vitals: 'BP 128/76 · HR 84 · RR 18 · SpO₂ 95% (RA)',
    notes: 'HRCT 변화 없음. 6분 보행거리 412m (+8m vs 직전). FVC 2.84L (68% predicted, +1%). 약물 부작용 호소 없음.',
  },
  {
    date: '2026-03-12', visit: '재진',
    complaint: 'FU · 피로감 호소',
    dx: 'IPF · 안정적 · 약물 적응 양호',
    tx: 'Pirfenidone 유지 · NAC 추가 처방',
    physician: '정민수 과장',
    vitals: 'BP 132/80 · HR 88 · RR 18 · SpO₂ 94% (RA)',
    notes: 'FVC 2.81L (67% pred). 식욕부진은 자연 호전. NAC 600mg bid 추가하여 산화 스트레스 감소 시도.',
  },
  {
    date: '2026-02-05', visit: '재진',
    complaint: '저용량 적응 · 1개월 평가',
    dx: 'IPF · 약물 적응 양호',
    tx: 'Pirfenidone 267mg tid → 534mg tid 증량',
    physician: '정민수 과장',
    vitals: 'BP 130/78 · HR 82 · RR 18 · SpO₂ 95% (RA)',
    notes: '경미한 오심 외 특이사항 없음. AST/ALT 정상 범위. 권고대로 단계적 증량 시작.',
  },
  {
    date: '2026-01-15', visit: '초진',
    complaint: '호흡곤란 6개월 · 마른기침 · 체중감소 4kg',
    dx: 'IPF (UIP pattern HRCT 확진)',
    tx: 'Pirfenidone 267mg tid 시작 · 폐 재활 의뢰',
    physician: '정민수 과장',
    vitals: 'BP 134/82 · HR 90 · RR 20 · SpO₂ 93% (RA)',
    notes: 'HRCT: 양측 하부 honeycombing + traction bronchiectasis · UIP pattern definite. PFT: restrictive (FVC 2.78L, 66% pred · DLCO 52% pred). MDT 결과 IPF 확진. Anti-fibrotic 시작.',
  },
  {
    date: '2025-12-08', visit: '의뢰',
    complaint: 'CXR 이상 → 호흡기내과 의뢰 (1차의원)',
    dx: 'ILD 의심 (HRCT 권고)',
    tx: 'HRCT 예약 · 폐기능검사 처방',
    physician: '김재현 (의뢰)',
    vitals: 'BP 128/76 · HR 86 · RR 18 · SpO₂ 95% (RA)',
    notes: '직장 건강검진 CXR에서 양측 하부 reticular opacity 발견되어 호흡기내과 의뢰됨. HRCT + PFT 예약.',
  },
];

function ChartHistory({ patient }) {
  const history = patient.history || DEFAULT_HISTORY;
  return (
    <div className="h-full fade-in">
      <Panel
        title="진단 히스토리"
        mono={`${history.length} visits · 클릭 → 상세`}
        fill
      >
        {/* Header row */}
        <div
          className="grid items-baseline font-mono text-[9px] uppercase tracking-widest pb-1.5 px-2"
          style={{
            gridTemplateColumns: HISTORY_GRID,
            columnGap: 8,
            color: 'var(--rl-ink-4)',
            borderBottom: '1px solid var(--rl-border-soft)',
          }}
        >
          <div>방문일</div>
          <div>유형</div>
          <div>주호소</div>
          <div>진단</div>
          <div>처방</div>
          <div></div>
        </div>
        <div>
          {history.map((h, i) => (
            <HistoryRow
              key={i}
              h={h}
              onClick={() => openHistoryPopup(patient, h)}
              isLast={i === history.length - 1}
            />
          ))}
        </div>
      </Panel>
    </div>
  );
}

const HISTORY_GRID = '90px 50px minmax(0, 1.4fr) minmax(0, 1.2fr) minmax(0, 1.2fr) 16px';

function HistoryRow({ h, onClick, isLast }) {
  return (
    <div
      onClick={onClick}
      className="grid items-baseline px-2 py-2 row-hover transition cursor-pointer"
      style={{
        gridTemplateColumns: HISTORY_GRID,
        columnGap: 8,
        borderBottom: isLast ? 'none' : '1px solid var(--rl-border-soft)',
      }}
    >
      <div className="font-mono text-xs" style={{ color: 'var(--rl-ink)' }}>{h.date}</div>
      <div className="font-mono text-[10px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)' }}>{h.visit}</div>
      <div className="text-xs truncate" style={{ color: 'var(--rl-ink)' }}>{h.complaint}</div>
      <div className="text-xs truncate" style={{ color: 'var(--rl-primary)', fontWeight: 500 }}>{h.dx}</div>
      <div className="text-xs truncate" style={{ color: 'var(--rl-ink-2)' }}>{h.tx}</div>
      <ChevronRight size={12} style={{ color: 'var(--rl-ink-3)' }} />
    </div>
  );
}

/* History 상세 popup · 새 창 */
function openHistoryPopup(patient, h) {
  const w = window.open('', `hx-${patient.mrn}-${h.date}`, 'width=720,height=820,resizable=yes,scrollbars=yes');
  if (!w) {
    alert('팝업 차단을 해제해주세요.');
    return;
  }
  const sex = patient.sex === 'M' ? '남' : '여';
  w.document.write(`<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<title>방문 상세 · ${patient.name} · ${h.date}</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Serif:wght@500&family=IBM+Plex+Mono&display=swap" rel="stylesheet" />
<style>
  html, body { margin: 0; padding: 0; }
  body { background: #F8FAFC; font-family: 'IBM Plex Sans KR', sans-serif; color: #0A1628; padding: 24px; -webkit-font-smoothing: antialiased; }
  .card { background: white; max-width: 640px; margin: 0 auto; border: 1px solid #E2E8F0; border-radius: 6px; padding: 24px; }
  .header { display: flex; align-items: baseline; gap: 12px; padding-bottom: 12px; border-bottom: 1px solid #E2E8F0; }
  .header .name { font-family: 'IBM Plex Serif', serif; font-size: 20px; }
  .header .meta { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #64748B; }
  .header .label { margin-left: auto; font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em; color: #0C447C; }
  .visit { display: flex; align-items: baseline; gap: 12px; margin: 16px 0; padding: 10px 12px; background: #EFF4FB; border-left: 3px solid #0C447C; }
  .visit .date { font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 600; }
  .visit .type { font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em; color: #64748B; }
  .visit .by { margin-left: auto; font-size: 11px; color: #334155; }
  .section { margin-bottom: 14px; }
  .section .title { font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em; color: #64748B; margin-bottom: 4px; }
  .section .body { font-size: 13px; line-height: 1.6; }
  .section.dx .body { color: #0C447C; font-weight: 500; }
  .section.tx .body { color: #0E8574; font-weight: 500; }
  .vitals { font-family: 'IBM Plex Mono', monospace; font-size: 12px; padding: 8px 10px; background: #F1F5F9; border-radius: 4px; }
  .footer { margin-top: 18px; padding-top: 12px; border-top: 1px solid #E2E8F0; font-family: 'IBM Plex Mono', monospace; font-size: 9px; text-transform: uppercase; letter-spacing: 0.15em; color: #64748B; display: flex; justify-content: space-between; }
</style>
</head>
<body>
  <div class="card">
    <div class="header">
      <span class="name">${patient.name}</span>
      <span class="meta">${sex} · ${patient.age}세 · ${patient.mrn}</span>
      <span class="label">방문 상세</span>
    </div>
    <div class="visit">
      <span class="date">${h.date}</span>
      <span class="type">${h.visit}</span>
      <span class="by">담당 · ${h.physician || '정민수 과장'}</span>
    </div>

    <div class="section"><div class="title">주호소 · Chief Complaint</div><div class="body">${h.complaint}</div></div>

    ${h.vitals ? `<div class="section"><div class="title">활력 징후 · Vitals</div><div class="vitals">${h.vitals}</div></div>` : ''}

    <div class="section dx"><div class="title">진단 · Diagnosis</div><div class="body">${h.dx}</div></div>

    <div class="section tx"><div class="title">처방 · Treatment</div><div class="body">${h.tx}</div></div>

    ${h.notes ? `<div class="section"><div class="title">진료 메모 · Notes</div><div class="body">${h.notes}</div></div>` : ''}

    <div class="footer">
      <span>Rare-Link AI · Visit Detail</span>
      <span>EU AI Act Art. 22</span>
    </div>
  </div>
</body>
</html>`);
  w.document.close();
}

function InfoCell({ label, value, mono, compact }) {
  const valueClass = compact ? 'text-xs' : 'text-sm';
  const labelClass = compact ? 'text-[9px]' : 'text-[10px]';
  return (
    <div style={{ minWidth: 0 }}>
      <div
        className={`font-mono ${labelClass} uppercase tracking-widest mb-0.5 truncate`}
        style={{ color: 'var(--rl-ink-3)' }}
      >
        {label}
      </div>
      <div
        className={`${valueClass} ${mono ? 'font-mono' : ''} truncate`}
        style={{ color: 'var(--rl-ink)', whiteSpace: 'nowrap' }}
        title={typeof value === 'string' ? value : undefined}
      >
        {value}
      </div>
    </div>
  );
}

/* SVG markup builder · 원본 / heatmap 두 종류 동적 생성 */
const CXR_RIB_YS = [50, 65, 80, 95, 110, 125, 140];

function buildCxrSvg({ heatmap = false } = {}) {
  return `<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet" style="width:100%;height:100%;display:block;">
  <defs>
    <radialGradient id="lung-${heatmap ? 'h' : 'o'}" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#2C3E50" />
      <stop offset="100%" stop-color="#0A1628" />
    </radialGradient>
    ${heatmap ? `
    <radialGradient id="hot-l" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="rgba(255,80,40,0.85)" />
      <stop offset="55%" stop-color="rgba(255,160,30,0.45)" />
      <stop offset="100%" stop-color="rgba(255,200,40,0)" />
    </radialGradient>
    <radialGradient id="hot-r" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="rgba(255,90,60,0.7)" />
      <stop offset="60%" stop-color="rgba(255,170,40,0.35)" />
      <stop offset="100%" stop-color="rgba(255,200,40,0)" />
    </radialGradient>` : ''}
  </defs>
  <rect x="0" y="0" width="200" height="200" fill="url(#lung-${heatmap ? 'h' : 'o'})" />
  <line x1="100" y1="20" x2="100" y2="180" stroke="rgba(255,255,255,0.35)" stroke-width="3" />
  ${CXR_RIB_YS.map(y => `<path d="M 100 ${y} Q 50 ${y+15}, 25 ${y+10}" stroke="rgba(255,255,255,0.2)" stroke-width="1" fill="none" />`).join('')}
  ${CXR_RIB_YS.map(y => `<path d="M 100 ${y} Q 150 ${y+15}, 175 ${y+10}" stroke="rgba(255,255,255,0.2)" stroke-width="1" fill="none" />`).join('')}
  <ellipse cx="108" cy="115" rx="22" ry="30" fill="rgba(255,255,255,0.08)" />
  <path d="M 20 160 Q 60 150, 98 158 Q 140 165, 180 155" stroke="rgba(255,255,255,0.25)" stroke-width="1.2" fill="none" />
  ${heatmap ? `
  <ellipse cx="65" cy="130" rx="28" ry="22" fill="url(#hot-l)" />
  <ellipse cx="140" cy="128" rx="22" ry="18" fill="url(#hot-r)" />
  <line x1="0" y1="95" x2="200" y2="95" stroke="rgba(77,212,245,0.3)" stroke-width="0.5" stroke-dasharray="2 2" />
  <text x="145" y="14" fill="rgba(255,180,60,0.95)" font-size="6" font-family="monospace">HEATMAP · Grad-CAM</text>
  ` : ''}
  <text x="8" y="14" fill="rgba(255,255,255,0.55)" font-size="6" font-family="monospace">CXR · Frontal</text>
  <text x="8" y="193" fill="rgba(255,255,255,0.4)" font-size="6" font-family="monospace">448 × 448 · resized</text>
</svg>`;
}

function CXRMock({ heatmap = false }) {
  return (
    <div
      style={{ width: '100%', height: '100%' }}
      dangerouslySetInnerHTML={{ __html: buildCxrSvg({ heatmap }) }}
    />
  );
}

/* CXR 확대 보기 · 새 창 (window.open popup) · 좌우 비교 (원본 vs Heatmap) */
function openCxrPopup(patient) {
  const w = window.open('', `cxr-${patient.mrn}`, 'width=1280,height=820,resizable=yes,scrollbars=no');
  if (!w) {
    alert('팝업 차단을 해제해주세요.');
    return;
  }
  const sex = patient.sex === 'M' ? '남' : '여';
  const date = patient.visitDate || '오늘 (2026-04-23)';
  const meta = `${sex} · ${patient.age}세 · ${patient.mrn} · ${date} ${patient.time}`;
  w.document.write(`<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<title>CXR · ${patient.name} · ${patient.mrn} · 비교</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Serif:wght@500&family=IBM+Plex+Mono&display=swap" rel="stylesheet" />
<style>
  :root { color-scheme: dark; }
  html, body { margin: 0; padding: 0; height: 100%; }
  body {
    background: #0A1628; color: #fff;
    font-family: 'IBM Plex Sans KR', sans-serif;
    display: flex; flex-direction: column; height: 100vh;
    -webkit-font-smoothing: antialiased;
  }
  header {
    padding: 14px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    display: flex; align-items: baseline; gap: 12px;
  }
  header .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em;
    color: #4DD4F5;
  }
  header .name { font-family: 'IBM Plex Serif', serif; font-size: 20px; letter-spacing: -0.01em; }
  header .meta { font-family: 'IBM Plex Mono', monospace; font-size: 11px; opacity: 0.6; }
  header .spacer { flex: 1; }
  main {
    flex: 1; min-height: 0;
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
    padding: 16px 20px;
  }
  .pane { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
  .pane .caption {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em;
    margin-bottom: 6px; display: flex; align-items: center; gap: 8px;
  }
  .pane.original .caption { color: rgba(255,255,255,0.7); }
  .pane.heatmap  .caption { color: rgba(255,180,60,0.95); }
  .pane .caption .dot { width: 6px; height: 6px; border-radius: 50%; }
  .pane.original .dot { background: rgba(255,255,255,0.7); }
  .pane.heatmap  .dot { background: rgba(255,180,60,0.95); }
  .pane .frame {
    flex: 1; min-height: 0;
    display: flex; align-items: center; justify-content: center;
  }
  .pane .frame > div {
    height: 100%; aspect-ratio: 1 / 1; max-width: 100%;
    border: 1px solid rgba(255,255,255,0.18); border-radius: 4px;
    overflow: hidden;
  }
  .pane.heatmap .frame > div { border-color: rgba(255,180,60,0.4); }
  footer {
    padding: 8px 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; opacity: 0.55;
    border-top: 1px solid rgba(255,255,255,0.1);
    text-align: center;
  }
</style>
</head>
<body>
  <header>
    <span class="label">CXR · Frontal · 비교 보기</span>
    <span class="spacer"></span>
    <span class="name">${patient.name}</span>
    <span class="meta">${meta}</span>
  </header>
  <main>
    <div class="pane original">
      <div class="caption"><span class="dot"></span>원본 · Original</div>
      <div class="frame"><div>${buildCxrSvg({ heatmap: false })}</div></div>
    </div>
    <div class="pane heatmap">
      <div class="caption"><span class="dot"></span>AI Heatmap · Grad-CAM overlay</div>
      <div class="frame"><div>${buildCxrSvg({ heatmap: true })}</div></div>
    </div>
  </main>
  <footer>Rare-Link AI · 본 영상은 진단 보조용입니다 · EU AI Act Art. 22</footer>
</body>
</html>`);
  w.document.close();
}

function DxPreviewRow({ rank, name, prob, rare, dontMiss, orpha, onClick }) {
  return (
    <div
      onClick={onClick}
      className={`hairline-strong rounded px-3 py-2.5 transition ${onClick ? 'cursor-pointer hover:bg-slate-50' : ''}`}
      style={{ borderLeft: dontMiss ? '3px solid var(--rl-amber)' : undefined }}
      title={onClick ? '계산 근거 보기' : undefined}
    >
      <div className="flex items-baseline gap-2 mb-1.5">
        <div className="font-mono text-[10px]" style={{ color: 'var(--rl-ink-3)' }}>#{rank}</div>
        <div className="text-sm font-medium flex-1 min-w-0 truncate" style={{ color: 'var(--rl-ink)' }}>{name}</div>
        <div className="font-serif text-base leading-none" style={{ color: 'var(--rl-primary)' }}>{(prob * 100).toFixed(0)}<span className="text-[10px]">%</span></div>
        {onClick && <ArrowUpRight size={11} style={{ color: 'var(--rl-primary)' }} />}
      </div>
      <div className="h-1 rounded-full mb-1.5" style={{ background: 'var(--rl-bg-3)' }}>
        <div className="h-full rounded-full" style={{ width: prob * 100 + '%', background: 'var(--rl-primary)' }} />
      </div>
      <div className="flex items-center gap-1.5">
        {rare && (
          <span className="chip" style={{ background: 'var(--rl-rare-soft)', color: 'var(--rl-rare)' }}>
            <Flame size={9} /> 희귀
          </span>
        )}
        {dontMiss && (
          <span className="chip" style={{ background: 'var(--rl-amber-soft)', color: 'var(--rl-amber)' }}>
            <AlertTriangle size={9} /> Don't miss
          </span>
        )}
        {orpha && <span className="font-mono text-[10px]" style={{ color: 'var(--rl-ink-3)' }}>{orpha}</span>}
      </div>
    </div>
  );
}

/* ----------- DX EVIDENCE · LR 계산 근거 popup ----------- */
function buildDxEvidence(patient, dx) {
  // 환자 주호소에서 HPO term 추출 (mock)
  const c = patient.complaint || '';
  const candidate = [
    [/호흡곤란/, { id: 'HP:0002094', label: 'Dyspnea (호흡곤란)',          lr: 4.2 }],
    [/마른\s*기침|마른기침/, { id: 'HP:0031246', label: 'Nonproductive cough (마른기침)', lr: 3.6 }],
    [/기침/, { id: 'HP:0012735', label: 'Cough (기침)',                   lr: 2.1 }],
    [/체중감소/, { id: 'HP:0001824', label: 'Weight loss (체중감소)',     lr: 2.6 }],
    [/객혈/,    { id: 'HP:0002105', label: 'Hemoptysis (객혈)',           lr: 6.4 }],
    [/객담/,    { id: 'HP:0033709', label: 'Productive cough (객담)',     lr: 1.9 }],
    [/흉통/,    { id: 'HP:0100749', label: 'Chest pain (흉통)',           lr: 1.8 }],
    [/발한/,    { id: 'HP:0000989', label: 'Night sweats (야간 발한)',    lr: 3.2 }],
    [/기흉/,    { id: 'HP:0002107', label: 'Pneumothorax (기흉)',         lr: 8.5 }],
    [/관절염/,  { id: 'HP:0001370', label: 'Rheumatoid arthritis (RA)',   lr: 5.5 }],
    [/부종/,    { id: 'HP:0000969', label: 'Edema (부종)',                lr: 2.3 }],
    [/두근거림/, { id: 'HP:0001962', label: 'Palpitations (두근거림)',    lr: 1.6 }],
    [/면역결핍/, { id: 'HP:0002721', label: 'Immunodeficiency',           lr: 7.2 }],
  ];
  const observed = candidate.filter(([re]) => re.test(c)).map(([, t]) => ({ ...t, state: 'observed' }));
  // 보조 (미관찰) HPO 1-2개 추가 (대조용)
  const supplementary = [
    { id: 'HP:0030828', label: 'Velcro crackles (벨크로 수포음)', lr: 1.0, state: 'unknown' },
    { id: 'HP:0006510', label: 'Chronic pulmonary obstruction',   lr: 1.0, state: 'unknown' },
  ].slice(0, 2 - Math.min(2, observed.length === 0 ? 2 : 0));
  const hpoTerms = [...observed, ...supplementary];

  // CXR DenseNet score
  const cxrScore = Math.min(0.95, dx.prob * 0.85 + 0.10);

  // Combined LR (관찰된 것 곱)
  const combinedHpoLr = observed.reduce((p, t) => p * t.lr, 1);
  // Prior odds (희귀질환은 낮음)
  const priorProb = dx.rare ? 0.0001 : 0.005;
  const priorOdds = priorProb / (1 - priorProb);
  // Posterior odds = prior × combined LR × CXR LR (CXR LR 근사 = score / (1-score))
  const cxrLr = (cxrScore / Math.max(0.05, 1 - cxrScore));
  const postOdds = priorOdds * combinedHpoLr * cxrLr;
  const postProb = postOdds / (1 + postOdds);

  // Refs (dx별)
  const refs = ['Robinson PN et al. Am J Hum Genet 2020;107:403-417 (LIRICAL · LR paradigm)'];
  if (/IPF|섬유증/.test(dx.name))     refs.push('Raghu G et al. ATS/ERS/JRS/ALAT IPF Guideline. Am J Respir Crit Care Med 2022;205:e18-e47');
  if (/Sarcoidosis/.test(dx.name))    refs.push('Crouser ED et al. ATS Sarcoidosis Guideline. Am J Respir Crit Care Med 2020;201:e26-e51');
  if (/LAM|Langerhans/.test(dx.name)) refs.push('Gupta N et al. ATS/JRS LAM Guideline. Am J Respir Crit Care Med 2017;196:1337-1348');
  if (/Pneumonia|폐렴/.test(dx.name)) refs.push('Metlay JP et al. ATS/IDSA CAP Guideline. Am J Respir Crit Care Med 2019;200:e45-e67');
  if (/CHF|Heart Failure/.test(dx.name)) refs.push('Heidenreich PA et al. AHA/ACC/HFSA HF Guideline. Circulation 2022;145:e895-e1032');
  if (/RA-associated|NSIP/.test(dx.name)) refs.push('Travis WD et al. ATS/ERS NSIP Statement. Am J Respir Crit Care Med 2008;177:1338-1347');

  return {
    prevalence: dx.rare ? '< 5 / 100,000' : '~ 50 / 100,000',
    priorProb, priorOdds,
    hpoTerms, observedCount: observed.length,
    cxrScore, cxrLr,
    combinedHpoLr,
    postOdds, postProb,
    refs,
  };
}

function openDxEvidencePopup(patient, dx, rank) {
  const w = window.open('', `dx-${patient.mrn}-${rank}`, 'width=900,height=900,resizable=yes,scrollbars=yes');
  if (!w) {
    alert('팝업 차단을 해제해주세요.');
    return;
  }
  const ev = buildDxEvidence(patient, dx);
  const sex = patient.sex === 'M' ? '남' : '여';

  const hpoRows = ev.hpoTerms.map(t => `
    <tr class="${t.state === 'observed' ? '' : 'muted-row'}">
      <td class="mono">${t.id}</td>
      <td>${t.label}</td>
      <td>${t.state === 'observed'
        ? '<span class="chip teal">관찰</span>'
        : '<span class="chip muted">미관찰</span>'}</td>
      <td class="right mono"><b>${t.lr.toFixed(1)}</b></td>
      <td class="right mono muted">log₁₀ ${Math.log10(t.lr).toFixed(2)}</td>
    </tr>`).join('');

  const refList = ev.refs.map(r => `<li>${r}</li>`).join('');

  const flagBadges = [
    dx.dontMiss ? '<span class="chip amber">Don\'t miss</span>' : '',
    dx.rare     ? '<span class="chip rare">희귀질환</span>' : '',
    dx.orpha    ? `<span class="mono small muted">${dx.orpha}</span>` : '',
  ].filter(Boolean).join(' ');

  w.document.write(`<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<title>${dx.name} · 계산 근거</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Serif:wght@500&family=IBM+Plex+Mono&display=swap" rel="stylesheet" />
<style>
  html, body { margin: 0; padding: 0; }
  body { background: #F8FAFC; font-family: 'IBM Plex Sans KR', sans-serif; color: #0A1628; padding: 24px; -webkit-font-smoothing: antialiased; }
  .card { background: white; max-width: 760px; margin: 0 auto; border: 1px solid #E2E8F0; border-radius: 6px; padding: 28px; }
  h1 { font-family: 'IBM Plex Serif', serif; font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
  .meta { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #64748B; margin-bottom: 12px; }
  .header-strip { display: flex; align-items: baseline; gap: 12px; padding-bottom: 12px; border-bottom: 1px solid #E2E8F0; flex-wrap: wrap; }
  .header-strip .rank { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #0C447C; font-weight: 600; }
  .header-strip .post {
    margin-left: auto; font-family: 'IBM Plex Serif', serif; font-size: 28px; color: #0C447C;
  }
  .post .small { font-size: 12px; }
  .post .label { font-family: 'IBM Plex Mono', monospace; font-size: 9px; text-transform: uppercase; letter-spacing: 0.15em; color: #64748B; display: block; text-align: right; }
  .section { margin-top: 18px; }
  .section .title { font-family: 'IBM Plex Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.15em; color: #0C447C; margin-bottom: 6px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .stat { padding: 8px 10px; background: #F1F5F9; border-radius: 4px; }
  .stat .l { font-family: 'IBM Plex Mono', monospace; font-size: 9px; text-transform: uppercase; letter-spacing: 0.15em; color: #64748B; }
  .stat .v { font-family: 'IBM Plex Mono', monospace; font-size: 14px; color: #0A1628; }
  .stat.warn { background: #FEF3C7; }
  .stat.warn .v { color: #B45309; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  table th { text-align: left; padding: 6px 8px; border-bottom: 1px solid #CBD5E1; font-family: 'IBM Plex Mono', monospace; font-size: 9px; text-transform: uppercase; letter-spacing: 0.15em; color: #64748B; background: #F8FAFC; }
  table td { padding: 6px 8px; border-bottom: 1px solid #E2E8F0; }
  table .right { text-align: right; }
  .mono { font-family: 'IBM Plex Mono', monospace; }
  .muted { color: #64748B; }
  .small { font-size: 10px; }
  .chip { display: inline-block; padding: 1px 6px; border-radius: 3px; font-family: 'IBM Plex Mono', monospace; font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em; }
  .chip.teal  { background: #E6F5F2; color: #0E8574; }
  .chip.amber { background: #FEF3C7; color: #B45309; }
  .chip.rare  { background: #F3E8FF; color: #6B21A8; }
  .chip.muted { background: #F1F5F9; color: #94A3B8; }
  tr.muted-row td:nth-child(2) { color: #94A3B8; }
  .bayes {
    font-family: 'IBM Plex Mono', monospace; font-size: 13px;
    background: #EFF4FB; padding: 12px; border-radius: 4px; line-height: 1.8;
    border-left: 3px solid #0C447C;
  }
  .bayes .op { color: #64748B; }
  .bayes .res { color: #0C447C; font-weight: 600; }
  ul.refs { font-size: 11px; line-height: 1.6; padding-left: 18px; margin: 0; color: #334155; }
  .footer { margin-top: 18px; padding-top: 12px; border-top: 1px solid #E2E8F0; font-family: 'IBM Plex Mono', monospace; font-size: 9px; text-transform: uppercase; letter-spacing: 0.15em; color: #B45309; }
  .footer .disc { color: #64748B; text-transform: none; letter-spacing: 0; font-family: 'IBM Plex Sans KR', sans-serif; font-size: 10px; margin-top: 3px; }
</style>
</head>
<body>
  <div class="card">
    <div class="header-strip">
      <span class="rank">Top #${rank}</span>
      <h1>${dx.name}</h1>
      <span style="display:flex;gap:6px;align-items:baseline;">${flagBadges}</span>
      <div class="post">
        <span class="label">Posterior probability</span>
        ${(ev.postProb * 100).toFixed(0)}<span class="small">%</span>
      </div>
    </div>
    <div class="meta">환자 · ${patient.name} · ${sex} · ${patient.age}세 · ${patient.mrn}</div>

    <div class="section">
      <div class="title">1. Prior · 모집단 prevalence</div>
      <div class="grid2">
        <div class="stat"><div class="l">Prevalence</div><div class="v">${ev.prevalence}</div></div>
        <div class="stat"><div class="l">Prior probability</div><div class="v">${(ev.priorProb * 100).toFixed(3)}%</div></div>
      </div>
    </div>

    <div class="section">
      <div class="title">2. HPO 기반 Likelihood Ratio (Robinson 2020)</div>
      <table>
        <thead><tr><th>HPO ID</th><th>증상</th><th>관찰</th><th class="right">LR+</th><th class="right">log₁₀ LR</th></tr></thead>
        <tbody>${hpoRows}</tbody>
      </table>
      <div class="muted small mono" style="margin-top:6px;">관찰된 ${ev.observedCount}개 term의 LR을 곱하여 종합 ⇒ ∏LR<sub>HPO</sub> = <b>${ev.combinedHpoLr.toFixed(2)}</b></div>
    </div>

    <div class="section">
      <div class="title">3. CXR DenseNet-121 기여</div>
      <div class="grid2">
        <div class="stat"><div class="l">Model output (CXR)</div><div class="v">${(ev.cxrScore * 100).toFixed(1)}%</div></div>
        <div class="stat"><div class="l">변환 LR<sub>CXR</sub></div><div class="v">${ev.cxrLr.toFixed(2)}</div></div>
      </div>
    </div>

    <div class="section">
      <div class="title">4. Bayes 종합 계산</div>
      <div class="bayes">
        Posterior odds = Prior odds <span class="op">×</span> ∏LR<sub>HPO</sub> <span class="op">×</span> LR<sub>CXR</sub><br/>
        = ${ev.priorOdds.toExponential(2)} <span class="op">×</span> ${ev.combinedHpoLr.toFixed(2)} <span class="op">×</span> ${ev.cxrLr.toFixed(2)}<br/>
        = <span class="res">${ev.postOdds.toFixed(3)}</span><br/>
        ⇒ Posterior probability = odds / (1 + odds) = <span class="res">${(ev.postProb * 100).toFixed(1)}%</span>
      </div>
    </div>

    <div class="section">
      <div class="title">5. 참고 문헌</div>
      <ul class="refs">${refList}</ul>
    </div>

    <div class="footer">
      ⚠ AI-Assisted · Final diagnosis requires physician review
      <div class="disc">본 계산 결과는 진단 보조용이며 최종 진단 및 치료 결정은 주치의의 임상적 판단에 따릅니다. [EU AI Act Art. 22]</div>
    </div>
  </div>
</body>
</html>`);
  w.document.close();
}

/* ============================================================
   SCREEN · SETTINGS
   ============================================================ */
function SettingsScreen({ doctor, onLogout, onNavigate, onOpenPatient, onOpenAnnouncement }) {
  const [prefs, setPrefs] = useState({
    notif:    { unread: true,  dontMiss: true, daily: false, sound: false },
    ai:       { defaultCxrView: 'original', topN: 3, lrBar: true, rareFirst: true, explanation: true },
    display:  { lang: 'ko', density: 'compact', theme: 'light', zoom: 80 },
    worklist: { defaultSection: 'today', autoRefresh: 30, sortBy: 'time' },
  });

  const setSlice = (key, val) => setPrefs(p => ({ ...p, [key]: { ...p[key], ...val } }));

  return (
    <div className="min-h-screen flex flex-col" style={{ background: 'var(--rl-bg-2)' }}>
      <TopBar doctor={doctor} onLogout={onLogout} activeScreen="settings" onNavigate={onNavigate} onOpenPatient={onOpenPatient} onOpenAnnouncement={onOpenAnnouncement} />

      <main className="flex-1 max-w-[1440px] w-full mx-auto px-8 py-6">
        {/* Header */}
        <div className="flex items-baseline gap-4 mb-5">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-widest mb-1" style={{ color: 'var(--rl-ink-3)' }}>
              Settings · v0.1.0
            </div>
            <h1 className="font-serif text-3xl" style={{ color: 'var(--rl-ink)', letterSpacing: '-0.01em' }}>
              설정
            </h1>
          </div>
          <button
            onClick={() => onNavigate('worklist')}
            className="ml-auto flex items-center gap-1 px-3 py-1.5 rounded text-xs hairline-strong hover:bg-slate-50"
            style={{ color: 'var(--rl-ink-2)' }}
          >
            <ChevronLeft size={12} /> 환자 목록으로
          </button>
        </div>

        {/* 8 cards in 3-column grid */}
        <div className="grid grid-cols-3 gap-3">
          <AccountCard       doctor={doctor} onLogout={onLogout} />
          <NotificationCard  prefs={prefs.notif}    set={(v) => setSlice('notif', v)} />
          <AICard            prefs={prefs.ai}       set={(v) => setSlice('ai', v)} />
          <DisplayCard       prefs={prefs.display}  set={(v) => setSlice('display', v)} />
          <WorklistPrefCard  prefs={prefs.worklist} set={(v) => setSlice('worklist', v)} />
          <SecurityCard      onLogout={onLogout} />
          <SystemCard        onNavigate={onNavigate} />
          <HelpCard />
        </div>
      </main>
    </div>
  );
}

/* ----- Setting card primitives ----- */
function SettingRow({ icon, label, sub, children }) {
  return (
    <div
      className="flex items-center gap-2.5 py-2"
      style={{ borderBottom: '1px solid var(--rl-border-soft)' }}
    >
      {icon && <span style={{ color: 'var(--rl-ink-3)', flexShrink: 0 }}>{icon}</span>}
      <div className="min-w-0 flex-1">
        <div className="text-xs" style={{ color: 'var(--rl-ink)' }}>{label}</div>
        {sub && <div className="text-[10px] mt-0.5 truncate" style={{ color: 'var(--rl-ink-3)' }}>{sub}</div>}
      </div>
      <div style={{ flexShrink: 0 }}>{children}</div>
    </div>
  );
}

function Toggle({ value, onChange }) {
  return (
    <button
      onClick={() => onChange(!value)}
      className="relative rounded-full transition"
      style={{
        width: 30, height: 16,
        background: value ? 'var(--rl-primary)' : 'var(--rl-border)',
      }}
    >
      <span
        className="absolute top-0.5 bg-white rounded-full transition shadow"
        style={{ width: 12, height: 12, left: value ? 16 : 2 }}
      />
    </button>
  );
}

function SettingSelect({ value, onChange, options }) {
  return (
    <select
      value={value}
      onChange={(e) => {
        const v = e.target.value;
        const opt = options.find(o => String(o.k) === v);
        onChange(opt ? opt.k : v);
      }}
      className="font-mono text-[11px] hairline-strong rounded px-2 py-1 outline-none focus:border-[color:var(--rl-primary)]"
      style={{ background: 'white', color: 'var(--rl-ink)' }}
    >
      {options.map(o => <option key={String(o.k)} value={String(o.k)}>{o.label}</option>)}
    </select>
  );
}

function ReadValue({ children, mono }) {
  return (
    <span
      className={`text-xs ${mono ? 'font-mono' : ''}`}
      style={{ color: 'var(--rl-ink-2)' }}
    >
      {children}
    </span>
  );
}

/* ----- 8 setting cards ----- */
function AccountCard({ doctor, onLogout }) {
  return (
    <Panel title="계정" mono="Account" right={<User size={12} style={{ color: 'var(--rl-ink-3)' }} />}>
      <SettingRow label="이름">           <ReadValue>{doctor.name}</ReadValue></SettingRow>
      <SettingRow label="직급">           <ReadValue>{doctor.role}</ReadValue></SettingRow>
      <SettingRow label="소속">           <ReadValue>{doctor.institution}</ReadValue></SettingRow>
      <SettingRow label="진료과">         <ReadValue>{doctor.department}</ReadValue></SettingRow>
      <SettingRow label="의사 ID">        <ReadValue mono>{doctor.id}</ReadValue></SettingRow>
      <SettingRow label="면허 번호">      <ReadValue mono>#12345</ReadValue></SettingRow>
      <button
        onClick={onLogout}
        className="w-full py-2 rounded text-xs font-medium hairline-strong flex items-center justify-center gap-1.5 mt-2 hover:bg-slate-50"
        style={{ color: 'var(--rl-critical)', borderColor: 'var(--rl-critical)' }}
      >
        <LogOut size={12} /> 로그아웃
      </button>
    </Panel>
  );
}

function NotificationCard({ prefs, set }) {
  return (
    <Panel title="알림" mono="Notifications" right={<Bell size={12} style={{ color: 'var(--rl-ink-3)' }} />}>
      <SettingRow icon={<Inbox size={13} />}     label="미확인 결과 알림"    sub="결과 도착 시 배지 표시"><Toggle value={prefs.unread}   onChange={(v) => set({ unread: v })} /></SettingRow>
      <SettingRow icon={<AlertTriangle size={13} />} label="Don't miss 긴급" sub="희귀·중증 의심 시 우선 알림"><Toggle value={prefs.dontMiss} onChange={(v) => set({ dontMiss: v })} /></SettingRow>
      <SettingRow icon={<Volume2 size={13} />}    label="알림음"             sub="긴급 알림에 한정"><Toggle value={prefs.sound}    onChange={(v) => set({ sound: v })} /></SettingRow>
      <SettingRow icon={<Mail size={13} />}        label="일일 다이제스트"    sub="매일 오전 7시 이메일"><Toggle value={prefs.daily}    onChange={(v) => set({ daily: v })} /></SettingRow>
    </Panel>
  );
}

function AICard({ prefs, set }) {
  return (
    <Panel title="AI 모델 · 표시" mono="AI Preferences" right={<Microscope size={12} style={{ color: 'var(--rl-ink-3)' }} />}>
      <SettingRow label="디폴트 CXR 뷰">
        <SettingSelect
          value={prefs.defaultCxrView}
          onChange={(v) => set({ defaultCxrView: v })}
          options={[
            { k: 'original', label: '원본' },
            { k: 'heatmap',  label: 'Heatmap' },
            { k: 'compare',  label: '비교' },
          ]}
        />
      </SettingRow>
      <SettingRow label="감별진단 표시 개수">
        <SettingSelect
          value={prefs.topN}
          onChange={(v) => set({ topN: Number(v) })}
          options={[
            { k: 3,  label: 'Top 3' },
            { k: 5,  label: 'Top 5' },
            { k: 10, label: 'Top 10' },
          ]}
        />
      </SettingRow>
      <SettingRow label="LR 막대 표시" sub="Robinson 2020 Fig.2 형식"><Toggle value={prefs.lrBar}       onChange={(v) => set({ lrBar: v })} /></SettingRow>
      <SettingRow label="희귀질환 우선 정렬"><Toggle value={prefs.rareFirst}   onChange={(v) => set({ rareFirst: v })} /></SettingRow>
      <SettingRow label="설명가능성 기본 ON" sub="Heatmap·LR 자동 표시 (Neri 2023)"><Toggle value={prefs.explanation} onChange={(v) => set({ explanation: v })} /></SettingRow>
    </Panel>
  );
}

function DisplayCard({ prefs, set }) {
  return (
    <Panel title="표시 · 언어" mono="Display" right={<Palette size={12} style={{ color: 'var(--rl-ink-3)' }} />}>
      <SettingRow icon={<Languages size={13} />} label="언어">
        <SettingSelect
          value={prefs.lang}
          onChange={(v) => set({ lang: v })}
          options={[
            { k: 'ko', label: '한국어' },
            { k: 'en', label: 'English (W4)' },
          ]}
        />
      </SettingRow>
      <SettingRow label="글자 / 화면 비율">
        <SettingSelect
          value={prefs.zoom}
          onChange={(v) => set({ zoom: Number(v) })}
          options={[
            { k: 75,  label: '75%' },
            { k: 80,  label: '80% (현재)' },
            { k: 90,  label: '90%' },
            { k: 100, label: '100%' },
          ]}
        />
      </SettingRow>
      <SettingRow label="정보 밀도">
        <SettingSelect
          value={prefs.density}
          onChange={(v) => set({ density: v })}
          options={[
            { k: 'compact', label: '컴팩트' },
            { k: 'normal',  label: '일반' },
          ]}
        />
      </SettingRow>
      <SettingRow label="테마">
        <SettingSelect
          value={prefs.theme}
          onChange={(v) => set({ theme: v })}
          options={[
            { k: 'light', label: 'Light' },
            { k: 'dark',  label: 'Dark (W4)' },
          ]}
        />
      </SettingRow>
    </Panel>
  );
}

function WorklistPrefCard({ prefs, set }) {
  return (
    <Panel title="환자 목록" mono="Worklist" right={<ListFilter size={12} style={{ color: 'var(--rl-ink-3)' }} />}>
      <SettingRow label="기본 진입 섹션">
        <SettingSelect
          value={prefs.defaultSection}
          onChange={(v) => set({ defaultSection: v })}
          options={[
            { k: 'today',  label: '당일 외래' },
            { k: 'unread', label: '미확인 결과' },
            { k: 'search', label: '환자 검색' },
          ]}
        />
      </SettingRow>
      <SettingRow label="자동 새로고침">
        <SettingSelect
          value={prefs.autoRefresh}
          onChange={(v) => set({ autoRefresh: Number(v) })}
          options={[
            { k: 0,   label: '꺼짐' },
            { k: 30,  label: '30초' },
            { k: 60,  label: '1분' },
            { k: 300, label: '5분' },
          ]}
        />
      </SettingRow>
      <SettingRow label="기본 정렬">
        <SettingSelect
          value={prefs.sortBy}
          onChange={(v) => set({ sortBy: v })}
          options={[
            { k: 'time',     label: '예약 시간' },
            { k: 'priority', label: '우선순위' },
            { k: 'arrival',  label: '도착 순' },
          ]}
        />
      </SettingRow>
    </Panel>
  );
}

function SecurityCard({ onLogout }) {
  const session = loadSession();
  const remaining = session.expiresAt ? Math.max(0, session.expiresAt - Date.now()) : 0;
  const minutes = Math.floor(remaining / 60000);
  return (
    <Panel title="보안 · 세션" mono="Security" right={<Shield size={12} style={{ color: 'var(--rl-ink-3)' }} />}>
      <SettingRow label="세션 저장 위치"><ReadValue mono>sessionStorage</ReadValue></SettingRow>
      <SettingRow label="세션 TTL"><ReadValue mono>1 hour</ReadValue></SettingRow>
      <SettingRow icon={<Clock size={13} />} label="세션 잔여 시간">
        <span className="font-mono text-xs" style={{ color: minutes < 10 ? 'var(--rl-amber)' : 'var(--rl-teal)' }}>
          {minutes}분
        </span>
      </SettingRow>
      <SettingRow label="환자 정보 저장" sub="개인정보보호법 · HIPAA"><span className="chip" style={{ background: 'var(--rl-teal-soft)', color: 'var(--rl-teal)' }}>저장 안함</span></SettingRow>
      <SettingRow icon={<KeyRound size={13} />} label="2단계 인증" sub="AWS Cognito 연동 W3+"><span className="font-mono text-[10px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-4)' }}>예정</span></SettingRow>
      <button
        onClick={() => { clearSession(); onLogout(); }}
        className="w-full py-2 rounded text-xs font-medium flex items-center justify-center gap-1.5 mt-2"
        style={{ background: 'var(--rl-amber-soft)', color: 'var(--rl-amber)' }}
      >
        <Clock size={12} /> 세션 즉시 만료
      </button>
    </Panel>
  );
}

function SystemCard({ onNavigate }) {
  const sysCount = MOCK_NOTIFICATION_HISTORY.filter(n => n.category === 'system').length;
  return (
    <Panel title="시스템 정보" mono="System" right={<Database size={12} style={{ color: 'var(--rl-ink-3)' }} />}>
      <SettingRow label="앱 버전">           <ReadValue mono>v0.1.0 · build 2026-04-23</ReadValue></SettingRow>
      <SettingRow label="DenseNet-121">      <ReadValue mono>v2.3.1 · 2026-03-15 retrain</ReadValue></SettingRow>
      <SettingRow label="HPO-LR 엔진">       <ReadValue mono>v1.4 · LIRICAL ported</ReadValue></SettingRow>
      <SettingRow label="HPO 데이터베이스">  <ReadValue mono>2026-03-01 release</ReadValue></SettingRow>
      <SettingRow label="Orphadata">         <ReadValue mono>2026-Q1 · 9,872 dx</ReadValue></SettingRow>
      <SettingRow label="FHIR 서버">         <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full pulse-dot" style={{ background: 'var(--rl-teal)' }} /><span className="font-mono text-[10px]" style={{ color: 'var(--rl-teal)' }}>SMART Health IT</span></span></SettingRow>
      <SettingRow label="SageMaker Endpoint"><span className="font-mono text-[10px]" style={{ color: 'var(--rl-ink-3)' }}>us-east-1 · pre-2-2team</span></SettingRow>
      <button
        onClick={() => onNavigate && onNavigate('announcement')}
        className="w-full py-2 rounded text-xs font-medium hairline-strong flex items-center justify-center gap-1.5 mt-2 hover:bg-slate-50 transition"
        style={{ color: 'var(--rl-primary)', borderColor: 'var(--rl-primary)' }}
        title="시스템 공지 페이지로 이동"
      >
        <Megaphone size={12} /> 시스템 공지 보기
        <span className="font-mono text-[9px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)' }}>· {sysCount}건</span>
        <ArrowUpRight size={11} />
      </button>
    </Panel>
  );
}

function HelpCard() {
  const links = [
    { label: '사용 가이드',          sub: '의사용 매뉴얼 · PDF' },
    { label: '키보드 단축키',        sub: '환자 검색 / 새로고침 / 로그아웃' },
    { label: '변경 내역',            sub: 'CHANGELOG · 빌드별 기능 추가' },
    { label: '개인정보 처리방침',    sub: '환자 정보 저장 정책' },
    { label: '오픈소스 라이선스',    sub: 'lucide-react · React · Tailwind …' },
  ];
  return (
    <Panel title="도움말 · 정보" mono="Help" right={<HelpCircle size={12} style={{ color: 'var(--rl-ink-3)' }} />}>
      {links.map(l => (
        <SettingRow key={l.label} label={l.label} sub={l.sub}>
          <ChevronRight size={12} style={{ color: 'var(--rl-ink-3)' }} />
        </SettingRow>
      ))}
      <div className="mt-3 pt-3 text-[10px]" style={{ borderTop: '1px solid var(--rl-border-soft)', color: 'var(--rl-ink-3)' }}>
        <div className="font-mono uppercase tracking-widest mb-1" style={{ color: 'var(--rl-primary)' }}>SKKU AWS SAY 2기 · 2팀</div>
        박성수 (Frontend) · 배기태 · 허태웅 (Model · AWS) · 권미라 · 양희인 (MIMIC · KB) · 이희찬 (멘토)
      </div>
    </Panel>
  );
}

/* ============================================================
   COMING SOON · Dashboard / Knowledge placeholder
   ============================================================ */
function ComingSoonScreen({ doctor, onLogout, onNavigate, screenKey, onOpenPatient, onOpenAnnouncement }) {
  const meta = {
    dashboard: {
      title: '분석 대시보드',
      mono: 'Analytics dashboard',
      icon: <BarChart3 size={32} style={{ color: 'var(--rl-primary)' }} />,
      desc: '병원 전체 KPI · 모델 성능 · 진단 트렌드 · 의사 동의율 (audit)',
      milestone: 'W4 · 5/11 ~ 5/17 구현 예정',
    },
    knowledge: {
      title: '지식 베이스',
      mono: 'Knowledge base',
      icon: <BookOpen size={32} style={{ color: 'var(--rl-primary)' }} />,
      desc: 'HPO term 검색 · Orphadata 참조 · 폐질환 임상 가이드라인 (Raghu 2022 · Travis 2008)',
      milestone: 'W4 · 5/11 ~ 5/17 구현 예정',
    },
  }[screenKey];

  return (
    <div className="min-h-screen flex flex-col" style={{ background: 'var(--rl-bg-2)' }}>
      <TopBar doctor={doctor} onLogout={onLogout} activeScreen={screenKey} onNavigate={onNavigate} onOpenPatient={onOpenPatient} onOpenAnnouncement={onOpenAnnouncement} />
      <main className="flex-1 max-w-[1440px] w-full mx-auto px-8 py-6">
        <div className="hairline rounded bg-white p-12 text-center fade-in">
          <div className="inline-block mb-3">{meta.icon}</div>
          <div className="font-mono text-[10px] uppercase tracking-widest mb-1" style={{ color: 'var(--rl-ink-3)' }}>
            {meta.mono}
          </div>
          <div className="font-serif text-2xl mb-2" style={{ color: 'var(--rl-ink)' }}>{meta.title}</div>
          <div className="text-sm mb-3" style={{ color: 'var(--rl-ink-3)' }}>{meta.desc}</div>
          <div className="font-mono text-[11px] uppercase tracking-widest" style={{ color: 'var(--rl-amber)' }}>
            {meta.milestone}
          </div>
          <button
            onClick={() => onNavigate('worklist')}
            className="mt-6 inline-flex items-center gap-1.5 px-4 py-2 rounded text-xs font-medium hairline-strong hover:bg-slate-50"
            style={{ color: 'var(--rl-primary)' }}
          >
            <ChevronLeft size={12} /> 환자 목록으로
          </button>
        </div>
      </main>
    </div>
  );
}

/* ============================================================
   SCREEN · ANNOUNCEMENT (시스템 공지)
   ============================================================ */
function AnnouncementScreen({ doctor, onLogout, onNavigate, onOpenPatient, onOpenAnnouncement, initialNotif }) {
  const allSys = MOCK_NOTIFICATION_HISTORY.filter(n => n.category === 'system')
    .sort((a, b) => (`${b.date} ${b.time}`).localeCompare(`${a.date} ${a.time}`));

  const firstKey = allSys[0] ? `${allSys[0].date}-${allSys[0].time}` : null;
  const initialKey = initialNotif ? `${initialNotif.date}-${initialNotif.time}` : firstKey;
  const [selectedKey, setSelectedKey] = useState(initialKey);

  const selected = allSys.find(n => `${n.date}-${n.time}` === selectedKey) || allSys[0];

  return (
    <div className="min-h-screen flex flex-col" style={{ background: 'var(--rl-bg-2)' }}>
      <TopBar doctor={doctor} onLogout={onLogout} activeScreen="settings" onNavigate={onNavigate} onOpenPatient={onOpenPatient} onOpenAnnouncement={onOpenAnnouncement} />

      <main className="flex-1 max-w-[1440px] w-full mx-auto px-8 py-6">
        {/* Breadcrumb · 설정 > 시스템 공지 */}
        <div className="flex items-center gap-1.5 mb-3 font-mono text-[10px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)' }}>
          <button
            onClick={() => onNavigate('settings')}
            className="hover:underline"
            style={{ color: 'var(--rl-primary)' }}
          >
            Settings
          </button>
          <ChevronRight size={11} />
          <span>System Announcements</span>
        </div>

        <div className="flex items-baseline gap-4 mb-5">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-widest mb-1" style={{ color: 'var(--rl-ink-3)' }}>
              System Announcements · {allSys.length}건
            </div>
            <h1 className="font-serif text-3xl" style={{ color: 'var(--rl-ink)', letterSpacing: '-0.01em' }}>
              시스템 공지
            </h1>
          </div>
          <button
            onClick={() => onNavigate('settings')}
            className="ml-auto flex items-center gap-1 px-3 py-1.5 rounded text-xs hairline-strong hover:bg-slate-50"
            style={{ color: 'var(--rl-ink-2)' }}
          >
            <ChevronLeft size={12} /> 설정으로
          </button>
        </div>

        {/* Layout: 좌 리스트 + 우 상세 */}
        <div className="grid gap-3" style={{ gridTemplateColumns: '300px 1fr', minHeight: 560 }}>
          {/* List */}
          <div className="hairline rounded bg-white overflow-hidden flex flex-col">
            <div className="px-3 py-2" style={{ borderBottom: '1px solid var(--rl-border-soft)', background: 'var(--rl-bg-3)' }}>
              <div className="font-mono text-[10px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)' }}>
                {allSys.length} announcements
              </div>
            </div>
            <div style={{ overflowY: 'auto', flex: 1 }}>
              {allSys.map(n => {
                const key = `${n.date}-${n.time}`;
                const active = selectedKey === key;
                return (
                  <button
                    key={key}
                    onClick={() => setSelectedKey(key)}
                    className="w-full text-left px-3 py-2.5 transition"
                    style={{
                      background: active ? 'var(--rl-primary-soft)' : 'transparent',
                      borderLeft: `3px solid ${active ? 'var(--rl-primary)' : 'transparent'}`,
                      borderBottom: '1px solid var(--rl-border-soft)',
                      cursor: 'pointer',
                    }}
                    onMouseEnter={e => { if (!active) e.currentTarget.style.background = 'var(--rl-bg-2)'; }}
                    onMouseLeave={e => { if (!active) e.currentTarget.style.background = 'transparent'; }}
                  >
                    <div className="flex items-baseline gap-2">
                      <SettingsIcon size={11} style={{ color: 'var(--rl-ink-3)', flexShrink: 0 }} />
                      <div className="text-xs font-medium truncate flex-1" style={{ color: 'var(--rl-ink)' }}>{n.title}</div>
                    </div>
                    <div className="font-mono text-[10px] mt-0.5 truncate" style={{ color: 'var(--rl-ink-3)' }}>
                      {n.date} · {n.time}
                    </div>
                    <div className="text-[11px] truncate mt-0.5" style={{ color: 'var(--rl-ink-3)' }}>
                      {n.text}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Detail */}
          <div className="hairline rounded bg-white p-6">
            {selected ? <AnnouncementDetail n={selected} /> : (
              <div className="text-center py-20 text-sm" style={{ color: 'var(--rl-ink-3)' }}>
                공지를 선택하세요
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

function AnnouncementDetail({ n }) {
  const d = n.detail || {};
  return (
    <div className="fade-in">
      <div className="flex items-baseline gap-3 pb-3" style={{ borderBottom: '1px solid var(--rl-border-soft)' }}>
        <span className="chip" style={{ background: 'var(--rl-bg-3)', color: 'var(--rl-ink-2)' }}>
          <SettingsIcon size={10} /> System
        </span>
        <span className="font-mono text-[10px] uppercase tracking-widest" style={{ color: 'var(--rl-ink-3)' }}>
          {d.component || ''}
        </span>
        <span className="ml-auto font-mono text-[11px]" style={{ color: 'var(--rl-ink-3)' }}>
          {n.date} {n.time} KST
        </span>
      </div>

      <h2 className="font-serif text-xl mt-4 mb-2" style={{ color: 'var(--rl-ink)', letterSpacing: '-0.01em' }}>
        {n.title}
      </h2>
      <div className="text-sm mb-5 leading-relaxed" style={{ color: 'var(--rl-ink-2)' }}>
        {n.text}
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4">
        {d.version && (
          <div className="p-3 rounded" style={{ background: 'var(--rl-bg-2)' }}>
            <div className="font-mono text-[9px] uppercase tracking-widest mb-0.5" style={{ color: 'var(--rl-ink-3)' }}>Version</div>
            <div className="font-mono text-sm" style={{ color: 'var(--rl-ink)' }}>{d.version}</div>
          </div>
        )}
        {d.deployedBy && (
          <div className="p-3 rounded" style={{ background: 'var(--rl-bg-2)' }}>
            <div className="font-mono text-[9px] uppercase tracking-widest mb-0.5" style={{ color: 'var(--rl-ink-3)' }}>Deployed by</div>
            <div className="text-sm" style={{ color: 'var(--rl-ink)' }}>{d.deployedBy}</div>
          </div>
        )}
      </div>

      {d.changes && (
        <div
          className="p-4 rounded text-sm leading-relaxed"
          style={{ background: 'var(--rl-primary-soft)', borderLeft: '3px solid var(--rl-primary)', color: 'var(--rl-ink)' }}
        >
          <div className="font-mono text-[10px] uppercase tracking-widest mb-2" style={{ color: 'var(--rl-primary)' }}>
            Changes · 변경 내역
          </div>
          {d.changes}
        </div>
      )}

      <div className="mt-5 pt-3 text-[11px]" style={{ borderTop: '1px solid var(--rl-border-soft)', color: 'var(--rl-ink-3)' }}>
        <span className="font-mono uppercase tracking-widest" style={{ color: 'var(--rl-amber)' }}>⚠ EU AI Act Art. 22</span>
        &nbsp;· 본 공지는 AI 시스템 변경 사항이며, 진단 결과 해석에 영향을 줄 수 있습니다.
      </div>
    </div>
  );
}

/* ============================================================
   MOCK DATA · 오늘 외래 (9명)
   - acknowledged: 의사가 결과를 이미 확인했는지
   - resultAt:     AI 분석 완료 시각 (HH:mm KST)
   ============================================================ */

/* ---- Demo용 finalReport mock 데이터 · Bedrock JSON 포맷 ---- */
const DEMO_FINAL_REPORT_IPF = {
  generated_at: '2026-04-23T09:14:00+09:00',
  llm_model: 'anthropic.claude-3-5-sonnet-20240620-v1:0',
  rag_apis_used: ['pubcasefinder', 'monarch', 'pubmed', 'clinicaltrials'],
  recommendation: {
    immediate_workup: [
      'HRCT 흉부 고해상도 — UIP/NSIP 패턴 감별 (우선 시행)',
      '폐기능검사 (PFT): FVC, DLCO, TLC — 기저치 확립',
      '6분 보행거리검사 (6MWT) — 운동 내성 평가',
      'BAL (기관지폐포세척) — HRCT 비전형 소견 시 추가 고려',
    ],
    specialist_referral: [
      '[MDT 필수 · ORPHA:2032] 희귀질환센터 다학제팀 — 호흡기내과 + 영상의학과 + 병리과',
      '유전체 의학과 — 가족성 IPF 가능성 평가 (TERT/TERC 변이)',
      '호흡 재활팀 — 폐 재활 프로그램 조기 의뢰',
    ],
    treatment_guideline: [
      '[IPF] Nintedanib 150mg bid (1차 선택) · INPULSIS trial (PMID: 24937360)',
      '[IPF] Pirfenidone 801mg tid (대안) · ATS/ERS/JRS/ALAT 2022 (PMID: 35167184)',
      '[IPF] 안정 시 SpO₂ < 88% 이면 장기 산소 요법 처방',
    ],
    clinical_trial_info: [
      'NCT04052334 — FIBRONEER-IPF: Zinpentraxin alfa vs. Placebo (모집 중 · 성인 IPF)',
      'NCT05607745 — BBT-877 Phase 2 (한국 포함 다국가 · 모집 중)',
    ],
    genetic_test: [
      'TERT, TERC — 복수 소스 확인 (Orphadata + Monarch 일치)',
      'SFTPC, SFTPA2 — 가족성 ILD 관련',
      'MUC5B rs35705950 — IPF 최대 단일 유전적 위험 인자 (OR ~5.0)',
    ],
    additional_lab: [
      'KL-6 (혈청) — ILD 활성도 지표',
      'SP-D (Surfactant Protein D)',
      '자가항체 패널: ANA, RF, Anti-CCP, Anti-MDA5 — RA-ILD 감별',
      'NT-proBNP — 폐고혈압 동반 여부 스크리닝',
    ],
  },
  clinical_notes: {
    summary: '58세 남성, 3개월간 진행성 호흡곤란·마른기침·체중 4kg 감소 내원. SpO₂ 93% (RA). CXR 양측 하부 reticular opacity. DenseNet-121: IPF 84%, Sarcoidosis 62%, HP 41%. Penicillin 알레르기.',
    top1_reasoning: '【Positive HPO】 HP:0002094 호흡곤란·HP:0006510 진행성 폐섬유증 패턴·HP:0045051 CXR reticular opacity 3개 일치. 【Lab】 SpO₂ 93% 저하·KL-6 예상 상승. 【CXR】 DenseNet 84% 하부 reticular — UIP 패턴 강력 의심. 3개월 점진 진행 경과가 IPF 자연경과와 합치.',
    differential_note: 'Sarcoidosis(62%): 상부폐 침범·림프절종대 미확인, serum ACE 정상이면 우선순위 낮음. HP(41%): 항원 노출력 없음·BAL 림프구증다증 확인 필요. 두 질환 모두 HRCT에서 UIP 외 다른 패턴 예상.',
    rag_evidence: 'DB·API 교차검증 일치 — Orphadata ORPHA:2032: 유전자 TERT/TERC/SFTPC (Disease-causing). Monarch 데이터 일치. PubCaseFinder 상위 매칭 확인. PubMed INPULSIS(PMID:24937360) 치료 근거. 유병률 10/100,000 (성인 40대↑ 남성 호발).',
    case_comparison: 'PubMed(PMID:35167184): 56세 남성, 6개월 호흡곤란·UIP 패턴 HRCT, TERT 변이 확인 → Nintedanib 치료 반응 양호. 본 환자와 발병 연령·성별·진행 경과 유사. TERT 유전자 검사 및 가족력 추가 청취 권고.',
    epidemiology_note: 'ORPHA:2032 · 특발성 폐섬유증(IPF) · 유병률 10/100,000 (성인). 발병: 주로 50대↑ 남성. 상염색체 우성 (가족성 일부). Monarch + Orphadata 데이터 일치.',
    disclaimer: 'AI 결과는 진단 보조이며 최종 진단은 주치의의 임상 판단과 추가 검사 결과를 종합하여 확정합니다.',
  },
  confidence_metrics: {
    overall_confidence_score: 0.87,
    rationale: 'DenseNet 84% + HPO 3개 양성 일치 + Orphadata-Monarch 교차검증 일치 + PubMed 케이스 유사도 높음. HRCT/PFT 미확인으로 0.87.',
    data_sufficiency: { genomic_evidence: 'High', clinical_case_match: 'High', trial_availability: 'Medium' },
  },
};

const DEMO_FINAL_REPORT_LAM = {
  generated_at: '2026-04-23T08:58:00+09:00',
  llm_model: 'anthropic.claude-3-5-sonnet-20240620-v1:0',
  rag_apis_used: ['pubcasefinder', 'monarch', 'pubmed', 'clinicaltrials'],
  recommendation: {
    immediate_workup: [
      'HRCT 흉부 고해상도 — thin-walled bilateral cyst 패턴 확인',
      '혈청 VEGF-D 측정 — LAM 진단 바이오마커 (cutoff ≥800 pg/mL)',
      '폐기능검사: FEV1/FVC ratio (폐쇄성 패턴 예상)',
      '복부/골반 MRI — 신장 혈관근지방종(AML) 동반 확인',
    ],
    specialist_referral: [
      '[MDT 필수 · ORPHA:538] 희귀질환센터 다학제팀 — LAM Foundation 프로토콜',
      '신장내과 협진 — AML 동반 시 출혈 위험 평가',
      '유전자 상담 — TSC2 변이 검사 (TSC-LAM vs. sporadic LAM 감별)',
    ],
    treatment_guideline: [
      '[LAM] Sirolimus (Rapamycin) 2mg/day — ATS 가이드라인 권고 (PMID: 23985991)',
      '[LAM] 폐기능 FEV1 진행성 저하 시 조기 시작 권고',
      '[기흉 예방] 재발성 기흉 기왕력 — 흉막 유착술 고려 (양측 발생 시)',
    ],
    clinical_trial_info: [
      'NCT05431972 — LAM 바이오마커 종단 연구 (모집 중 · 여성 환자 대상)',
      'NCT03155399 — mTOR 억제제 최적 용량 연구 (결과 발표 예정)',
    ],
    genetic_test: [
      'TSC1, TSC2 — 복수 소스 확인 (Orphadata + Monarch 일치)',
      'MTOR 경로 변이 패널 — Somatic/germline 구분',
    ],
    additional_lab: [
      'VEGF-D (혈청) — LAM 진단 바이오마커 (≥800 pg/mL 진단적)',
      '소변 세포학 — LAM 세포 확인',
      'ESR, CRP — 기저 염증 평가',
    ],
  },
  clinical_notes: {
    summary: '29세 여성, 재발성 기흉 및 호흡곤란 내원. SpO₂ 94% (RA). CXR 양측 cystic change 의심. DenseNet-121: LAM 58%, PLCH 31%, 재발성 기흉 18%. 생식연령 여성에서 재발성 기흉은 LAM 강력 시사.',
    top1_reasoning: '【Positive HPO】 HP:0002107 기흉·HP:0002094 호흡곤란 일치. 【인구통계】 29세 가임기 여성 — LAM의 전형적 발병 집단. 【CXR】 bilateral cystic pattern DenseNet 58%. 재발성 기흉 기왕력이 LAM 가장 특징적 임상 발현.',
    differential_note: 'PLCH(31%): 흡연력 없음 → 가능성 낮음. PLCH는 성인 흡연자 호발. 단순 재발성 기흉(18%): bilateral cystic pattern이 원발성 자연기흉과 불일치. VEGF-D + HRCT thin-walled cyst 확인 시 LAM 확진 가능.',
    rag_evidence: 'DB·API 교차검증 일치 — Orphadata ORPHA:538: TSC1/TSC2 Disease-causing, Monarch 교차확인. PubCaseFinder HPO 매칭 Top 1. VEGF-D 진단적 cutoff 확인(PMID:23985991). 유병률 1/100,000 여성. 발병 20~40대 가임기 여성.',
    case_comparison: 'PubMed(PMID:23985991): 31세 여성, 재발성 기흉 3회, HRCT thin-walled cyst, VEGF-D 1,240 pg/mL → LAM 확진, Sirolimus 치료. 본 환자와 연령·성별·주증상 거의 동일. VEGF-D 즉시 측정 권고.',
    epidemiology_note: 'ORPHA:538 · Lymphangioleiomyomatosis · 유병률 1/100,000 (여성). 발병: 주로 20~40대 가임기 여성. mTOR 경로 이상 (TSC2 변이). Sporadic/TSC-related 두 형태.',
    disclaimer: 'AI 결과는 진단 보조이며 최종 진단은 주치의의 임상 판단과 추가 검사 결과를 종합하여 확정합니다.',
  },
  confidence_metrics: {
    overall_confidence_score: 0.82,
    rationale: '가임기 여성 + 재발성 기흉 + bilateral cystic pattern = LAM 전형적 3요소. VEGF-D 미측정으로 0.82. HRCT + VEGF-D 확인 후 0.92+ 예상.',
    data_sufficiency: { genomic_evidence: 'High', clinical_case_match: 'High', trial_availability: 'Medium' },
  },
};

const MOCK_PATIENTS = [
  {
    time: '08:30', visit: '초진',
    name: '김○○', sex: 'M', age: 58, mrn: '20-145982',
    complaint: '호흡곤란 3개월 · 마른기침 · 체중감소 4kg',
    allergy: 'Penicillin',
    vitals: 'BP 128/76 · HR 88 · RR 22 · SpO₂ 93% (RA) · T 36.6°C',
    cxr: 'arrived', status: 'ready',
    rare: true, dontMiss: true,
    acknowledged: false, resultAt: '08:21',
    topDx: null,
    preview: [
      { name: '특발성 폐섬유증 (IPF)', prob: 0.84, rare: true, dontMiss: true, orpha: 'ORPHA:2032' },
      { name: 'Sarcoidosis',         prob: 0.62, rare: false },
      { name: '과민성 폐렴 (HP)',       prob: 0.41, rare: false },
    ],
    finalReport: DEMO_FINAL_REPORT_IPF,
  },
  {
    time: '09:00', visit: '재진',
    name: '이○○', sex: 'F', age: 42, mrn: '21-093127',
    complaint: '만성기침 2주 · 가래',
    vitals: 'BP 124/78 · HR 96 · RR 22 · SpO₂ 94% (RA) · T 38.2°C',
    cxr: 'arrived', status: 'ready',
    rare: false, dontMiss: false,
    acknowledged: true, resultAt: '07:55',
    topDx: 'Pneumonia',
    preview: [
      { name: 'Community-acquired Pneumonia', prob: 0.76, rare: false },
      { name: 'Acute Bronchitis',             prob: 0.52, rare: false },
      { name: 'Asthma exacerbation',          prob: 0.18, rare: false },
    ],
  },
  {
    time: '09:30', visit: '초진',
    name: '박○○', sex: 'F', age: 34, mrn: '22-014556',
    complaint: '객혈 · 야간 발한 2주',
    vitals: 'BP 118/72 · HR 102 · RR 22 · SpO₂ 92% (RA) · T 37.8°C',
    cxr: 'arrived', status: 'analyzing',
    rare: true, dontMiss: true,
    topDx: null,
    preview: null,
  },
  {
    time: '10:00', visit: '재진',
    name: '최○○', sex: 'M', age: 67, mrn: '19-445621',
    complaint: '흉통 · 호흡곤란 · 부종',
    vitals: 'BP 144/92 · HR 98 · RR 22 · SpO₂ 91% (RA) · T 36.7°C',
    cxr: 'arrived', status: 'ready',
    rare: false, dontMiss: false,
    acknowledged: true, resultAt: '08:42',
    topDx: 'CHF',
    preview: [
      { name: 'Congestive Heart Failure', prob: 0.81, rare: false },
      { name: 'Pleural Effusion · RT',    prob: 0.54, rare: false },
      { name: 'Pneumonia',                prob: 0.22, rare: false },
    ],
  },
  {
    time: '10:30', visit: '초진',
    name: '정○○', sex: 'F', age: 29, mrn: '22-089433',
    complaint: '호흡곤란 · 재발성 기흉',
    vitals: 'BP 110/68 · HR 84 · RR 20 · SpO₂ 94% (RA) · T 36.5°C',
    cxr: 'arrived', status: 'ready',
    rare: true, dontMiss: false,
    acknowledged: false, resultAt: '08:58',
    topDx: null,
    preview: [
      { name: 'Lymphangioleiomyomatosis (LAM)', prob: 0.58, rare: true, orpha: 'ORPHA:538' },
      { name: 'Pulmonary Langerhans Cell Histiocytosis', prob: 0.31, rare: true, orpha: 'ORPHA:99874' },
      { name: '재발성 기흉 (idiopathic)',     prob: 0.18, rare: false },
    ],
    finalReport: DEMO_FINAL_REPORT_LAM,
  },
  {
    time: '11:00', visit: '재진',
    name: '한○○', sex: 'M', age: 71, mrn: '18-332108',
    complaint: '만성 흡연력 · 객담 · 운동 시 호흡곤란',
    vitals: 'BP 138/86 · HR 82 · RR 20 · SpO₂ 92% (RA) · T 36.4°C',
    cxr: 'pending', status: 'pending',
    rare: false, dontMiss: false,
    topDx: null,
  },
  {
    time: '11:30', visit: '초진',
    name: '윤○○', sex: 'F', age: 51, mrn: '22-145012',
    complaint: '흉통 · 두근거림 1주',
    vitals: 'BP 128/82 · HR 90 · RR 18 · SpO₂ 96% (RA) · T 36.6°C',
    cxr: 'arrived', status: 'analyzing',
    rare: false, dontMiss: false,
    topDx: null,
  },
  {
    time: '13:00', visit: '초진',
    name: '오○○', sex: 'M', age: 45, mrn: '22-145098',
    complaint: '단순 건강검진 · CXR 이상 소견 FU',
    vitals: 'BP 122/78 · HR 76 · RR 16 · SpO₂ 97% (RA) · T 36.5°C',
    cxr: 'pending', status: 'pending',
    rare: false, dontMiss: false,
    topDx: null,
  },
  {
    time: '13:30', visit: '초진',
    name: '장○○', sex: 'F', age: 62, mrn: '22-145103',
    complaint: '기침 · 체중감소 · 류마티스 관절염 과거력',
    vitals: 'BP 130/80 · HR 88 · RR 20 · SpO₂ 93% (RA) · T 36.9°C',
    cxr: 'arrived', status: 'ready',
    rare: true, dontMiss: true,
    acknowledged: false, resultAt: '07:12',
    topDx: null,
    preview: [
      { name: 'RA-associated ILD (NSIP pattern)', prob: 0.69, rare: true, dontMiss: true, orpha: 'ORPHA:79126' },
      { name: '과민성 폐렴 (HP)',                    prob: 0.42, rare: false },
      { name: 'IPF',                              prob: 0.28, rare: true, orpha: 'ORPHA:2032' },
    ],
  },
  {
    time: '14:00', visit: '재진',
    name: '원○○', sex: 'F', age: 22, mrn: '23-145220',
    complaint: '과호흡 · 두근거림 · 시험 기간 스트레스',
    vitals: 'BP 128/82 · HR 112 · RR 28 · SpO₂ 99% (RA) · T 36.7°C',
    cxr: 'arrived', status: 'ready',
    rare: false, dontMiss: false,
    acknowledged: false, resultAt: '13:42',
    topDx: 'HVS',
    preview: [
      { name: 'Hyperventilation Syndrome (과호흡증후군)', prob: 0.71, rare: false },
      { name: 'Anxiety · Panic attack',                  prob: 0.58, rare: false },
      { name: 'Asthma exacerbation',                     prob: 0.16, rare: false },
    ],
  },
];

/* ============================================================
   MOCK DATA · 과거 환자 (검색 섹션 데모용)
   visitDate: 'YYYY-MM-DD' (오늘 = 2026-04-23)
   ============================================================ */
const MOCK_PATIENT_HISTORY = [
  {
    time: '14:00', visit: '재진',
    visitDate: '2026-04-22',
    name: '강○○', sex: 'M', age: 73, mrn: '15-228714',
    complaint: '만성 호흡곤란 · 흡연력 50 pack-year',
    cxr: 'arrived', status: 'ready',
    rare: false, dontMiss: false,
    acknowledged: true, resultAt: '13:48',
    preview: [
      { name: 'COPD · GOLD III', prob: 0.79, rare: false },
      { name: 'Bronchiectasis',  prob: 0.34, rare: false },
    ],
  },
  {
    time: '10:30', visit: '재진',
    visitDate: '2026-04-16',
    name: '문○○', sex: 'F', age: 38, mrn: '20-118245',
    complaint: '활동 시 호흡곤란 · 광범위 ground-glass',
    cxr: 'arrived', status: 'ready',
    rare: true, dontMiss: false,
    acknowledged: true, resultAt: '10:12',
    preview: [
      { name: 'Nonspecific Interstitial Pneumonia (NSIP)', prob: 0.66, rare: true, orpha: 'ORPHA:79126' },
      { name: '과민성 폐렴 (HP)', prob: 0.39, rare: false },
    ],
  },
  {
    time: '11:15', visit: '초진',
    visitDate: '2026-03-28',
    name: '서○○', sex: 'M', age: 29, mrn: '21-301122',
    complaint: '재발성 폐렴 · 면역결핍 의심',
    allergy: 'Sulfa',
    cxr: 'arrived', status: 'ready',
    rare: true, dontMiss: true,
    acknowledged: true, resultAt: '11:01',
    preview: [
      { name: 'Common Variable Immune Deficiency (CVID)', prob: 0.71, rare: true, dontMiss: true, orpha: 'ORPHA:1572' },
      { name: 'Bronchiectasis · post-infectious', prob: 0.48, rare: false },
    ],
  },
  {
    time: '09:00', visit: '재진',
    visitDate: '2026-03-04',
    name: '구○○', sex: 'F', age: 56, mrn: '17-882034',
    complaint: '천식 악화 · 야간 기침',
    cxr: 'arrived', status: 'ready',
    rare: false, dontMiss: false,
    acknowledged: true, resultAt: '08:48',
    preview: [
      { name: 'Asthma · severe persistent', prob: 0.83, rare: false },
    ],
  },
  {
    time: '13:30', visit: '초진',
    visitDate: '2026-02-12',
    name: '백○○', sex: 'M', age: 64, mrn: '22-145210',
    complaint: '체중감소 · 객혈 · 림프절 종대',
    cxr: 'arrived', status: 'ready',
    rare: false, dontMiss: true,
    acknowledged: true, resultAt: '13:14',
    preview: [
      { name: '폐결핵 · 활동성',        prob: 0.74, rare: false, dontMiss: true },
      { name: 'Lung Cancer (NSCLC)', prob: 0.41, rare: false },
    ],
  },
  {
    time: '15:00', visit: '재진',
    visitDate: '2026-01-30',
    name: '나○○', sex: 'F', age: 47, mrn: '19-557741',
    complaint: '아급성 발열 · 양측성 결절',
    cxr: 'arrived', status: 'ready',
    rare: true, dontMiss: false,
    acknowledged: true, resultAt: '14:42',
    preview: [
      { name: 'Granulomatosis with Polyangiitis (GPA)', prob: 0.62, rare: true, orpha: 'ORPHA:900' },
      { name: 'Sarcoidosis', prob: 0.44, rare: false },
    ],
  },
];
