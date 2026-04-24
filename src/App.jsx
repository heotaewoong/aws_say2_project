import React, { useState } from 'react';
import RareLinkApp from './LoginWorklist.jsx';
import RareLinkDesignSystem from './DesignSystem.jsx';

/**
 * App root. Includes a tiny dev-only view switcher at bottom-right
 * so you can toggle between:
 *   - 'app'    : Login → Worklist (실제 의사 플로우)
 *   - 'system' : Design System v0.1 showcase (컬러/타이포/컴포넌트)
 *
 * 실전 배포 때는 이 스위처 블록을 제거하거나 환경변수로 가려주세요.
 */
export default function App() {
  const [view, setView] = useState('app');

  return (
    <>
      {view === 'app' ? <RareLinkApp /> : <RareLinkDesignSystem />}

      {/* Dev-only view switcher */}
      <div
        style={{
          position: 'fixed',
          bottom: 16,
          right: 16,
          zIndex: 99999,
          background: 'white',
          border: '1px solid #CBD5E1',
          borderRadius: 8,
          padding: 4,
          boxShadow: '0 8px 24px rgba(10,22,40,0.12)',
          display: 'flex',
          gap: 2,
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: 11,
        }}
      >
        <SwitchBtn active={view === 'app'}    onClick={() => setView('app')}>App</SwitchBtn>
        <SwitchBtn active={view === 'system'} onClick={() => setView('system')}>Design System</SwitchBtn>
      </div>
    </>
  );
}

function SwitchBtn({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '6px 12px',
        borderRadius: 6,
        border: 'none',
        background: active ? '#0C447C' : 'transparent',
        color: active ? 'white' : '#334155',
        cursor: 'pointer',
        fontSize: 11,
        fontWeight: 500,
        letterSpacing: '0.02em',
        textTransform: 'uppercase',
      }}
    >
      {children}
    </button>
  );
}
