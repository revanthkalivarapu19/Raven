import { SunIcon, MoonIcon } from './Icons';

export function TopBar({ theme = 'light', onToggleTheme = () => {} }) {
  const isDark = theme === 'dark';

  return (
    <header className="app-topbar">
      <div className="topbar-inner">
        <span className="topbar-mobile-wordmark" aria-hidden="true">RAVEN</span>
        <h1 className="topbar-title">Claim verification</h1>
      </div>
      <div className="topbar-actions">
        <button
          type="button"
          className="btn-theme-toggle"
          onClick={onToggleTheme}
          aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {isDark ? <SunIcon size={14} /> : <MoonIcon size={14} />}
          <span className="theme-toggle-label">{isDark ? 'Light' : 'Dark'}</span>
        </button>
      </div>
    </header>
  );
}
