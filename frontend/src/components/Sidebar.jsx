import { PlusIcon } from './Icons';

export function Sidebar({ claims = [], activeClaimId, onSelectClaim, onNewCheck, onOpenAbout }) {
  return (
    <aside className="app-sidebar" aria-label="Sidebar navigation">
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <span className="brand-wordmark">RAVEN</span>
          <span className="brand-tagline">Evidence before opinion.</span>
        </div>
        <button
          type="button"
          className="btn-new-check"
          onClick={onNewCheck}
          aria-label="Start a new check"
        >
          <PlusIcon size={14} />
          <span>New check</span>
        </button>
      </div>

      <div className="sidebar-session-section">
        <div className="session-section-header">
          <h2 className="session-heading">This session</h2>
          <span className="session-note">Temporary conversation</span>
        </div>
        <nav aria-label="Claims in this session" className="session-claims-nav">
          {claims.length === 0 ? (
            <div className="empty-session-list" aria-hidden="true" />
          ) : (
            <ul className="claims-list">
              {claims.map((claim, index) => {
                const claimNumber = String(index + 1).padStart(2, '0');
                const isSelected = claim.id === activeClaimId;
                return (
                  <li key={claim.id}>
                    <button
                      type="button"
                      className={`claim-nav-item ${isSelected ? 'active' : ''}`}
                      onClick={() => onSelectClaim && onSelectClaim(claim.id)}
                      aria-current={isSelected ? 'true' : undefined}
                    >
                      <span className="claim-item-number">Claim {claimNumber}</span>
                      <span className="claim-item-preview">{claim.preview || claim.text || 'Screenshot check'}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </nav>
      </div>

      <div className="sidebar-footer">
        <button
          type="button"
          className="btn-about-raven"
          onClick={onOpenAbout}
        >
          About RAVEN
        </button>
      </div>
    </aside>
  );
}
