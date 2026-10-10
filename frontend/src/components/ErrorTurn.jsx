export function ErrorTurn({ onRetry }) {
  return (
    <div className="error-turn-card" role="alert">
      <div className="error-turn-content">
        <span className="error-turn-title">Verification failed</span>
        <p className="error-turn-message">
          We encountered an issue checking this claim against our sources.
        </p>
      </div>
      <button
        type="button"
        className="btn-retry-claim"
        onClick={onRetry}
      >
        Try again
      </button>
    </div>
  );
}
