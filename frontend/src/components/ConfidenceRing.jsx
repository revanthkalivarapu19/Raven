export function ConfidenceRing({ score = 0, verdict = 'unverified' }) {
  const radius = 24;
  const stroke = 5;
  const normalizedRadius = radius - stroke / 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  return (
    <div className="confidence-wrapper">
      <div
        className={`confidence-ring-container verdict-${verdict}`}
        role="img"
        aria-label={`Confidence in this verdict: ${score}%`}
      >
        <svg
          height={radius * 2}
          width={radius * 2}
          className="confidence-svg"
          aria-hidden="true"
        >
          <circle
            className="confidence-track"
            strokeWidth={stroke}
            fill="transparent"
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
          <circle
            className="confidence-indicator"
            strokeWidth={stroke}
            strokeDasharray={`${circumference} ${circumference}`}
            style={{ strokeDashoffset }}
            strokeLinecap="round"
            fill="transparent"
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
        </svg>
        <span className="confidence-value">{score}%</span>
      </div>
      <div className="confidence-meta">
        <span className="confidence-label">Confidence in this verdict</span>
      </div>
    </div>
  );
}
