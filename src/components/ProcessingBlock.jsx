import { CheckCircleIcon } from './Icons';

const PROCESSING_STEPS = [
  'Reading your claim',
  'Finding sources',
  'Comparing what they say',
  'Weighing the result',
];

export function ProcessingBlock({ currentStep = 0 }) {
  return (
    <div className="processing-card" role="status" aria-label="Processing claim">
      <div className="processing-header">
        <span className="processing-title">Verifying against sources…</span>
      </div>
      <ol className="processing-steps-list">
        {PROCESSING_STEPS.map((stepText, idx) => {
          const isDone = idx < currentStep;
          const isActive = idx === currentStep;
          const isPending = idx > currentStep;

          let stepClass = 'step-item';
          if (isDone) stepClass += ' done';
          if (isActive) stepClass += ' active';
          if (isPending) stepClass += ' pending';

          return (
            <li key={stepText} className={stepClass} aria-current={isActive ? 'step' : undefined}>
              <span className="step-indicator">
                {isDone ? (
                  <CheckCircleIcon size={14} className="step-icon-done" />
                ) : (
                  <span className="step-number">{idx + 1}</span>
                )}
              </span>
              <span className="step-text">{stepText}</span>
              {isActive && <span className="step-spinner" aria-hidden="true" />}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
