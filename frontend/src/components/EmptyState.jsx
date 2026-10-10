const EXAMPLE_PROMPTS = [
  'Is this claim true: scientists developed an AI model that can detect cancer much earlier?',
  'Does drinking coffee every day prevent heart disease?',
  "Did scientists recently discover a new treatment for Alzheimer's disease?",
  'Does this article support the claim it makes?',
];

export function EmptyState({ onSelectExample = () => {} }) {
  return (
    <div className="empty-state-container">
      <div className="empty-state-header">
        <h2 className="empty-state-heading">Check a claim against the evidence</h2>
        <p className="empty-state-copy">
          Enter a factual claim or upload a screenshot to evaluate what verified sources say.
        </p>
      </div>

      <div className="empty-state-examples">
        <span className="examples-heading">Suggested claims</span>
        <div className="examples-grid">
          {EXAMPLE_PROMPTS.map((prompt) => (
            <button
              key={prompt}
              type="button"
              className="example-prompt-btn"
              onClick={() => onSelectExample(prompt)}
            >
              <span className="example-prompt-text">{prompt}</span>
              <span className="example-prompt-arrow" aria-hidden="true">→</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
