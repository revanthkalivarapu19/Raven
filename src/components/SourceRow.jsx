import { useState } from 'react';
import { ChevronDownIcon } from './Icons';

export function SourceRow({ sourceData }) {
  const [isExpanded, setIsExpanded] = useState(false);

  const {
    source = 'Source',
    stance = 'unclear',
    relevance = 0,
    trust = 0,
    excerpt = '',
    note = '',
  } = sourceData;

  const stanceLabels = {
    support: 'Supports',
    contradict: 'Contradicts',
    limiting: 'Limiting',
    unclear: 'Unclear',
  };

  const stanceText = stanceLabels[stance] || 'Unclear';

  return (
    <div className={`source-item-container ${isExpanded ? 'expanded' : ''}`}>
      <button
        type="button"
        className="source-row-button"
        aria-expanded={isExpanded}
        onClick={() => setIsExpanded((prev) => !prev)}
      >
        <div className="source-row-top">
          <span className="source-name">{source}</span>
          <div className="source-badges">
            <span className={`stance-badge stance-${stance}`}>
              {stanceText}
            </span>
            <div className="source-mini-bar" role="presentation" aria-hidden="true">
              <div
                className="source-mini-bar-fill"
                style={{ width: `${trust || relevance}%` }}
              />
            </div>
            <span className="trust-badge">
              {trust || relevance}%
            </span>
          </div>
        </div>

        <div className={`source-excerpt ${isExpanded ? 'full' : 'clamped'}`}>
          {excerpt}
        </div>

        <div className="source-expand-hint">
          <span className="expand-hint-text">
            {isExpanded ? 'Show less' : 'View full excerpt and credibility scores'}
          </span>
          <ChevronDownIcon
            size={14}
            className={`expand-chevron ${isExpanded ? 'rotated' : ''}`}
          />
        </div>
      </button>

      {isExpanded && (
        <div className="source-details-panel">
          <div className="source-metrics">
            <div className="metric-row">
              <div className="metric-label-group">
                <span className="metric-label">Relevance</span>
                <span className="metric-score">{relevance}%</span>
              </div>
              <div
                className="metric-bar-track"
                role="progressbar"
                aria-valuenow={relevance}
                aria-valuemin="0"
                aria-valuemax="100"
                aria-label={`Relevance: ${relevance}%`}
              >
                <div
                  className="metric-bar-fill"
                  style={{ width: `${relevance}%` }}
                />
              </div>
            </div>

            <div className="metric-row">
              <div className="metric-label-group">
                <span className="metric-label">Source trust</span>
                <span className="metric-score">{trust}%</span>
              </div>
              <div
                className="metric-bar-track"
                role="progressbar"
                aria-valuenow={trust}
                aria-valuemin="0"
                aria-valuemax="100"
                aria-label={`Source trust: ${trust}%`}
              >
                <div
                  className="metric-bar-fill"
                  style={{ width: `${trust}%` }}
                />
              </div>
            </div>
          </div>

          {note && (
            <div className="source-note-box">
              <span className="source-note-title">Assessment note:</span>
              <p className="source-note-content">{note}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
