import { useState, useMemo, useRef } from 'react';
import { ConfidenceRing } from './ConfidenceRing';
import { SourceRow } from './SourceRow';

export function ResultCard({
  result,
  onOpenWhyResult = () => {},
  onOpenAnotherSource = () => {},
}) {
  const [activeTab, setActiveTab] = useState('all');
  const sourcesSectionRef = useRef(null);

  const {
    verdict = 'unverified',
    confidence = 0,
    status = '',
    why = '',
    evidence = [],
  } = result || {};

  // Formatted verdict display
  const verdictMap = {
    real: { label: 'Real', className: 'verdict-badge-real' },
    fake: { label: 'Fake', className: 'verdict-badge-fake' },
    unverified: { label: 'Unverified', className: 'verdict-badge-unverified' },
    uncertain: { label: 'Uncertain', className: 'verdict-badge-unverified' },
  };

  const verdictConfig = verdictMap[verdict.toLowerCase()] || verdictMap.unverified;

  // Stance counts
  const counts = useMemo(() => {
    const supportCount = evidence.filter((e) => e.stance === 'support').length;
    const contradictCount = evidence.filter((e) => e.stance === 'contradict' || e.stance === 'limiting').length;
    const unclearCount = evidence.filter((e) => e.stance === 'unclear').length;
    return {
      all: evidence.length,
      support: supportCount,
      contradict: contradictCount,
      unclear: unclearCount,
    };
  }, [evidence]);

  // Filtered evidence
  const filteredEvidence = useMemo(() => {
    if (activeTab === 'all') return evidence;
    if (activeTab === 'contradict') {
      return evidence.filter((e) => e.stance === 'contradict' || e.stance === 'limiting');
    }
    return evidence.filter((e) => e.stance === activeTab);
  }, [evidence, activeTab]);

  const handleScrollToSources = () => {
    if (sourcesSectionRef.current) {
      sourcesSectionRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  };

  return (
    <article className="result-card-container" aria-label={`Fact check result: ${verdictConfig.label}`}>
      <div className="result-card-inner">
        {/* Left Column: Verdict Overview */}
        <section className="result-overview-column" aria-label="Verdict summary">
          <div className="verdict-banner">
            <span className="verdict-category-label">Verdict</span>
            <div className={`verdict-display ${verdictConfig.className}`}>
              <span className="verdict-word">{verdictConfig.label}</span>
            </div>
          </div>

          <ConfidenceRing
            score={confidence}
            verdict={verdict.toLowerCase() === 'uncertain' ? 'unverified' : verdict.toLowerCase()}
          />

          <div className="verdict-narrative">
            <p className="verdict-status-sentence">{status}</p>
            <p className="verdict-why-explanation">{why}</p>
          </div>

          <div className="verdict-actions-group">
            <button
              type="button"
              className="btn-why-result"
              onClick={() => onOpenWhyResult(result)}
            >
              Why this result?
            </button>
            <button
              type="button"
              className="btn-read-sources"
              onClick={handleScrollToSources}
            >
              Read the sources
            </button>
            <button
              type="button"
              className="btn-another-source"
              onClick={() => onOpenAnotherSource(result)}
            >
              I have another source
            </button>
          </div>
        </section>

        {/* Right Column: Sources and Evidence */}
        <section
          className="result-sources-column"
          ref={sourcesSectionRef}
          aria-label="What the sources say"
        >
          <div className="sources-header">
            <h3 className="sources-heading">What the sources say</h3>
          </div>

          {evidence.length === 0 ? (
            <div className="no-sources-state">
              <p className="no-sources-message">No sources were found for this claim.</p>
            </div>
          ) : (
            <>
              {/* Tabs with counts - hide empty tabs */}
              <div className="sources-tabs-nav" role="tablist" aria-label="Filter evidence by stance">
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeTab === 'all'}
                  className={`source-tab-btn ${activeTab === 'all' ? 'active' : ''}`}
                  onClick={() => setActiveTab('all')}
                >
                  All ({counts.all})
                </button>

                {counts.support > 0 && (
                  <button
                    type="button"
                    role="tab"
                    aria-selected={activeTab === 'support'}
                    className={`source-tab-btn ${activeTab === 'support' ? 'active' : ''}`}
                    onClick={() => setActiveTab('support')}
                  >
                    Supports ({counts.support})
                  </button>
                )}

                {counts.contradict > 0 && (
                  <button
                    type="button"
                    role="tab"
                    aria-selected={activeTab === 'contradict'}
                    className={`source-tab-btn ${activeTab === 'contradict' ? 'active' : ''}`}
                    onClick={() => setActiveTab('contradict')}
                  >
                    Contradicts ({counts.contradict})
                  </button>
                )}

                {counts.unclear > 0 && (
                  <button
                    type="button"
                    role="tab"
                    aria-selected={activeTab === 'unclear'}
                    className={`source-tab-btn ${activeTab === 'unclear' ? 'active' : ''}`}
                    onClick={() => setActiveTab('unclear')}
                  >
                    Unclear ({counts.unclear})
                  </button>
                )}
              </div>

              {/* Source Rows */}
              <div className="sources-list" role="region" aria-label="Evidence sources list">
                {filteredEvidence.map((src, index) => (
                  <SourceRow key={`${src.source}-${index}`} sourceData={src} />
                ))}
              </div>
            </>
          )}
        </section>
      </div>
    </article>
  );
}
