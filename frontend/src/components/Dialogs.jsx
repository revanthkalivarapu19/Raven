import { useEffect, useRef } from 'react';
import { CloseIcon } from './Icons';

export function WhyResultDialog({ isOpen, onClose, result, triggerRef }) {
  const dialogRef = useRef(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (isOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
    } else {
      if (dialog.open) {
        dialog.close();
      }
    }
  }, [isOpen]);

  const handleClose = () => {
    onClose();
    if (triggerRef?.current) {
      triggerRef.current.focus();
    }
  };

  const evidence = result?.evidence || [];
  const supportsCount = evidence.filter((e) => e.stance === 'support').length;
  const contradictsCount = evidence.filter((e) => e.stance === 'contradict').length;
  const sourcesChecked = evidence.length;
  const avgTrust =
    sourcesChecked > 0
      ? Math.round(evidence.reduce((acc, e) => acc + (e.trust || 0), 0) / sourcesChecked)
      : 0;

  return (
    <dialog
      ref={dialogRef}
      className="native-app-dialog why-result-dialog"
      onCancel={handleClose}
      aria-labelledby="why-result-title"
    >
      <div className="dialog-content">
        <div className="dialog-header">
          <h2 id="why-result-title" className="dialog-title">Why this result?</h2>
          <button
            type="button"
            className="btn-dialog-close"
            onClick={handleClose}
            aria-label="Close dialog"
          >
            <CloseIcon size={16} />
          </button>
        </div>

        <div className="dialog-body">
          <p className="dialog-intro-text">
            This verdict is calculated from evidence gathered across independent, credible sources.
          </p>

          <div className="dialog-metrics-grid">
            <div className="dialog-metric-card">
              <span className="dialog-metric-value">{sourcesChecked}</span>
              <span className="dialog-metric-label">Sources checked</span>
            </div>
            <div className="dialog-metric-card">
              <span className="dialog-metric-value">{supportsCount}</span>
              <span className="dialog-metric-label">Supporting</span>
            </div>
            <div className="dialog-metric-card">
              <span className="dialog-metric-value">{contradictsCount}</span>
              <span className="dialog-metric-label">Contradicting</span>
            </div>
            <div className="dialog-metric-card">
              <span className="dialog-metric-value">{avgTrust}%</span>
              <span className="dialog-metric-label">Average source trust</span>
            </div>
          </div>

          <div className="dialog-explanation-section">
            <h3 className="dialog-subtitle">Assessment rationale</h3>
            <p className="dialog-body-text">{result?.why || 'No explanation provided.'}</p>
          </div>
        </div>

        <div className="dialog-footer">
          <button
            type="button"
            className="btn-dialog-primary"
            onClick={handleClose}
          >
            Close
          </button>
        </div>
      </div>
    </dialog>
  );
}

export function AboutDialog({ isOpen, onClose, triggerRef }) {
  const dialogRef = useRef(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (isOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
    } else {
      if (dialog.open) {
        dialog.close();
      }
    }
  }, [isOpen]);

  const handleClose = () => {
    onClose();
    if (triggerRef?.current) {
      triggerRef.current.focus();
    }
  };

  return (
    <dialog
      ref={dialogRef}
      className="native-app-dialog about-dialog"
      onCancel={handleClose}
      aria-labelledby="about-dialog-title"
    >
      <div className="dialog-content">
        <div className="dialog-header">
          <h2 id="about-dialog-title" className="dialog-title">About RAVEN</h2>
          <button
            type="button"
            className="btn-dialog-close"
            onClick={handleClose}
            aria-label="Close dialog"
          >
            <CloseIcon size={16} />
          </button>
        </div>

        <div className="dialog-body">
          <div className="about-dialog-section">
            <h3 className="dialog-subtitle">Evidence before opinion</h3>
            <p className="dialog-body-text">
              RAVEN (Real-time Agentic Verification & Evidence-based Fake News Detection) is an AI-assisted claim verification system designed to evaluate factual claims using relevant, traceable evidence. It supports text and screenshot-based inputs in English and Telugu.
            </p>
          </div>

          <div className="about-dialog-section">
            <h3 className="dialog-subtitle">How verification works</h3>
            <p className="dialog-body-text">
              RAVEN processes the submitted input, identifies the central claim, detects its domain, and searches for relevant evidence. It checks the local evidence collection first and can use external sources when additional or more recent evidence is needed. A multi-stage verification workflow evaluates evidence relevance, supporting and contradicting information, and whether the available evidence is sufficient.
            </p>
          </div>

          <div className="about-dialog-section">
            <h3 className="dialog-subtitle">Verification outcomes</h3>
            <p className="dialog-body-text">
              Results are categorised as:
            </p>
            <div className="outcomes-cards-list">
              <div className="outcome-item-card">
                <span className="outcome-badge-pill verdict-badge-real">Real</span>
                <span className="outcome-desc">The available evidence supports the claim.</span>
              </div>
              <div className="outcome-item-card">
                <span className="outcome-badge-pill verdict-badge-fake">Fake</span>
                <span className="outcome-desc">The available evidence contradicts the claim.</span>
              </div>
              <div className="outcome-item-card">
                <span className="outcome-badge-pill verdict-badge-unverified">Unverified</span>
                <span className="outcome-desc">The available evidence is insufficient or inconclusive.</span>
              </div>
            </div>
            <p className="dialog-caveat-text">
              Verdicts reflect the strength of the gathered evidence and are not assertions of certainty beyond what the evidence supports.
            </p>
          </div>

          <div className="about-dialog-section">
            <h3 className="dialog-subtitle">Transparency and responsible use</h3>
            <p className="dialog-body-text">
              RAVEN aims to make verification more transparent by presenting relevant evidence, source information, and explanations behind its conclusions. Results should be interpreted in the context of the available evidence, and users should consult the original sources for important decisions.
            </p>
          </div>
        </div>

        <div className="dialog-footer">
          <button
            type="button"
            className="btn-dialog-primary"
            onClick={handleClose}
          >
            Close
          </button>
        </div>
      </div>
    </dialog>
  );
}
