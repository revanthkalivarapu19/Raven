import '../styles/design-system.css'
import ThemeToggle from '../components/ThemeToggle'
import ConfidenceRing from '../components/result/ConfidenceRing'
import { confidenceReading } from '../lib/verdict'

/**
 * Internal design-primitives review. Not part of the user-facing product —
 * it exists so the type scale, palette and controls can be checked in one
 * place while the interface is built. Reachable at `#design`.
 */

const TYPE_SPECIMENS = [
  { label: 'Hero', token: '--text-hero', sample: 'What the evidence shows', style: { fontFamily: 'var(--font-display)', fontSize: 'var(--text-hero)', lineHeight: 'var(--leading-hero)', letterSpacing: 'var(--tracking-hero)' } },
  { label: 'Display', token: '--text-display', sample: 'Here’s what I found', style: { fontFamily: 'var(--font-display)', fontSize: 'var(--text-display)', lineHeight: 'var(--leading-tight)', letterSpacing: 'var(--tracking-tight)' } },
  { label: 'Title', token: '--text-title', sample: 'The available evidence', style: { fontFamily: 'var(--font-display)', fontSize: 'var(--text-title)', lineHeight: 'var(--leading-snug)' } },
  { label: 'Lead', token: '--text-lead', sample: 'Two primary sources agree on the same sequence of events.', style: { fontSize: 'var(--text-lead)', lineHeight: 'var(--leading-body)', color: 'var(--ink-secondary)' } },
  { label: 'Body', token: '--text-body', sample: 'Anyone can check a claim here. Paste a headline, describe something you saw, or share a screenshot, and RAVEN will look for what the sources actually say.', style: { fontSize: 'var(--text-body)', lineHeight: 'var(--leading-body)' } },
  { label: 'UI', token: '--text-ui', sample: 'Find more evidence', style: { fontSize: 'var(--text-ui)' } },
  { label: 'Small', token: '--text-small', sample: 'Updated 4 days ago · 2 sources', style: { fontSize: 'var(--text-small)', color: 'var(--ink-tertiary)' } },
  { label: 'Folio', token: '--text-micro', sample: 'EVIDENCE — 01 / 04', style: { fontFamily: 'var(--font-mono)', fontSize: 'var(--text-micro)', letterSpacing: 'var(--tracking-label)', textTransform: 'uppercase', color: 'var(--ink-tertiary)' } },
]

const SWATCHES = [
  { name: 'Page', token: '--paper', varName: '--paper' },
  { name: 'Raised', token: '--paper-raised', varName: '--paper-raised' },
  { name: 'Tint', token: '--paper-tint', varName: '--paper-tint' },
  { name: 'Sunken', token: '--paper-sunken', varName: '--paper-sunken' },
  { name: 'Ink', token: '--ink-primary', varName: '--ink-primary' },
  { name: 'Ink 2', token: '--ink-secondary', varName: '--ink-secondary' },
  { name: 'Ink 3', token: '--ink-tertiary', varName: '--ink-tertiary' },
  { name: 'Accent', token: '--accent', varName: '--accent' },
]

const SPACE_STEPS = ['--space-1', '--space-2', '--space-3', '--space-4', '--space-5', '--space-6', '--space-7', '--space-8', '--space-9', '--space-10']

/** The four confidence readings the product actually produces, plus extremes. */
const CONFIDENCE_SAMPLES = [
  { value: 0.97, verdict: 'Fake' },
  { value: 0.93, verdict: 'Real' },
  { value: 0.42, verdict: 'Unverified' },
  { value: 0.08, verdict: 'Unverified' },
  { value: 0, verdict: null },
]

const VERDICT_CLASS = { Real: 'real', Fake: 'fake', Unverified: 'unverified' }

export default function DesignSystem() {
  return (
    <div className="ds">
      <div className="ds__inner">
        <header className="ds__masthead">
          <div className="ds__masthead-top">
            <span className="folio">
              <span className="folio__tick" />
              Internal · Development only
            </span>
            <ThemeToggle />
          </div>
          <h1>Design primitives</h1>
          <p>
            The shared foundation the interface is assembled from. Editorial display
            type, warm paper, hairline rules and a single accent — checked here in one
            place before they are used in the conversation. Both themes live in
            <code> tokens.css</code>; this page follows whichever is switched on.
          </p>
        </header>

        <section className="ds__section">
          <span className="folio">01 — Type scale</span>
          <div>
            {TYPE_SPECIMENS.map((spec) => (
              <div className="ds__specimen" key={spec.label}>
                <span className="folio">{spec.label}</span>
                <div className="ds__specimen-row">
                  <span style={spec.style}>{spec.sample}</span>
                  <span className="ds__note">{spec.token}</span>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="ds__section">
          <span className="folio">02 — Paper &amp; ink</span>
          <div className="ds__grid">
            {SWATCHES.map((swatch) => (
              <div className="ds__swatch" key={swatch.name}>
                <div className="ds__swatch-chip" style={{ background: `var(${swatch.varName})` }} />
                <div className="ds__swatch-meta">
                  <strong style={{ fontSize: 'var(--text-ui)' }}>{swatch.name}</strong>
                  <code>{swatch.token}</code>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="ds__section">
          <span className="folio">03 — Verdicts</span>
          <div className="ds__grid">
            <div className="ds__verdict ds__verdict--real">
              <span className="dot" />
              <strong>Real</strong>
            </div>
            <div className="ds__verdict ds__verdict--fake">
              <span className="dot" />
              <strong>Fake</strong>
            </div>
            <div className="ds__verdict ds__verdict--unverified">
              <span className="dot dot--hollow" />
              <strong>Unverified</strong>
            </div>
          </div>
          <p className="ds__note" style={{ marginTop: 'var(--space-4)' }}>
            Verdict colour is the only saturated colour in the product, so it always
            means something.
          </p>
        </section>

        <section className="ds__section">
          <span className="folio">04 — Confidence ring</span>
          <div className="ds__rings">
            {CONFIDENCE_SAMPLES.map((sample) => (
              <div
                className={`ds__ring-sample ${sample.verdict ? `assessment--${VERDICT_CLASS[sample.verdict]}` : ''}`}
                key={sample.value}
              >
                <ConfidenceRing value={sample.value} size={92} stroke={3.5} />
                <div>
                  <p className="ds__ring-level">
                    {confidenceReading(sample.value, sample.verdict).level} confidence
                  </p>
                  <span className="ds__note">{sample.value.toFixed(2)}</span>
                </div>
              </div>
            ))}
          </div>
          <p className="ds__note" style={{ marginTop: 'var(--space-4)' }}>
            The arc length is the number, and the ring takes its colour from the
            verdict — so confidence never reads as a free-floating score.
          </p>
        </section>

        <section className="ds__section">
          <span className="folio">05 — Buttons &amp; states</span>
          <div className="ds__stack">
            <div className="ds__row">
              <button className="btn btn--primary">Check this claim</button>
              <button className="btn btn--accent">Send</button>
              <button className="btn btn--quiet">Find more evidence</button>
              <button className="btn btn--bare">Cancel</button>
              <button className="btn btn--quiet" disabled>
                Disabled
              </button>
              <button className="btn btn--quiet is-loading">Checking</button>
            </div>
            <div className="ds__row">
              <button className="btn btn--quiet btn--sm">Small</button>
              <button className="btn btn--quiet">Default</button>
              <button className="btn btn--quiet btn--lg">Large</button>
            </div>
          </div>
        </section>

        <section className="ds__section">
          <span className="folio">06 — Fields</span>
          <div className="ds__grid">
            <label className="ds__stack" style={{ gap: 'var(--space-2)' }}>
              <span className="folio">Default</span>
              <input className="field" placeholder="Paste a headline or ask a question" />
            </label>
            <label className="ds__stack" style={{ gap: 'var(--space-2)' }}>
              <span className="folio">Invalid</span>
              <input className="field" aria-invalid="true" defaultValue="unreachable source" />
            </label>
          </div>
        </section>

        <section className="ds__section">
          <span className="folio">07 — Chips, tags &amp; source stances</span>
          <div className="ds__stack">
            <div className="ds__row">
              <button className="chip">Why this verdict?</button>
              <button className="chip">Find more evidence</button>
              <button className="chip chip--accent">Check another claim</button>
            </div>
            <div className="ds__row">
              <span className="tag">Real</span>
              <span className="tag tag--accent">High confidence</span>
              <span className="tag">4 sources</span>
            </div>
            <div className="ds__row">
              <span className="evidence__stance evidence__stance--supports">
                <span className="dot" />
                Supports the claim
              </span>
              <span className="evidence__stance evidence__stance--contradicts">
                <span className="dot" />
                Contradicts the claim
              </span>
              <span className="evidence__stance evidence__stance--unclear">
                <span className="dot dot--hollow" />
                Doesn’t settle it
              </span>
            </div>
          </div>
        </section>

        <section className="ds__section">
          <span className="folio">08 — Spacing scale</span>
          <div className="ds__ruler">
            {SPACE_STEPS.map((step) => (
              <div className="ds__ruler-row" key={step}>
                <span className="ds__ruler-label">{step}</span>
                <span className="ds__ruler-bar" style={{ width: `var(${step})` }} />
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}
