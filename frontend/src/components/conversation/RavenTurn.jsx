import Wordmark from '../Wordmark'
import FollowUps from './FollowUps'
import EvidenceSection from '../result/EvidenceSection'
import { formatClock, formatLongDate } from '../../lib/format'
import { explanationLabel } from '../../lib/verdict'

/**
 * What RAVEN answers.
 *
 * Structure is deliberately document-like: a quiet speaker line, then prose at
 * reading size under a label that says what the section is. The verdict block
 * and the evidence sit above and below the reasoning, so this component stays
 * about the shape of an answer rather than the shape of a result.
 */
export default function RavenTurn({
  turn,
  isLatest,
  onFollowUp,
  onAboutSource,
  priorEvidence,
  children,
}) {
  const { at, intro, reasoning, note, followUps, evidence } = turn
  const isAssessment = turn.kind === 'assessment'

  return (
    <article className="turn turn--raven">
      <header className="turn__head">
        <Wordmark size="sm" />
        <span className="turn__rule" aria-hidden="true" />
        <time className="turn__time" dateTime={at} title={formatLongDate(at)}>
          {formatClock(at)}
        </time>
      </header>

      {/* The verdict block — rendered by the parent, so this component stays
          about the shape of an answer rather than the shape of a result. */}
      {children}

      {/* For an assessment the lead is rendered by the verdict block itself,
          so it sits directly under the word it qualifies. */}
      {intro && !isAssessment && <p className="turn__lead">{intro}</p>}

      {reasoning?.length > 0 && (
        <section className="explanation" aria-label={explanationLabel(turn.kind)}>
          <span className="folio explanation__label">{explanationLabel(turn.kind)}</span>
          <div className="turn__prose">
            {reasoning.map((paragraph, index) => (
              // Prose paragraphs are stable per turn and never reordered.
              <p key={index}>{paragraph}</p>
            ))}
          </div>
        </section>
      )}

      {evidence?.length > 0 && (
        <EvidenceSection
          evidence={evidence}
          priorEvidence={priorEvidence}
          onAboutSource={onAboutSource}
          heading={turn.kind === 'evidence' ? 'Further sources' : 'Sources'}
        />
      )}

      {note && <p className="turn__note">{note}</p>}

      {followUps?.length > 0 && (
        <FollowUps
          items={followUps}
          onSelect={onFollowUp}
          onAboutSource={onAboutSource}
          quiet={!isLatest}
        />
      )}
    </article>
  )
}
