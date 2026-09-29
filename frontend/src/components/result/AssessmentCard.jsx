import { confidenceReading, verdictMeta } from '../../lib/verdict'
import ConfidenceRing from './ConfidenceRing'

function normalise(value = '') {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim()
}

/**
 * The verdict block — the visual centre of a verification answer.
 *
 * Read top to bottom it answers, in order: what did the sources say, how sure
 * is that reading, and which claim was read. It is deliberately not a card and
 * not a dashboard: the authority comes from one word at display size, a ring
 * that makes the confidence figure mean something, and a hairline legend for
 * how the sources line up. Everything else in the answer stays prose.
 */
export default function AssessmentCard({ turn, sourceText = '', fromImage = false }) {
  const verdict = verdictMeta(turn.verdict)
  const confidence = confidenceReading(turn.confidence ?? 0, turn.verdict)
  const evidence = turn.evidence ?? []

  const balance = evidence.reduce(
    (acc, item) => ({ ...acc, [item.stance]: (acc[item.stance] ?? 0) + 1 }),
    { supports: 0, contradicts: 0, unclear: 0 },
  )
  const total = evidence.length

  // The claim as RAVEN read it. Shown when it differs from what was written,
  // or when the claim was read out of an image — otherwise it would just repeat
  // the user's own turn back at them.
  const interpreted = turn.claim && normalise(turn.claim) !== normalise(sourceText)
  const showInterpretation = Boolean(turn.claim) && (interpreted || fromImage)

  const tally = [
    balance.supports > 0 && `${balance.supports} support`,
    balance.contradicts > 0 && `${balance.contradicts} contradict`,
    balance.unclear > 0 && `${balance.unclear} don’t settle it`,
  ]
    .filter(Boolean)
    .join(' · ')

  return (
    <section className={`assessment assessment--${verdict.key}`} aria-label="Result">
      <header className="assessment__head">
        <span className="folio">The verdict</span>
        <div className="assessment__verdict">
          <h2 className="assessment__word">
            <span className={verdict.key === 'unverified' ? 'dot dot--hollow' : 'dot'} />
            {verdict.label}
          </h2>
          <p className="assessment__stance">{verdict.stance}</p>
        </div>

        {/* RAVEN's opening line belongs to the verdict, not to the reasoning
            below it — so it is set here, under the word it qualifies. */}
        {turn.intro && <p className="turn__lead assessment__lead">{turn.intro}</p>}
      </header>

      <div className="assessment__reading">
        <ConfidenceRing value={turn.confidence ?? 0} size={92} stroke={3.5} />
        <div className="assessment__reading-body">
          <p className="assessment__level">{confidence.level} confidence</p>
          <p className="assessment__note">{confidence.note}</p>
        </div>
      </div>

      {total > 0 && (
        <div className="assessment__sources">
          <span className="folio">How the sources line up</span>
          <div className="assessment__balance" aria-hidden="true">
            {['supports', 'contradicts', 'unclear'].map((stance) =>
              balance[stance] > 0 ? (
                <span
                  key={stance}
                  className={`assessment__balance-part assessment__balance-part--${stance}`}
                  style={{ flexGrow: balance[stance] }}
                />
              ) : null,
            )}
          </div>
          <p className="assessment__note assessment__tally">{tally}</p>
        </div>
      )}

      {showInterpretation && (
        <div className="assessment__claim">
          <span className="folio">
            {fromImage && !sourceText ? 'Read from the image' : 'As I understood it'}
          </span>
          <p>{turn.claim}</p>
        </div>
      )}
    </section>
  )
}
