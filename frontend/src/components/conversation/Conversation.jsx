import ErrorTurn from './ErrorTurn'
import RavenTurn from './RavenTurn'
import ThinkingTurn from './ThinkingTurn'
import UserTurn from './UserTurn'
import AssessmentCard from '../result/AssessmentCard'

/** Walks the conversation once, giving each answer:
 *  - the user turn it answers, so a result block can tell whether RAVEN's
 *    reading differs from what was actually written
 *  - the ids of sources already shown earlier, so a wider search can mark what
 *    is genuinely new instead of appearing to discover the same source twice */
function prepare(turns) {
  const items = []
  const shown = new Set()
  let source = null

  for (const turn of turns) {
    if (turn.role === 'user') {
      source = turn
      items.push({ turn, source: null, priorEvidence: shown })
      continue
    }

    items.push({ turn, source, priorEvidence: new Set(shown) })
    turn.evidence?.forEach((item) => shown.add(item.id))
  }

  return items
}

/**
 * The conversation, oldest first.
 *
 * Rendered as one editorial sequence rather than a list of chat bubbles. Only
 * the newest answer offers suggestions at full strength; everything above it
 * stays exactly as it was, including the result of an earlier check.
 */
export default function Conversation({
  turns,
  isThinking,
  thinkingKind,
  onFollowUp,
  onAboutSource,
  onRetry,
}) {
  const lastRavenId = [...turns].reverse().find((turn) => turn.role === 'raven')?.id

  return (
    <div className="conversation">
      {prepare(turns).map(({ turn, source, priorEvidence }) => {
        if (turn.role === 'user') {
          return <UserTurn key={turn.id} turn={turn} />
        }

        if (turn.kind === 'error') {
          return (
            <ErrorTurn key={turn.id} turn={turn} onRetry={onRetry} isRetrying={isThinking} />
          )
        }

        return (
          <RavenTurn
            key={turn.id}
            turn={turn}
            isLatest={turn.id === lastRavenId}
            onFollowUp={onFollowUp}
            onAboutSource={onAboutSource}
            priorEvidence={priorEvidence}
          >
            {turn.kind === 'assessment' && (
              <AssessmentCard
                turn={turn}
                sourceText={source?.text ?? ''}
                fromImage={Boolean(source?.image)}
              />
            )}
          </RavenTurn>
        )
      })}

      {isThinking && <ThinkingTurn kind={thinkingKind} />}
    </div>
  )
}
