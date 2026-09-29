import Wordmark from '../Wordmark'
import { formatClock } from '../../lib/format'

/**
 * A failure, kept in the conversation.
 *
 * An error is part of the record, not a dialog that overwrites what was there.
 * It says what happened in plain words, reassures that nothing was lost, and
 * offers the one action that helps: run it again.
 */
export default function ErrorTurn({ turn, onRetry, isRetrying }) {
  return (
    <article className="turn turn--raven turn--error">
      <header className="turn__head">
        <Wordmark size="sm" />
        <span className="turn__rule" aria-hidden="true" />
        <span className="folio turn__failed">Not completed</span>
        <time className="turn__time">{formatClock(turn.at)}</time>
      </header>

      <p className="turn__lead">{turn.message}</p>

      <div className="turn__prose">
        <p>
          Nothing you sent has been lost, and checking it again costs nothing. If it
          keeps failing, the sources may be temporarily unreachable.
        </p>
      </div>

      {turn.retryable && (
        <div className="turn__actions">
          <button
            type="button"
            className="btn btn--quiet btn--sm"
            onClick={() => onRetry(turn)}
            disabled={isRetrying}
          >
            Try again
          </button>
        </div>
      )}
    </article>
  )
}
