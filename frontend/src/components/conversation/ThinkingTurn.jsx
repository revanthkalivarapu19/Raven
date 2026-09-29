import { useEffect, useState } from 'react'
import Wordmark from '../Wordmark'
import { stagesFor } from '../../lib/verdict'

/**
 * Progress while an answer is being produced.
 *
 * Instead of a spinner, a short line of plain-language stages that settle on
 * the last one. It tells the reader what is happening — and the sweeping rule
 * gives the wait a sense of length without a fake progress percentage.
 */
export default function ThinkingTurn({ kind = 'assessment' }) {
  const stages = stagesFor(kind)
  const [index, setIndex] = useState(0)

  useEffect(() => {
    if (stages.length < 2) return undefined
    const timer = setInterval(() => {
      setIndex((current) => (current < stages.length - 1 ? current + 1 : current))
    }, 1500)
    return () => clearInterval(timer)
  }, [stages.length])

  return (
    <article className="turn turn--raven turn--thinking" aria-busy="true">
      <header className="turn__head">
        <Wordmark size="sm" />
        <span className="turn__rule" aria-hidden="true" />
      </header>

      <div className="thinking">
        <p className="thinking__stage" aria-live="polite">
          {stages[index]}
          <span className="thinking__ellipsis" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
        </p>
        <span className="thinking__track" aria-hidden="true">
          <span className="thinking__sweep" />
        </span>
      </div>
    </article>
  )
}
