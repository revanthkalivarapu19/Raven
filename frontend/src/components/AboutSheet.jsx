import { useEffect, useRef } from 'react'
import { X } from 'lucide-react'

/**
 * About — the only secondary surface in the product.
 *
 * It explains, in plain language, what RAVEN does, what the three answers
 * mean, and what the product is not. A verification tool has to be honest
 * about its own limits, so that is written into the interface rather than
 * buried in a tooltip.
 */
export default function AboutSheet({ onClose }) {
  const panelRef = useRef(null)
  const closeRef = useRef(null)

  useEffect(() => {
    const previouslyFocused = document.activeElement
    closeRef.current?.focus()

    const onKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.stopPropagation()
        onClose()
        return
      }

      // Keep focus inside the dialog while it is open.
      if (event.key === 'Tab' && panelRef.current) {
        const focusable = panelRef.current.querySelectorAll(
          'button, [href], input, textarea, select, [tabindex]:not([tabindex="-1"])',
        )
        if (focusable.length === 0) return
        const first = focusable[0]
        const last = focusable[focusable.length - 1]

        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault()
          last.focus()
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault()
          first.focus()
        }
      }
    }

    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      if (previouslyFocused instanceof HTMLElement) previouslyFocused.focus()
    }
  }, [onClose])

  return (
    <div className="sheet">
      <div className="sheet__scrim" onClick={onClose} aria-hidden="true" />

      <div
        className="sheet__panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="about-title"
        ref={panelRef}
      >
        <button
          type="button"
          className="btn btn--icon sheet__close"
          onClick={onClose}
          aria-label="Close"
          ref={closeRef}
        >
          <X size={17} strokeWidth={1.6} />
        </button>

        <h2 className="sheet__title" id="about-title">
          About RAVEN
        </h2>

        <section className="sheet__section">
          <h3>What this is</h3>
          <p>
            RAVEN checks a claim against evidence. You paste a headline, describe
            something you saw, or attach a screenshot, and it looks for sources that
            speak to it — then shows you which way they point and what they actually say.
          </p>
        </section>

        <section className="sheet__section">
          <h3>The three answers</h3>
          <dl className="sheet__verdicts">
            <div className="sheet__verdict sheet__verdict--real">
              <dt>
                <span className="dot" />
                Real
              </dt>
              <dd>Reliable sources state the same thing, and nothing credible contradicts it.</dd>
            </div>
            <div className="sheet__verdict sheet__verdict--fake">
              <dt>
                <span className="dot" />
                Fake
              </dt>
              <dd>Reliable sources contradict the claim, or it can be traced back to something fabricated.</dd>
            </div>
            <div className="sheet__verdict sheet__verdict--unverified">
              <dt>
                <span className="dot dot--hollow" />
                Unverified
              </dt>
              <dd>
                The evidence is thin, too recent, or genuinely divided. This is RAVEN
                saying it does not know — not that the claim is false.
              </dd>
            </div>
          </dl>
        </section>

        <section className="sheet__section">
          <h3>What an answer contains</h3>
          <p>
            Every result shows the claim as RAVEN understood it, a short account of the
            reasoning, how much confidence the evidence supports, and the sources
            themselves — with the passages that were used.
          </p>
          <p>
            You can ask why a conclusion was reached, ask for more evidence, or ask
            about a particular source. It stays in the same conversation.
          </p>
        </section>

        <section className="sheet__section">
          <h3>How to read it</h3>
          <p>
            Confidence is not certainty. A high reading means the sources agree, not
            that the answer is beyond question. The sources are always open — read them
            and judge for yourself.
          </p>
          <p>
            RAVEN is not an authority, and it does not replace official guidance from
            the organisations involved. Treat it as a starting point for your own
            reading.
          </p>
        </section>
      </div>
    </div>
  )
}
