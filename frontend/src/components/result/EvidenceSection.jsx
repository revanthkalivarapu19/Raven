import { useState } from 'react'
import { ChevronDown, ExternalLink } from 'lucide-react'
import { formatAge, formatDomain, formatLongDate } from '../../lib/format'
import { stanceLabel } from '../../lib/verdict'

/**
 * The sources behind an answer.
 *
 * Readable first, technical second. Each source is one row of a small table —
 * index, publisher, what it says, and what it says about the *claim* — so
 * several sources can be compared down the page. Passages stay folded to three
 * lines until someone asks for them, which is what keeps a list of five sources
 * from turning into a wall of text.
 */
export default function EvidenceSection({
  evidence = [],
  priorEvidence = new Set(),
  onAboutSource,
  heading = 'Sources',
}) {
  const [open, setOpen] = useState(() => new Set())

  if (evidence.length === 0) return null

  const toggle = (id) =>
    setOpen((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  const allOpen = open.size === evidence.length

  return (
    <section className="evidence">
      <header className="evidence__head">
        <span className="folio">
          {heading} — {String(evidence.length).padStart(2, '0')}
        </span>
        <button type="button" className="evidence__all" onClick={() => setOpen(allOpen ? new Set() : new Set(evidence.map((item) => item.id)))}>
          {allOpen ? 'Fold all passages' : 'Open all passages'}
        </button>
      </header>

      <ol className="evidence__list">
        {evidence.map((item, index) => {
          const isOpen = open.has(item.id)
          const isRepeat = priorEvidence.has(item.id)

          return (
            <li className={`evidence__item evidence__item--${item.stance}`} key={item.id}>
              <span className="evidence__index">{String(index + 1).padStart(2, '0')}</span>

              <div className="evidence__body">
                <div className="evidence__byline">
                  <span className="evidence__publisher">{item.publisher}</span>
                  <span className="evidence__kind">{item.kind}</span>
                  {isRepeat && <span className="evidence__repeat">Already shown</span>}
                </div>

                <a
                  className="evidence__title"
                  href={item.url}
                  target="_blank"
                  rel="noreferrer noopener"
                >
                  {item.title}
                  <ExternalLink size={12} strokeWidth={1.8} className="evidence__title-icon" />
                </a>

                <blockquote className={`evidence__excerpt ${isOpen ? 'is-open' : ''}`}>
                  {item.excerpt}
                </blockquote>

                <div className="evidence__actions">
                  <button
                    type="button"
                    className="evidence__toggle"
                    onClick={() => toggle(item.id)}
                    aria-expanded={isOpen}
                  >
                    {isOpen ? 'Fold the passage' : 'Read the passage'}
                    <ChevronDown
                      size={13}
                      strokeWidth={1.9}
                      className={`evidence__chevron ${isOpen ? 'is-open' : ''}`}
                    />
                  </button>

                  {onAboutSource && (
                    <button
                      type="button"
                      className="evidence__toggle evidence__toggle--quiet"
                      onClick={() => onAboutSource(item)}
                    >
                      What is this source?
                    </button>
                  )}
                </div>
              </div>

              <div className="evidence__meta">
                <span className={`evidence__stance evidence__stance--${item.stance}`}>
                  <span className={item.stance === 'unclear' ? 'dot dot--hollow' : 'dot'} />
                  {stanceLabel(item.stance)}
                </span>
                <time className="evidence__date" dateTime={item.published} title={formatLongDate(item.published)}>
                  {formatAge(item.published)}
                </time>
                <span className="evidence__domain">{formatDomain(item.url)}</span>
              </div>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
