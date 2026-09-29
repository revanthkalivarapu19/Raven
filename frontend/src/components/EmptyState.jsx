import IntroStatement from './IntroStatement'

/**
 * The first screen.
 *
 * The proposition, then three real-shaped examples of the kind of thing
 * someone might bring. Examples rather than feature bullets: they show what a
 * claim looks like here without explaining the product.
 */
export default function EmptyState({ starters, onPick }) {
  return (
    <div className="empty">
      <IntroStatement />

      {starters.length > 0 && (
        <section className="empty__examples" aria-label="Example claims">
          <span className="folio empty__examples-label">Try one of these</span>
          <ul className="empty__list">
            {starters.map((starter) => (
              <li key={starter.id}>
                <button
                  type="button"
                  className="example"
                  onClick={() => onPick(starter.label)}
                >
                  <span className="example__text">{starter.label}</span>
                  <span className="example__detail">{starter.detail}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}
