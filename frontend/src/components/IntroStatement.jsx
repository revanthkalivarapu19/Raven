/**
 * The product making its case, in type. This is the first thing a new
 * visitor reads, so it carries the identity instead of a graphic.
 * Reused as the opening of the empty state.
 */
export default function IntroStatement() {
  return (
    <section className="intro">
      <span className="folio">
        <span className="folio__tick" />
        Evidence, not opinion
      </span>

      <h1 className="intro__title">Check a claim against the evidence.</h1>

      <p className="intro__lead">
        Paste a headline, describe what you saw, or share a screenshot. RAVEN looks for
        sources that speak to it, then shows you what they actually say.
      </p>
    </section>
  )
}
