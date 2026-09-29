/**
 * The RAVEN mark is pure typography — a serif wordmark with wide tracking.
 * No bird, no glyph, no glow. The identity comes from type and composition.
 */
export default function Wordmark({ size = 'md' }) {
  return (
    <span className={`wordmark wordmark--${size}`} aria-label="RAVEN">
      Raven
    </span>
  )
}
