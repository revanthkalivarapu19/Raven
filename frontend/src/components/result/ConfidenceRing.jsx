import { formatPercent } from '../../lib/format'

const TAU = Math.PI * 2

/**
 * Confidence, as a ring.
 *
 * The arc's length *is* the number: at 93% the ring is nearly closed, at 34%
 * it is a little over a third. That relationship is the whole point — a bar
 * reads as an amount of something, a ring reads as a measured value, and it
 * sits next to the verdict without becoming a gauge cluster.
 *
 * The number is real, selectable text rather than part of the drawing, so it
 * survives a screen reader and can be copied. The SVG itself is decorative —
 * the level word is rendered by the verdict block beside it, once.
 */
export default function ConfidenceRing({ value = 0, size = 92, stroke = 4 }) {
  const ratio = Math.min(1, Math.max(0, Number(value) || 0))
  const radius = (size - stroke) / 2
  const circumference = TAU * radius
  const offset = circumference * (1 - ratio)
  const centre = size / 2

  return (
    // `--ring-full` is the whole circumference. The stylesheet animates the arc
    // from there down to its real offset, so the ring draws itself to the
    // reading instead of simply appearing at it.
    <div className="ring" style={{ '--ring-size': `${size}px`, '--ring-full': circumference }}>
      <svg
        className="ring__svg"
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        aria-hidden="true"
        focusable="false"
      >
        <circle
          className="ring__track"
          cx={centre}
          cy={centre}
          r={radius}
          strokeWidth={stroke}
          fill="none"
        />
        {ratio > 0 && (
          <circle
            className="ring__arc"
            cx={centre}
            cy={centre}
            r={radius}
            strokeWidth={stroke}
            fill="none"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        )}
      </svg>

      <span className="ring__centre">
        <span className="u-visually-hidden">Confidence </span>
        <span className="ring__value">{formatPercent(ratio)}</span>
      </span>
    </div>
  )
}
