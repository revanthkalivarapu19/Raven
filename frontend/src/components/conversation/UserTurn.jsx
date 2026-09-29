import { useState } from 'react'
import { formatClock, formatLongDate } from '../../lib/format'
import ImageLightbox from '../ImageLightbox'

/**
 * What the user said.
 *
 * Presented as a pull-quote in display type rather than a chat bubble: the
 * claim being checked is the subject of the document, so it is set as a
 * headline. Longer pastes drop to reading size so a forwarded message does
 * not fill the screen.
 */
export default function UserTurn({ turn }) {
  const { text, image, at } = turn
  const [zoomed, setZoomed] = useState(false)
  const isLong = text && text.length > 110

  return (
    <article className="turn turn--you">
      <header className="turn__meta">
        <span className="folio">You</span>
        <time className="turn__time" dateTime={at} title={formatLongDate(at)}>
          {formatClock(at)}
        </time>
      </header>

      {image && (
        <figure className="turn__attachment">
          <button
            type="button"
            className="turn__attachment-button"
            onClick={() => setZoomed(true)}
            aria-label="View the attached image at full size"
          >
            <img
              src={image.dataUrl}
              alt={image.name ? `Attached image: ${image.name}` : 'Attached image'}
            />
          </button>
        </figure>
      )}

      {text && <p className={`turn__claim ${isLong ? 'turn__claim--long' : ''}`}>{text}</p>}

      {zoomed && <ImageLightbox image={image} onClose={() => setZoomed(false)} />}
    </article>
  )
}
