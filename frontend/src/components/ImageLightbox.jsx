import { useEffect } from 'react'
import { X } from 'lucide-react'

/** Shows an attached image at full size. Escape or a click closes it. */
export default function ImageLightbox({ image, onClose }) {
  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  return (
    <div className="lightbox" onClick={onClose} role="dialog" aria-modal="true" aria-label="Attached image">
      <button type="button" className="btn btn--icon lightbox__close" aria-label="Close" autoFocus>
        <X size={18} strokeWidth={1.7} />
      </button>
      <img src={image.dataUrl} alt={image.name ? `Attached image: ${image.name}` : 'Attached image'} />
      {image.name && <span className="lightbox__name">{image.name}</span>}
    </div>
  )
}
