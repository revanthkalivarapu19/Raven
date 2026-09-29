import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react'
import { ArrowUp, ImagePlus, X } from 'lucide-react'
import { ImageError, firstImageFile, readImageFile } from '../lib/image'

/**
 * Where a claim is written.
 *
 * One field, one attachment, one action. It grows with the text, accepts an
 * image by button, drag or paste, and shows what will be sent before it is
 * sent. It stays this size on purpose — the brief asks for calm, not a toolbar.
 */
const Composer = forwardRef(function Composer(
  { onSend, disabled = false, placeholder, busy = false },
  ref,
) {
  const [text, setText] = useState('')
  const [image, setImage] = useState(null)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [reading, setReading] = useState(false)

  const textareaRef = useRef(null)
  const fileRef = useRef(null)

  useImperativeHandle(ref, () => ({
    focus: () => textareaRef.current?.focus(),
  }))

  // Grow with the content, up to a point, then scroll.
  useEffect(() => {
    const element = textareaRef.current
    if (!element) return
    element.style.height = 'auto'
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`
  }, [text])

  const attach = async (file) => {
    if (!file) return
    setError(null)
    setReading(true)
    try {
      setImage(await readImageFile(file))
    } catch (caught) {
      setError(
        caught instanceof ImageError
          ? caught.userMessage
          : 'That image could not be attached.',
      )
    } finally {
      setReading(false)
    }
  }

  /**
   * Drag anywhere in the app, not just over this small field — dragging a file
   * onto a 200px composer to have the browser open it instead is a bad trade.
   */
  useEffect(() => {
    const onDragOver = (event) => {
      if (!event.dataTransfer?.types?.includes('Files')) return
      event.preventDefault()
      setDragging(true)
    }
    const onDragLeave = (event) => {
      if (event.relatedTarget === null) setDragging(false)
    }
    const onDrop = (event) => {
      if (!event.dataTransfer?.types?.includes('Files')) return
      event.preventDefault()
      setDragging(false)
      attach(firstImageFile(event.dataTransfer.files))
    }

    window.addEventListener('dragover', onDragOver)
    window.addEventListener('dragleave', onDragLeave)
    window.addEventListener('drop', onDrop)
    return () => {
      window.removeEventListener('dragover', onDragOver)
      window.removeEventListener('dragleave', onDragLeave)
      window.removeEventListener('drop', onDrop)
    }
  }, [])

  const submit = () => {
    const value = text.trim()
    // While a check is running the message is neither sent nor discarded: the
    // field keeps its contents so nothing is lost by a stray Enter.
    if ((!value && !image) || disabled || reading || busy) return
    onSend({ text: value, image })
    setText('')
    setImage(null)
    setError(null)
  }

  const onKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  const onPaste = (event) => {
    const file = firstImageFile(event.clipboardData?.files)
    if (file) {
      event.preventDefault()
      attach(file)
    }
  }

  return (
    <form
      className="composer"
      onSubmit={(event) => {
        event.preventDefault()
        submit()
      }}
    >
      {image && (
        <div className="composer__preview">
          <img src={image.dataUrl} alt="" />
          <div className="composer__preview-meta">
            <span className="composer__preview-name">{image.name}</span>
            <span className="composer__preview-size">
              {image.width} × {image.height}
            </span>
          </div>
          <button
            type="button"
            className="composer__preview-remove"
            onClick={() => setImage(null)}
            aria-label={`Remove ${image.name}`}
          >
            <X size={14} strokeWidth={2} />
          </button>
        </div>
      )}

      <div className="composer__row">
        <button
          type="button"
          className="composer__attach"
          onClick={() => fileRef.current?.click()}
          aria-label="Attach an image"
          title="Attach an image"
          disabled={disabled || reading}
        >
          <ImagePlus size={17} strokeWidth={1.7} />
        </button>

        <label className="u-visually-hidden" htmlFor="raven-composer">
          Write a claim or a question
        </label>
        <textarea
          id="raven-composer"
          ref={textareaRef}
          className="composer__input"
          rows={1}
          value={text}
          placeholder={image ? 'Add a note about this image (optional)' : placeholder}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={onKeyDown}
          onPaste={onPaste}
          disabled={disabled}
        />

        <button
          type="submit"
          className="composer__send"
          disabled={disabled || reading || busy || (!text.trim() && !image)}
          aria-label={image ? 'Check this image' : 'Check this claim'}
          title={
            busy
              ? 'Wait for the current check to finish'
              : image
                ? 'Check this image'
                : 'Check this claim'
          }
        >
          <ArrowUp size={17} strokeWidth={2} />
        </button>
      </div>

      <input
        ref={fileRef}
        className="u-visually-hidden"
        type="file"
        accept="image/png,image/jpeg,image/webp,image/gif"
        onChange={(event) => {
          attach(event.target.files?.[0])
          event.target.value = ''
        }}
        tabIndex={-1}
        aria-hidden="true"
      />

      {error && (
        <p className="composer__error" role="alert">
          {error}
        </p>
      )}

      {dragging && (
        <div className="composer__drop" aria-hidden="true">
          <span>Drop the image here</span>
        </div>
      )}

      {busy && <span className="u-visually-hidden" aria-live="polite">Checking your claim</span>}
    </form>
  )
})

export default Composer
