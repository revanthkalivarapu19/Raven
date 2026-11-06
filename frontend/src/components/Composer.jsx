import { useRef, useEffect, useState } from 'react';
import { ImageIcon, SendIcon, CloseIcon } from './Icons';

export function Composer({
  text = '',
  setText = () => {},
  screenshot = null,
  onAttachScreenshot = () => {},
  onRemoveScreenshot = () => {},
  onSubmit = () => {},
  disabled = false,
}) {
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  // Auto-grow textarea up to ~140px
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    const newHeight = Math.min(ta.scrollHeight, 140);
    ta.style.height = `${newHeight}px`;
    ta.style.overflowY = ta.scrollHeight > 140 ? 'auto' : 'hidden';
  }, [text]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && (text.trim() || screenshot)) {
        onSubmit();
      }
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onAttachScreenshot(file);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handlePaste = (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) {
          onAttachScreenshot(file);
          break;
        }
      }
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onAttachScreenshot(file);
    }
  };

  const isSubmitDisabled = disabled || (!text.trim() && !screenshot);

  return (
    <div className="composer-wrapper">
      <div
        className={`composer-container ${isDragging ? 'dragging-active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {screenshot && (
          <div className="composer-attachment-preview">
            <div
              className="attachment-chip"
              role="group"
              aria-label={`Attached screenshot: ${screenshot.name || 'Screenshot'}`}
            >
              <div className="attachment-thumb-box">
                {screenshot.preview ? (
                  <img
                    src={screenshot.preview}
                    alt="Thumbnail preview"
                    className="attachment-thumb"
                  />
                ) : (
                  <ImageIcon size={14} className="attachment-fallback-icon" />
                )}
              </div>
              <div className="attachment-meta">
                <span className="attachment-filename">
                  {screenshot.name || 'Screenshot.png'}
                </span>
                <span className="attachment-tag">Screenshot attached</span>
              </div>
              <button
                type="button"
                className="btn-remove-attachment"
                onClick={onRemoveScreenshot}
                aria-label="Remove screenshot"
                title="Remove screenshot"
              >
                <CloseIcon size={12} />
              </button>
            </div>
          </div>
        )}

        <div className="composer-input-row">
          <textarea
            ref={textareaRef}
            className="composer-textarea"
            placeholder="Type or paste a claim to check..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            rows={1}
            aria-label="Claim text"
          />
        </div>

        <div className="composer-actions-row">
          <div className="composer-tools">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              style={{ display: 'none' }}
              id="screenshot-upload"
            />
            <button
              type="button"
              className={`btn-attach-screenshot ${screenshot ? 'has-attachment' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              title={screenshot ? 'Change attached screenshot' : 'Attach a claim screenshot (English or Telugu)'}
              aria-label={screenshot ? 'Change attached screenshot' : 'Attach a claim screenshot'}
            >
              <ImageIcon size={15} className="attach-btn-icon" />
              <span className="attach-btn-label">
                {screenshot ? 'Change screenshot' : 'Add screenshot'}
              </span>
            </button>
            <span className="composer-language-note">
              <span className="lang-indicator" aria-hidden="true" />
              English & Telugu supported
            </span>
          </div>

          <div className="composer-submit-group">
            <button
              type="button"
              className="btn-check-claim"
              onClick={onSubmit}
              disabled={isSubmitDisabled}
              aria-label="Check claim"
            >
              <SendIcon size={14} />
              <span>Check claim</span>
            </button>
          </div>
        </div>
      </div>

      <p className="composer-footer-tagline">
        Evidence informs. It doesn't replace your judgment.
      </p>
    </div>
  );
}
