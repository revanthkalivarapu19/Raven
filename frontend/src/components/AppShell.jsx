import { useState } from 'react'
import Wordmark from './Wordmark'
import AboutSheet from './AboutSheet'
import ThemeToggle from './ThemeToggle'

/**
 * The application frame.
 *
 * Layout: a masthead rail carrying identity and the two or three things a
 * user can do besides checking a claim, then one continuous workspace that
 * scrolls, with the composer docked at its foot. The rail collapses into a
 * compact masthead bar on small screens; the workspace never changes shape.
 *
 * `children` is the scrolling region. `dock` is the composer slot.
 */
export default function AppShell({
  children,
  dock,
  hasConversation = false,
  onNewCheck,
  scrollRef,
}) {
  // The `#about` hash exists so the sheet can be opened directly for review.
  const [aboutOpen, setAboutOpen] = useState(() => window.location.hash === '#about')

  return (
    <div className="shell">
      <aside className="rail">
        <div className="rail__identity">
          <Wordmark size="md" />
          <p className="rail__statement">
            Check a claim against the evidence — and see the sources for yourself.
          </p>
        </div>

        <nav className="rail__nav" aria-label="Secondary">
          <button type="button" className="rail__nav-item" onClick={() => setAboutOpen(true)}>
            About
          </button>
          {hasConversation && (
            <button
              type="button"
              className="rail__nav-item rail__nav-item--accent"
              onClick={onNewCheck}
            >
              Start a new check
            </button>
          )}
        </nav>

        <div className="rail__foot">
          <ThemeToggle />
          <p className="rail__foot-note">
            RAVEN shows its working. Every answer names the sources it used, so you can
            read them and decide for yourself.
          </p>
        </div>
      </aside>

      <div className="workspace">
        <div
          className="workspace__scroll"
          ref={scrollRef}
          tabIndex={hasConversation ? 0 : -1}
          aria-label={hasConversation ? 'Conversation' : undefined}
        >
          {children}
        </div>
        {dock && (
          <div className="workspace__dock">
            <div className="workspace__dock-inner">{dock}</div>
          </div>
        )}
      </div>

      {aboutOpen && <AboutSheet onClose={() => setAboutOpen(false)} />}
    </div>
  )
}
