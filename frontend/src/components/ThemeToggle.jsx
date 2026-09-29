import { Moon, Sun } from 'lucide-react'
import { useTheme } from '../hooks/useTheme'

/**
 * The light/dark switch.
 *
 * A track with both glyphs and a knob that slides between them, rather than a
 * button that flips its own icon: one glance says which state it is in and
 * which one is available. It is a `switch`, not a button, because it has two
 * states and one of them is currently on.
 */
export default function ThemeToggle() {
  const { isDark, toggle } = useTheme()
  const hint = isDark ? 'Switch to the light theme' : 'Switch to the dark theme'

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={toggle}
      role="switch"
      aria-checked={isDark}
      aria-label="Dark theme"
      title={hint}
    >
      <span className="theme-toggle__track" aria-hidden="true">
        <Sun size={11} strokeWidth={2.1} />
        <Moon size={11} strokeWidth={2.1} />
        <span className="theme-toggle__knob" />
      </span>
    </button>
  )
}
