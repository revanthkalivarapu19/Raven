import { useCallback, useEffect, useState } from 'react'
import { applyTheme, readStoredTheme, resolveTheme, storeTheme } from '../lib/theme'

/**
 * The current theme and the switch for it.
 *
 * The stored preference is read during the first render rather than in an
 * effect, so React agrees with what `index.html` already painted and the
 * interface never flickers from one theme into the other on load.
 */
export function useTheme() {
  const [theme, setTheme] = useState(resolveTheme)

  useEffect(() => {
    applyTheme(theme)
  }, [theme])

  useEffect(() => {
    const media = window.matchMedia?.('(prefers-color-scheme: dark)')
    if (!media?.addEventListener) return undefined

    // A reader who has never touched the switch keeps following their system.
    // One who has, doesn't.
    const onSystemChange = (event) => {
      if (readStoredTheme()) return
      setTheme(event.matches ? 'dark' : 'light')
    }

    media.addEventListener('change', onSystemChange)
    return () => media.removeEventListener('change', onSystemChange)
  }, [])

  const toggle = useCallback(() => {
    const next = theme === 'dark' ? 'light' : 'dark'
    storeTheme(next)
    setTheme(next)
  }, [theme])

  return { theme, toggle, isDark: theme === 'dark' }
}
