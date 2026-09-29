/**
 * Theme preference.
 *
 * Kept in `localStorage`, unlike the conversation, which lives in
 * `sessionStorage`. The distinction is deliberate: a conversation is a piece of
 * work that should disappear when the tab closes, whereas a preference is a
 * standing choice and should survive it.
 *
 * The first visit has no stored preference and follows the operating system.
 * Once the reader uses the switch, that choice wins from then on — including
 * over later changes to the system setting.
 *
 * `index.html` carries a tiny copy of the boot step so the correct theme is set
 * before the first paint. If the key or the attribute below changes, that copy
 * has to change with it.
 */

const KEY = 'raven:theme:v1'

export const THEMES = ['light', 'dark']

const META_COLOR = {
  light: '#faf8f4',
  dark: '#131211',
}

export function isTheme(value) {
  return THEMES.includes(value)
}

/** The reader's explicit choice, or `null` if they have never made one. */
export function readStoredTheme() {
  try {
    const value = localStorage.getItem(KEY)
    return isTheme(value) ? value : null
  } catch {
    // Storage can be unavailable in private mode. Falling back to the system
    // setting is a better answer than an error nobody asked for.
    return null
  }
}

export function systemTheme() {
  return window.matchMedia?.('(prefers-color-scheme: dark)')?.matches ? 'dark' : 'light'
}

export function resolveTheme() {
  return readStoredTheme() ?? systemTheme()
}

export function storeTheme(theme) {
  try {
    localStorage.setItem(KEY, theme)
  } catch {
    // Nothing to do — the theme still applies for this page.
  }
}

/** Writes the theme to the document, where every stylesheet reads it. */
export function applyTheme(theme) {
  document.documentElement.dataset.theme = theme

  const meta = document.querySelector('meta[name="theme-color"]')
  if (meta) meta.setAttribute('content', META_COLOR[theme] ?? META_COLOR.light)
}
