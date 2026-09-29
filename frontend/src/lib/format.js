/** Small, dependency-free formatters. */

const clock = new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' })
const longDate = new Intl.DateTimeFormat(undefined, { day: 'numeric', month: 'long', year: 'numeric' })
const shortDate = new Intl.DateTimeFormat(undefined, { day: 'numeric', month: 'short', year: 'numeric' })

function parse(value) {
  const date = value instanceof Date ? value : new Date(value)
  return Number.isNaN(date.getTime()) ? null : date
}

/** "14:32" — used beside a conversation turn. */
export function formatClock(value) {
  const date = parse(value)
  return date ? clock.format(date) : ''
}

/** "11 May 2023" */
export function formatDate(value) {
  const date = parse(value)
  return date ? shortDate.format(date) : ''
}

/** "11 May 2023" in full, for tooltips. */
export function formatLongDate(value) {
  const date = parse(value)
  return date ? longDate.format(date) : ''
}

/** Reading-friendly age: "Today", "Yesterday", "4 days ago", else a date. */
export function formatAge(value, now = new Date()) {
  const date = parse(value)
  if (!date) return ''

  const days = Math.floor((now - date) / 86_400_000)
  if (days < 0) return shortDate.format(date)
  if (days === 0) return 'Today'
  if (days === 1) return 'Yesterday'
  if (days < 14) return `${days} days ago`
  if (days < 45) return `${Math.floor(days / 7)} weeks ago`
  if (days < 365) return `${Math.floor(days / 30)} months ago`
  return shortDate.format(date)
}

/** 0.934 → "93%". Rounded to whole percent: a fake-precise reading would not
 *  be honest about what a confidence figure is. */
export function formatPercent(value) {
  return `${Math.round(value * 100)}%`
}

/** "who.int" from a full URL, for showing a domain without the noise. */
export function formatDomain(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return ''
  }
}

/** Trims to a whole word near `max` characters, appending an ellipsis. */
export function clampText(text, max) {
  if (!text || text.length <= max) return text
  const cut = text.slice(0, max)
  const lastSpace = cut.lastIndexOf(' ')
  return `${cut.slice(0, lastSpace > max * 0.6 ? lastSpace : max).trimEnd()}…`
}
