/**
 * Conversation persistence for the current tab session.
 *
 * Deliberately `sessionStorage`, not `localStorage`, and deliberately not a
 * database: this exists so an accidental refresh does not throw away a check
 * that is still open. It disappears when the tab closes, and it never implies
 * a history feature the product does not have.
 *
 * A conversation containing an attachiment can get large — data URLs are bulky —
 * so anything over the size limit is simply not persisted rather than being
 * partially stored or silently truncated.
 */

const KEY = 'raven:session:v1'
const VERSION = 1
const MAX_CHARS = 1_500_000

export function saveConversation(turns, context) {
  try {
    if (!turns?.length) {
      sessionStorage.removeItem(KEY)
      return
    }

    const payload = JSON.stringify({ version: VERSION, turns, context })
    if (payload.length > MAX_CHARS) {
      sessionStorage.removeItem(KEY)
      return
    }

    sessionStorage.setItem(KEY, payload)
  } catch {
    // Storage can be unavailable (private mode) or full. Neither is worth
    // interrupting a verification for.
  }
}

export function loadConversation() {
  try {
    const raw = sessionStorage.getItem(KEY)
    if (!raw) return null

    const parsed = JSON.parse(raw)
    if (!parsed || parsed.version !== VERSION || !Array.isArray(parsed.turns)) return null

    return { turns: parsed.turns, context: parsed.context ?? null }
  } catch {
    return null
  }
}

export function clearConversation() {
  try {
    sessionStorage.removeItem(KEY)
  } catch {
    // Nothing to do.
  }
}
