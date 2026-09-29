import { useCallback, useEffect, useRef, useState } from 'react'
import { RavenError, request } from '../services/raven'
import { clearConversation, loadConversation, saveConversation } from '../lib/session'

let sequence = 0
function nextId(prefix) {
  sequence += 1
  return `${prefix}-${Date.now().toString(36)}-${sequence}`
}

/**
 * The conversation.
 *
 * Everything the user says and everything RAVEN answers lives in one ordered
 * list of turns, so nothing is ever replaced by something newer — a second
 * check appears underneath the first, and evidence from an earlier answer is
 * still there to scroll back to.
 *
 * Failures become turns too, which means an error is part of the record and
 * can be retried in place.
 */
export function useConversation() {
  // Read once, at mount: an accidental refresh should not lose a check that is
  // still open, but nothing here is a history feature.
  const [restored] = useState(loadConversation)

  const [turns, setTurns] = useState(() => restored?.turns ?? [])
  const [pending, setPending] = useState(null)
  const busy = useRef(false)
  const context = useRef(restored?.context ?? { claim: null, verdict: null })

  useEffect(() => {
    saveConversation(turns, context.current)
  }, [turns])

  const execute = useCallback(async ({ kind, payload, say = null }) => {
    if (busy.current) return
    busy.current = true

    if (say) {
      setTurns((prev) => [
        ...prev,
        { id: nextId('you'), role: 'user', at: new Date().toISOString(), ...say },
      ])
    }
    setPending({ kind })

    try {
      const result = await request({ kind, payload })

      // Remember what the current claim is, so follow-ups stay on topic.
      const claim = result.claim ?? context.current.claim
      const verdict = result.verdict ?? context.current.verdict
      if (result.claim) context.current = { claim, verdict }

      // Each suggestion carries the claim it belongs to. Without this, choosing
      // an older suggestion further down the conversation would answer about
      // whatever claim was checked most recently instead of its own.
      const followUps = result.followUps?.map((item) => ({
        ...item,
        context: { claim, verdict },
      }))

      setTurns((prev) => [
        ...prev,
        {
          id: nextId('raven'),
          role: 'raven',
          at: new Date().toISOString(),
          ...result,
          followUps,
        },
      ])
    } catch (error) {
      const isRaven = error instanceof RavenError
      setTurns((prev) => [
        ...prev,
        {
          id: nextId('raven'),
          role: 'raven',
          at: new Date().toISOString(),
          kind: 'error',
          message: isRaven
            ? error.userMessage
            : 'Something went wrong while checking that. The message is still here — try again whenever you like.',
          retryable: isRaven ? error.retryable : true,
          failed: { kind, payload },
        },
      ])
    } finally {
      setPending(null)
      busy.current = false
    }
  }, [])

  /** A new claim, typed or attached. */
  const send = useCallback(
    ({ text = '', image = null }) => {
      if (!text.trim() && !image) return
      return execute({
        kind: 'assessment',
        payload: { text: text.trim(), image },
        say: { text: text.trim(), image },
      })
    },
    [execute],
  )

  /** A suggestion chosen under an answer. Reads as a turn, because it is one. */
  const followUp = useCallback(
    (action) => {
      const { claim, verdict } = action.context ?? context.current

      if (action.action === 'why') {
        return execute({
          kind: 'explanation',
          payload: { claim, verdict },
          say: { text: action.label },
        })
      }

      if (action.action === 'evidence') {
        return execute({
          kind: 'evidence',
          payload: { claim },
          say: { text: action.label },
        })
      }

      return undefined
    },
    [execute],
  )

  /** A question about one source. */
  const aboutSource = useCallback(
    (source) => {
      return execute({
        kind: 'answer',
        payload: { source },
        say: { text: `What is ${source.publisher}?` },
      })
    },
    [execute],
  )

  /** Re-runs whatever failed, in place. */
  const retry = useCallback(
    (turn) => {
      if (!turn?.failed) return undefined
      return execute({ kind: turn.failed.kind, payload: turn.failed.payload })
    },
    [execute],
  )

  const startOver = useCallback(() => {
    context.current = { claim: null, verdict: null }
    setTurns([])
    setPending(null)
    busy.current = false
    clearConversation()
  }, [])

  return {
    turns,
    isThinking: pending !== null,
    thinkingKind: pending?.kind ?? 'assessment',
    send,
    followUp,
    aboutSource,
    retry,
    startOver,
    hasTurns: turns.length > 0,
  }
}
