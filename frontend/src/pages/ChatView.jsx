import { useMemo, useRef, useState } from 'react'
import { ArrowDown } from 'lucide-react'
import AppShell from '../components/AppShell'
import Conversation from '../components/conversation/Conversation'
import Composer from '../components/Composer'
import EmptyState from '../components/EmptyState'
import { useConversation } from '../hooks/useConversation'
import { useScrollAnchor } from '../hooks/useScrollAnchor'
import { getStarters } from '../services/raven'

/**
 * The product.
 *
 * One surface: either the opening proposition with its examples, or the
 * conversation. The composer stays where it is in both cases, so starting a
 * second check never feels like arriving somewhere new.
 */
export default function ChatView() {
  const { turns, isThinking, thinkingKind, send, followUp, aboutSource, retry, startOver } =
    useConversation()

  const starters = useMemo(() => getStarters(), [])
  const [nudge, setNudge] = useState(null)

  const scrollRef = useRef(null)
  const composerRef = useRef(null)
  const { toEnd, atEnd } = useScrollAnchor(scrollRef, turns.length + (isThinking ? 0.5 : 0))

  /**
   * "Check another claim" is an invitation, not a turn. It brings the composer
   * into focus and says so, rather than putting words in the user's mouth.
   */
  const handleFollowUp = (action) => {
    if (action.action === 'another') {
      setNudge('Paste the next claim whenever you’re ready.')
      composerRef.current?.focus()
      return
    }
    followUp(action)
  }

  const handleSend = (payload) => {
    setNudge(null)
    send(payload)
  }

  const handleStartOver = () => {
    startOver()
    setNudge(null)
  }

  return (
    <AppShell
      scrollRef={scrollRef}
      hasConversation={turns.length > 0}
      onNewCheck={handleStartOver}
      dock={
        <>
          {/* Offered only when the reader has moved back up through a long
              conversation, so they can return to the newest answer. */}
          {!atEnd && turns.length > 0 && (
            <button type="button" className="jump" onClick={() => toEnd()}>
              <ArrowDown size={14} strokeWidth={2} />
              Latest answer
            </button>
          )}

          {nudge && <p className="composer-nudge">{nudge}</p>}
          <Composer
            ref={composerRef}
            onSend={handleSend}
            busy={isThinking}
            placeholder={
              turns.length > 0
                ? 'Add another claim, or ask a follow-up'
                : 'Paste a claim, or describe something you saw'
            }
          />
        </>
      }
    >
      {turns.length === 0 ? (
        <div className="workspace__column workspace__column--centered">
          <EmptyState
            starters={starters}
            onPick={(text) => handleSend({ text })}
          />
        </div>
      ) : (
        <div className="workspace__column">
          <Conversation
            turns={turns}
            isThinking={isThinking}
            thinkingKind={thinkingKind}
            onFollowUp={handleFollowUp}
            onAboutSource={aboutSource}
            onRetry={retry}
          />
        </div>
      )}
    </AppShell>
  )
}
