import { useCallback, useEffect, useRef, useState } from 'react'

const NEAR_END = 140
/** Scrolling we started ourselves is not the reader scrolling away. */
const PROGRAMMATIC_GRACE = 900

/**
 * Keeps a scrolling region at its end as new turns arrive — but only while the
 * reader is already at the end. Someone who has scrolled up to read an earlier
 * answer or a source passage is never moved.
 *
 * Two details matter here:
 *
 * 1. The scroll events produced by our own smooth scrolling are ignored for a
 *    short grace period. Without that, a turn arriving mid-animation reads as
 *    the reader having scrolled away, and the page silently stops following.
 * 2. A long jump is instant rather than smooth. Animating several screens of
 *    content is slower than arriving.
 *
 * `trigger` should change whenever content is added (turn count + pending state).
 */
export function useScrollAnchor(scrollRef, trigger) {
  const [atEnd, setAtEnd] = useState(true)
  const programmaticUntil = useRef(0)

  const distanceFromEnd = useCallback(() => {
    const element = scrollRef.current
    if (!element) return 0
    return element.scrollHeight - element.scrollTop - element.clientHeight
  }, [scrollRef])

  const toEnd = useCallback(
    (behavior = 'smooth') => {
      const element = scrollRef.current
      if (!element) return
      programmaticUntil.current = Date.now() + PROGRAMMATIC_GRACE
      const distance = element.scrollHeight - element.scrollTop - element.clientHeight
      element.scrollTo({
        top: element.scrollHeight,
        behavior: distance > element.clientHeight * 1.5 ? 'auto' : behavior,
      })
      setAtEnd(true)
    },
    [scrollRef],
  )

  useEffect(() => {
    const element = scrollRef.current
    if (!element) return undefined

    const onScroll = () => {
      if (Date.now() < programmaticUntil.current) return
      setAtEnd(element.scrollHeight - element.scrollTop - element.clientHeight <= NEAR_END)
    }

    onScroll()
    element.addEventListener('scroll', onScroll, { passive: true })
    return () => element.removeEventListener('scroll', onScroll)
  }, [scrollRef])

  useEffect(() => {
    if (atEnd || distanceFromEnd() < NEAR_END) toEnd()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trigger])

  return { toEnd, atEnd }
}
