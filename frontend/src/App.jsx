import { useEffect, useState } from 'react'
import ChatView from './pages/ChatView'
import DesignSystem from './pages/DesignSystem'

/** Tiny hash switch. The product is a single continuous surface; this exists
 *  only so internal review views are reachable without a router. */
function useHashRoute() {
  const [hash, setHash] = useState(() => window.location.hash)
  useEffect(() => {
    const onChange = () => setHash(window.location.hash)
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return hash
}

export default function App() {
  const hash = useHashRoute()

  if (hash === '#design') return <DesignSystem />

  return <ChatView />
}
