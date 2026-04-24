import { useEffect, useRef } from 'react'
export function usePolling(fn, intervalMs, active = true) {
  const fnRef = useRef(fn); fnRef.current = fn
  useEffect(() => {
    if (!active) return; fn()
    const id = setInterval(() => fnRef.current(), intervalMs)
    return () => clearInterval(id)
  }, [active, intervalMs])
}
