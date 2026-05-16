import { useEffect, useRef, useState } from 'react'
import { getSession, SessionState } from '../api'

const TERMINAL_STATUSES = new Set([
  'awaiting_approval',
  'awaiting_md_decision',
  'manuscript_done',
  'error',
])

export function usePolling(sessionId: string | null, intervalMs = 1500) {
  const [session, setSession] = useState<SessionState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (!sessionId) return

    const poll = async () => {
      try {
        const { data } = await getSession(sessionId)
        setSession(data)
        if (TERMINAL_STATUSES.has(data.status)) {
          if (timerRef.current) clearInterval(timerRef.current)
        }
      } catch (e: any) {
        setError(e.message)
        if (timerRef.current) clearInterval(timerRef.current)
      }
    }

    poll()
    timerRef.current = setInterval(poll, intervalMs)
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [sessionId, intervalMs])

  const resume = () => {
    if (!sessionId) return
    timerRef.current = setInterval(async () => {
      try {
        const { data } = await getSession(sessionId)
        setSession(data)
        if (TERMINAL_STATUSES.has(data.status)) {
          if (timerRef.current) clearInterval(timerRef.current)
        }
      } catch {}
    }, intervalMs)
  }

  return { session, error, resume }
}
