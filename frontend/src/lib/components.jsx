import { useCallback, useEffect, useState } from 'react'
import { ApiError, FLAG_LABELS, ROUTING_LABELS } from './api.js'

const SEVERE_FLAGS = new Set([
  'injection_suspicion',
  'extraction_failed',
  'high_confidence_must_have_failure',
  'high_score_weak_evidence',
])

export function RoutingBadge({ routing }) {
  return <span className={`badge ${routing}`}>{ROUTING_LABELS[routing] ?? routing}</span>
}

export function Flags({ flags }) {
  if (!flags || flags.length === 0) return <span className="muted small">none</span>
  return (
    <>
      {flags.map((flag) => (
        <span key={flag} className={`flag${SEVERE_FLAGS.has(flag) ? ' severe' : ''}`}>
          {FLAG_LABELS[flag] ?? flag}
        </span>
      ))}
    </>
  )
}

export function Bar({ value, max = 100 }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100))
  return (
    <div className="bar" title={`${value}`}>
      <span style={{ width: `${pct}%` }} />
    </div>
  )
}

export function ErrorBox({ error, onDismiss }) {
  if (!error) return null
  const message = error instanceof ApiError ? `${error.status}: ${error.message}` : String(error.message ?? error)
  return (
    <div className="error">
      {message}
      {onDismiss ? (
        <>
          {' '}
          <button className="secondary small" onClick={onDismiss}>
            dismiss
          </button>
        </>
      ) : null}
    </div>
  )
}

export function Loading({ children = 'Loading…' }) {
  return <p className="muted">{children}</p>
}

export function Empty({ children }) {
  return <p className="muted">{children}</p>
}

/** Fetch-on-mount with loading/error state and a manual reload. */
export function useAsync(fn, deps = []) {
  const [state, setState] = useState({ data: null, loading: true, error: null })

  const run = useCallback(() => {
    let cancelled = false
    setState((s) => ({ ...s, loading: true }))
    fn()
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: null })
      })
      .catch((error) => {
        if (!cancelled) setState({ data: null, loading: false, error })
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => run(), [run])

  return { ...state, reload: run, setData: (data) => setState({ data, loading: false, error: null }) }
}

/** A job picker shared by every page that operates on one job. */
export function JobPicker({ jobs, jobId, onChange, label = 'Job' }) {
  return (
    <div className="grow">
      <label htmlFor="job-picker">{label}</label>
      <select
        id="job-picker"
        value={jobId ?? ''}
        onChange={(e) => onChange(e.target.value ? Number(e.target.value) : null)}
      >
        <option value="">Select a job…</option>
        {jobs.map((job) => (
          <option key={job.id} value={job.id}>
            {job.title}
            {job.approved ? '' : ' (JD not approved)'}
          </option>
        ))}
      </select>
    </div>
  )
}

export function Score({ value }) {
  return <strong>{Number(value).toFixed(1)}</strong>
}

export function Confidence({ value }) {
  const pct = Math.round(Number(value) * 100)
  return <span title={`confidence ${value}`}>{pct}%</span>
}

export function Quote({ text, verified }) {
  if (!text) return <p className="muted small">no quote supplied</p>
  return (
    <blockquote className={`quote ${verified ? 'verified' : 'unverified'}`}>
      “{text}”
      <div className="small">{verified ? 'verified against the source text' : 'NOT found in the source text'}</div>
    </blockquote>
  )
}
