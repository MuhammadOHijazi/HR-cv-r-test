import { useEffect, useState } from 'react'
import { api } from '../lib/api.js'
import {
  Confidence,
  Empty,
  ErrorBox,
  Flags,
  JobPicker,
  Loading,
  Quote,
  RoutingBadge,
  Score,
  useAsync,
} from '../lib/components.jsx'

export default function ReviewQueue() {
  const jobs = useAsync(() => api.listJobs(), [])
  const reasons = useAsync(() => api.rejectReasons(), [])
  const [jobId, setJobId] = useState(null)
  const [queue, setQueue] = useState([])
  const [rejects, setRejects] = useState([])
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)
  const [busy, setBusy] = useState(false)

  async function load(id = jobId) {
    if (!id) {
      setQueue([])
      setRejects([])
      return
    }
    try {
      setQueue(await api.reviewQueue(id))
      setRejects(await api.preliminaryRejects(id))
    } catch (err) {
      setError(err)
    }
  }

  useEffect(() => {
    load(jobId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])

  async function act(action, message) {
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      await action()
      setNotice(message)
      await load()
      jobs.reload()
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <header>
        <h2>Review Queue</h2>
        <p className="lede">
          Everything the machine would not decide on its own. Nothing here is rejected until a person
          says so.
        </p>
      </header>

      <ErrorBox error={error ?? jobs.error} onDismiss={() => setError(null)} />
      {notice ? <div className="notice">{notice}</div> : null}

      <div className="panel">
        {jobs.loading ? <Loading /> : null}
        <div className="row">
          <JobPicker jobs={jobs.data ?? []} jobId={jobId} onChange={setJobId} />
          <button className="secondary" onClick={() => load()} disabled={!jobId || busy}>
            Reload
          </button>
        </div>
      </div>

      {!jobId ? <Empty>Select a job to review its queue.</Empty> : null}

      {jobId && queue.length === 0 ? (
        <div className="panel">
          <Empty>Nothing waiting for review on this job.</Empty>
        </div>
      ) : null}

      {queue.map((entry) => (
        <ReviewCard
          key={entry.review_entry_id}
          entry={entry}
          reasons={reasons.data ?? []}
          busy={busy}
          onApprove={(note) =>
            act(
              () => api.approveReview(entry.review_entry_id, note),
              `${entry.candidate_name ?? 'Candidate'} approved to the shortlist.`,
            )
          }
          onReject={(reason, note) =>
            act(
              () => api.rejectReview(entry.review_entry_id, reason, note),
              `${entry.candidate_name ?? 'Candidate'} rejected: ${reason}.`,
            )
          }
          onCorrect={(corrections) =>
            act(
              () => api.correctReview(entry.review_entry_id, corrections),
              'Correction applied — the candidate was re-scored and re-routed.',
            )
          }
        />
      ))}

      {jobId && rejects.length > 0 ? (
        <div className="panel">
          <h3>Preliminary rejects awaiting confirmation</h3>
          <p className="small muted">
            The machine has never rejected anybody. These are queued for one-click human
            confirmation.
          </p>
          <table>
            <thead>
              <tr>
                <th>Candidate</th>
                <th>Score</th>
                <th>Confidence</th>
                <th>Why</th>
              </tr>
            </thead>
            <tbody>
              {rejects.map((row) => (
                <tr key={row.id}>
                  <td>{row.candidate_name ?? `Candidate ${row.candidate_id}`}</td>
                  <td>
                    <Score value={row.score} />
                  </td>
                  <td>
                    <Confidence value={row.confidence} />
                  </td>
                  <td className="small muted">
                    {row.rules?.missing_must_have?.length
                      ? `missing: ${row.rules.missing_must_have.join(', ')}`
                      : 'score below the reject threshold'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="spacer" />
          <button
            className="danger"
            disabled={busy}
            onClick={() =>
              act(
                () => api.confirmRejects(jobId, rejects.map((r) => r.id)),
                `Confirmed ${rejects.length} rejections.`,
              )
            }
          >
            Confirm all {rejects.length} rejections
          </button>
        </div>
      ) : null}
    </>
  )
}

function ReviewCard({ entry, reasons, busy, onApprove, onReject, onCorrect }) {
  const [reason, setReason] = useState(reasons[0] ?? 'other_documented')
  const [note, setNote] = useState('')
  const [skill, setSkill] = useState('')
  const [years, setYears] = useState('')

  const missing = entry.rules?.missing_must_have ?? []

  return (
    <div className="panel">
      {/* The routing reason comes first: why before what. */}
      <h3>Why this is here</h3>
      <ul className="tight">
        {(entry.reasons ?? []).map((r) => (
          <li key={r.code}>
            <strong>{r.code}</strong> — {r.detail}
          </li>
        ))}
      </ul>

      <div className="row">
        <div className="grow">
          <h4 style={{ marginTop: 0 }}>
            {entry.candidate_name ?? `Candidate ${entry.candidate_id}`}{' '}
            <RoutingBadge routing={entry.routing} />
          </h4>
          <p className="small">
            Score <Score value={entry.score} /> · confidence <Confidence value={entry.confidence} /> ·{' '}
            <span className="mono">{entry.filename}</span>
          </p>
          <p>
            <Flags flags={entry.flags} />
          </p>
        </div>
      </div>

      <h4>Score breakdown</h4>
      <table>
        <thead>
          <tr>
            <th>Dimension</th>
            <th>Raw</th>
            <th>Contribution</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(entry.dimensions ?? {}).map(([name, d]) => (
            <tr key={name}>
              <td>{name.replace(/_/g, ' ')}</td>
              <td>{(d.raw * 100).toFixed(0)}%</td>
              <td>{d.contribution.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>Evidence</h4>
      {Object.entries(entry.evidence ?? {}).map(([dim, e]) => (
        <div key={dim}>
          <strong className="small">
            {dim.replace(/_/g, ' ')} — {e.score}
          </strong>
          <Quote text={e.quote} verified={e.verified} />
        </div>
      ))}

      <SourceText candidateId={entry.candidate_id} jobId={entry.job_id} />

      <h4>Decide</h4>
      <div className="row">
        <div className="grow">
          <label htmlFor={`note-${entry.review_entry_id}`}>Note (optional)</label>
          <input
            id={`note-${entry.review_entry_id}`}
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
        </div>
        <button className="good" disabled={busy} onClick={() => onApprove(note)}>
          Approve to shortlist
        </button>
      </div>

      <div className="spacer" />
      <div className="row">
        <div className="grow">
          <label htmlFor={`reason-${entry.review_entry_id}`}>Rejection reason (closed list)</label>
          <select
            id={`reason-${entry.review_entry_id}`}
            value={reason}
            onChange={(e) => setReason(e.target.value)}
          >
            {reasons.map((r) => (
              <option key={r} value={r}>
                {r.replace(/_/g, ' ')}
              </option>
            ))}
          </select>
        </div>
        <button className="danger" disabled={busy} onClick={() => onReject(reason, note)}>
          Reject
        </button>
      </div>

      <h4>Or correct a field and re-score</h4>
      <p className="small muted">
        A correction is authoritative: it is recorded with your name, and the candidate is
        immediately re-scored and re-routed.
      </p>
      {missing.length ? (
        <p className="small">
          Unmet must-haves:{' '}
          {missing.map((m) => (
            <button
              key={m}
              className="secondary small"
              style={{ marginRight: 6 }}
              disabled={busy}
              onClick={() => onCorrect({ add_skills: [{ name: m, canonical: m }] })}
            >
              the CV does have {m}
            </button>
          ))}
        </p>
      ) : null}
      <div className="row">
        <div className="grow">
          <label htmlFor={`skill-${entry.review_entry_id}`}>Add a skill the extractor missed</label>
          <input
            id={`skill-${entry.review_entry_id}`}
            value={skill}
            onChange={(e) => setSkill(e.target.value)}
            placeholder="Kubernetes"
          />
        </div>
        <button
          className="secondary"
          disabled={busy || !skill.trim()}
          onClick={() => {
            onCorrect({ add_skills: [{ name: skill.trim() }] })
            setSkill('')
          }}
        >
          Add and re-score
        </button>
      </div>
      <div className="spacer" />
      <div className="row">
        <div className="grow">
          <label htmlFor={`years-${entry.review_entry_id}`}>Correct the years of experience</label>
          <input
            id={`years-${entry.review_entry_id}`}
            type="number"
            min="0"
            step="0.5"
            value={years}
            onChange={(e) => setYears(e.target.value)}
          />
        </div>
        <button
          className="secondary"
          disabled={busy || years === ''}
          onClick={() => {
            const value = Number(years)
            onCorrect({ stated_years_experience: value, computed_years: value })
            setYears('')
          }}
        >
          Set and re-score
        </button>
      </div>
    </div>
  )
}

function SourceText({ candidateId, jobId }) {
  const [text, setText] = useState(null)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    if (!open || text !== null) return
    api
      .candidate(candidateId, jobId)
      .then((c) => setText(c.source_text || '(no text extracted)'))
      .catch(() => setText('(could not load the source text)'))
  }, [open, text, candidateId, jobId])

  return (
    <details onToggle={(e) => setOpen(e.target.open)}>
      <summary>Source text</summary>
      <div className="source-text">{text ?? 'Loading…'}</div>
    </details>
  )
}
