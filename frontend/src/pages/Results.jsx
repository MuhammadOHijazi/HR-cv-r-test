import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, ROUTING_LABELS } from '../lib/api.js'
import {
  Bar,
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

const SORTS = [
  ['score', 'Score'],
  ['confidence', 'Confidence'],
  ['candidate', 'Candidate'],
]

export default function Results() {
  const jobs = useAsync(() => api.listJobs(), [])
  const [params, setParams] = useSearchParams()
  const [jobId, setJobId] = useState(params.get('job') ? Number(params.get('job')) : null)
  const [rows, setRows] = useState([])
  const [expanded, setExpanded] = useState(null)
  const [detail, setDetail] = useState(null)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)
  const [busy, setBusy] = useState(false)
  const [filters, setFilters] = useState({ routing: '', min_score: '', min_confidence: '', flag: '' })
  const [sort, setSort] = useState('score')
  const [order, setOrder] = useState('desc')

  useEffect(() => {
    if (!jobId) {
      setRows([])
      return
    }
    setParams({ job: String(jobId) }, { replace: true })
    api
      .results(jobId, { ...filters, sort, order })
      .then(setRows)
      .catch(setError)
  }, [jobId, filters, sort, order])

  useEffect(() => {
    if (expanded === null) {
      setDetail(null)
      return
    }
    api.candidate(expanded, jobId).then(setDetail).catch(setError)
  }, [expanded, jobId])

  async function runScreening() {
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      const report = await api.screen(jobId)
      setNotice(
        `Screened ${report.screened} candidates: ` +
          Object.entries(report.counts)
            .map(([k, v]) => `${v} ${ROUTING_LABELS[k] ?? k}`)
            .join(', '),
      )
      setRows(await api.results(jobId, { ...filters, sort, order }))
      jobs.reload()
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  function toggleSort(field) {
    if (sort === field) setOrder((o) => (o === 'desc' ? 'asc' : 'desc'))
    else {
      setSort(field)
      setOrder('desc')
    }
  }

  return (
    <>
      <header>
        <h2>Results</h2>
        <p className="lede">Ranked candidates with their score breakdown, evidence and confidence.</p>
      </header>

      <ErrorBox error={error ?? jobs.error} onDismiss={() => setError(null)} />
      {notice ? <div className="notice">{notice}</div> : null}

      <div className="panel">
        {jobs.loading ? <Loading /> : null}
        <div className="row">
          <JobPicker jobs={jobs.data ?? []} jobId={jobId} onChange={setJobId} />
          <button onClick={runScreening} disabled={!jobId || busy}>
            {busy ? 'Screening…' : 'Run screening'}
          </button>
        </div>
        <div className="spacer" />
        <div className="row">
          <div className="grow">
            <label htmlFor="f-routing">Routing</label>
            <select
              id="f-routing"
              value={filters.routing}
              onChange={(e) => setFilters({ ...filters, routing: e.target.value })}
            >
              <option value="">all</option>
              {Object.entries(ROUTING_LABELS).map(([key, label]) => (
                <option key={key} value={key}>
                  {label}
                </option>
              ))}
            </select>
          </div>
          <div className="grow">
            <label htmlFor="f-score">Minimum score</label>
            <input
              id="f-score"
              type="number"
              min="0"
              max="100"
              value={filters.min_score}
              onChange={(e) => setFilters({ ...filters, min_score: e.target.value })}
            />
          </div>
          <div className="grow">
            <label htmlFor="f-conf">Minimum confidence</label>
            <input
              id="f-conf"
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={filters.min_confidence}
              onChange={(e) => setFilters({ ...filters, min_confidence: e.target.value })}
            />
          </div>
          <div className="grow">
            <label htmlFor="f-flag">Flag</label>
            <input
              id="f-flag"
              value={filters.flag}
              placeholder="e.g. injection_suspicion"
              onChange={(e) => setFilters({ ...filters, flag: e.target.value })}
            />
          </div>
          <button
            className="secondary"
            onClick={() => setFilters({ routing: '', min_score: '', min_confidence: '', flag: '' })}
          >
            Clear
          </button>
        </div>
      </div>

      <div className="panel">
        <h3>Ranked candidates</h3>
        {!jobId ? (
          <Empty>Select a job to see its results.</Empty>
        ) : rows.length === 0 ? (
          <Empty>No results yet. Sync CVs from Drive, then run screening.</Empty>
        ) : (
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Candidate</th>
                {SORTS.map(([field, label]) => (
                  <th key={field} className="sortable" onClick={() => toggleSort(field)}>
                    {label}
                    {sort === field ? (order === 'desc' ? ' ▼' : ' ▲') : ''}
                  </th>
                ))}
                <th>Routing</th>
                <th>Flags</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={row.id}>
                  <td className="muted">{index + 1}</td>
                  <td>
                    {row.candidate_name ?? `Candidate ${row.candidate_id}`}
                    <div className="muted small mono">{row.filename}</div>
                  </td>
                  <td>
                    <Score value={row.score} />
                    <Bar value={row.score} />
                  </td>
                  <td>
                    <Confidence value={row.confidence} />
                  </td>
                  <td className="muted">{row.candidate_id}</td>
                  <td>
                    <RoutingBadge routing={row.routing} />
                  </td>
                  <td>
                    <Flags flags={row.flags} />
                  </td>
                  <td>
                    <button
                      className="secondary small"
                      onClick={() => setExpanded(expanded === row.candidate_id ? null : row.candidate_id)}
                    >
                      {expanded === row.candidate_id ? 'hide' : 'details'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {expanded !== null ? (
        <ResultDetail
          result={rows.find((r) => r.candidate_id === expanded)}
          candidate={detail}
          onClose={() => setExpanded(null)}
        />
      ) : null}
    </>
  )
}

export function ResultDetail({ result, candidate, onClose }) {
  if (!result) return null
  return (
    <div className="panel">
      <div className="row">
        <h3 className="grow">
          {result.candidate_name ?? `Candidate ${result.candidate_id}`} — <Score value={result.score} />
        </h3>
        {onClose ? (
          <button className="secondary small" onClick={onClose}>
            close
          </button>
        ) : null}
      </div>

      <h4>Dimension breakdown</h4>
      <table>
        <thead>
          <tr>
            <th>Dimension</th>
            <th>Raw</th>
            <th>Weight</th>
            <th>Contribution</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(result.dimensions ?? {}).map(([name, d]) => (
            <tr key={name}>
              <td>{name.replace(/_/g, ' ')}</td>
              <td>{(d.raw * 100).toFixed(0)}%</td>
              <td className="muted">{d.weight}</td>
              <td>
                <strong>{d.contribution.toFixed(1)}</strong>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>Evidence from the judge</h4>
      {Object.entries(result.evidence ?? {}).map(([dim, e]) => (
        <div key={dim}>
          <strong className="small">
            {dim.replace(/_/g, ' ')} — {e.score}
          </strong>
          <Quote text={e.quote} verified={e.verified} />
          {e.rationale ? <p className="small muted">{e.rationale}</p> : null}
        </div>
      ))}

      <h4>Confidence</h4>
      <ul className="tight small">
        {Object.entries(result.confidence_detail?.components ?? {}).map(([key, value]) => (
          <li key={key}>
            {key.replace(/_/g, ' ')}: {(value * 100).toFixed(0)}%
          </li>
        ))}
        {result.confidence_detail?.caps_applied?.length ? (
          <li>
            <strong>capped by:</strong> {result.confidence_detail.caps_applied.join(', ')}
          </li>
        ) : null}
      </ul>

      <h4>Rules gate</h4>
      <p className="small">
        Must-have coverage {(100 * (result.rules?.must_have_coverage ?? 0)).toFixed(0)}%
        {result.rules?.missing_must_have?.length
          ? ` — missing: ${result.rules.missing_must_have.join(', ')}`
          : ''}
      </p>

      {result.review_reasons?.length ? (
        <>
          <h4>Why this routing</h4>
          <ul className="tight small">
            {result.review_reasons.map((r) => (
              <li key={r.code}>
                <strong>{r.code}</strong>: {r.detail}
              </li>
            ))}
          </ul>
        </>
      ) : null}

      <details>
        <summary>Audit trail</summary>
        <ul className="tight small mono">
          <li>prompts: {result.audit?.prompt_version}</li>
          <li>schema: {result.audit?.schema_version}</li>
          <li>models: {result.audit?.model_name}</li>
          <li>thresholds: {JSON.stringify(result.audit?.thresholds)}</li>
          <li>screened: {result.audit?.created_at}</li>
          <li>updated: {result.audit?.updated_at}</li>
        </ul>
      </details>

      {candidate ? (
        <details>
          <summary>Source text ({candidate.files?.[0]?.filename})</summary>
          <div className="source-text">{candidate.source_text || '(no text extracted)'}</div>
        </details>
      ) : null}
    </div>
  )
}
