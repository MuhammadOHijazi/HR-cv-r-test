import { useEffect, useState } from 'react'
import { api } from '../lib/api.js'
import { ErrorBox, Empty, JobPicker, Loading, useAsync } from '../lib/components.jsx'

const DEGREES = ['', 'diploma', 'bachelor', 'master', 'phd']

export default function JobDescriptions() {
  const jobs = useAsync(() => api.listJobs(), [])
  const [jobId, setJobId] = useState(null)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)
  const [busy, setBusy] = useState(false)

  const [title, setTitle] = useState('')
  const [text, setText] = useState('')

  const [versions, setVersions] = useState([])
  const [draft, setDraft] = useState(null)
  const [config, setConfig] = useState(null)

  useEffect(() => {
    if (!jobId) {
      setVersions([])
      setDraft(null)
      setConfig(null)
      return
    }
    Promise.all([api.listVersions(jobId), api.getConfig(jobId)])
      .then(([vs, cfg]) => {
        setVersions(vs)
        setConfig(cfg)
        setDraft(vs.length ? { version: vs[vs.length - 1].version, structured: vs[vs.length - 1].structured } : null)
      })
      .catch(setError)
  }, [jobId])

  async function guard(action, message) {
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      const result = await action()
      if (message) setNotice(message)
      return result
    } catch (err) {
      setError(err)
      return null
    } finally {
      setBusy(false)
    }
  }

  async function createJob(event) {
    event.preventDefault()
    const created = await guard(() => api.createJob(title, text), 'Job created. Structure it next.')
    if (created) {
      setTitle('')
      setText('')
      jobs.reload()
      setJobId(created.id)
    }
  }

  async function structure() {
    const version = await guard(() => api.structureJd(jobId), 'Gemini structured the JD. Review and approve it.')
    if (version) {
      setVersions((vs) => [...vs, version])
      setDraft({ version: version.version, structured: version.structured })
      jobs.reload()
    }
  }

  async function saveEdit() {
    const created = await guard(
      () => api.editVersion(jobId, draft.version, draft.structured),
      'Saved as a new version. Approve it to screen against it.',
    )
    if (created) {
      setVersions((vs) => [...vs, created])
      setDraft({ version: created.version, structured: created.structured })
    }
  }

  async function approve() {
    const approved = await guard(
      () => api.approveVersion(jobId, draft.version),
      `Version ${draft.version} approved. Screening can now run against it.`,
    )
    if (approved) {
      setVersions((vs) => vs.map((v) => (v.version === approved.version ? approved : v)))
      jobs.reload()
    }
  }

  async function saveConfig(event) {
    event.preventDefault()
    await guard(() => api.updateConfig(jobId, config), 'Thresholds saved for this job.')
  }

  function patch(path, value) {
    setDraft((d) => {
      const next = structuredClone(d)
      let node = next.structured
      for (const key of path.slice(0, -1)) node = node[key]
      node[path[path.length - 1]] = value
      return next
    })
  }

  const activeVersion = versions.find((v) => v.approved)

  return (
    <>
      <header>
        <h2>Job Descriptions</h2>
        <p className="lede">
          Paste free text, let Gemini structure it, then review, edit and approve. No screening runs
          against an unapproved job description.
        </p>
      </header>

      <ErrorBox error={error} onDismiss={() => setError(null)} />
      {notice ? <div className="notice">{notice}</div> : null}

      <div className="panel">
        <h3>Add a job</h3>
        <form onSubmit={createJob}>
          <div className="row">
            <div className="grow">
              <label htmlFor="jd-title">Job title</label>
              <input
                id="jd-title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Senior Backend Engineer"
                required
              />
            </div>
          </div>
          <div className="spacer" />
          <label htmlFor="jd-text">Job description (free text)</label>
          <textarea
            id="jd-text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={'Must have:\n- Python and PostgreSQL\n- 5 years of experience\n\nResponsibilities:\n- ...'}
            required
          />
          <div className="spacer" />
          <button type="submit" disabled={busy || !title.trim() || !text.trim()}>
            Create job
          </button>
        </form>
      </div>

      <div className="panel">
        <h3>Structure, review and approve</h3>
        {jobs.loading ? <Loading /> : null}
        <div className="row">
          <JobPicker jobs={jobs.data ?? []} jobId={jobId} onChange={setJobId} />
          <button onClick={structure} disabled={!jobId || busy}>
            Structure with Gemini
          </button>
        </div>

        {jobId && versions.length === 0 ? (
          <>
            <div className="spacer" />
            <Empty>No versions yet — structure the job description to create version 1.</Empty>
          </>
        ) : null}

        {draft ? (
          <>
            <div className="spacer" />
            <div className="row">
              <div className="grow">
                <label htmlFor="version-picker">Version</label>
                <select
                  id="version-picker"
                  value={draft.version}
                  onChange={(e) => {
                    const v = versions.find((x) => x.version === Number(e.target.value))
                    setDraft({ version: v.version, structured: v.structured })
                  }}
                >
                  {versions.map((v) => (
                    <option key={v.version} value={v.version}>
                      v{v.version}
                      {v.approved ? ' — approved' : ' — draft'}
                    </option>
                  ))}
                </select>
              </div>
              <button className="secondary" onClick={saveEdit} disabled={busy}>
                Save as new version
              </button>
              <button className="good" onClick={approve} disabled={busy}>
                Approve v{draft.version}
              </button>
            </div>

            {activeVersion ? (
              <p className="small muted">
                Screening currently runs against v{activeVersion.version}, approved by{' '}
                {activeVersion.approved_by ?? 'unknown'}.
              </p>
            ) : (
              <p className="small muted">No version is approved yet, so screening is blocked.</p>
            )}

            <h4>Must-have requirements</h4>
            <SkillEditor
              items={draft.structured.must_have ?? []}
              onChange={(items) => patch(['must_have'], items)}
            />

            <h4>Nice-to-have requirements</h4>
            <SkillEditor
              items={draft.structured.nice_to_have ?? []}
              onChange={(items) => patch(['nice_to_have'], items)}
            />

            <h4>Thresholds</h4>
            <div className="row">
              <div className="grow">
                <label htmlFor="min-years">Minimum years of experience</label>
                <input
                  id="min-years"
                  type="number"
                  min="0"
                  step="0.5"
                  value={draft.structured.thresholds?.min_years_experience ?? 0}
                  onChange={(e) => patch(['thresholds', 'min_years_experience'], Number(e.target.value))}
                />
              </div>
              <div className="grow">
                <label htmlFor="req-degree">Required degree</label>
                <select
                  id="req-degree"
                  value={draft.structured.thresholds?.required_degree ?? ''}
                  onChange={(e) => patch(['thresholds', 'required_degree'], e.target.value || null)}
                >
                  {DEGREES.map((d) => (
                    <option key={d} value={d}>
                      {d || 'none'}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <h4>Dimension weights</h4>
            <div className="row">
              {Object.entries(draft.structured.weights ?? {}).map(([key, value]) => (
                <div className="grow" key={key}>
                  <label htmlFor={`weight-${key}`}>{key.replace(/_/g, ' ')}</label>
                  <input
                    id={`weight-${key}`}
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    value={value}
                    onChange={(e) => patch(['weights', key], Number(e.target.value))}
                  />
                </div>
              ))}
            </div>

            <h4>Responsibilities</h4>
            <textarea
              value={(draft.structured.responsibilities ?? []).join('\n')}
              onChange={(e) =>
                patch(['responsibilities'], e.target.value.split('\n').filter((l) => l.trim()))
              }
            />
          </>
        ) : null}
      </div>

      {config ? (
        <div className="panel">
          <h3>Routing thresholds for this job</h3>
          <p className="small muted">
            Thresholds are per job, not global. Anything below the confidence floor goes to a human
            whatever its score.
          </p>
          <form onSubmit={saveConfig}>
            <div className="row">
              {[
                ['shortlist_score_min', 'Auto-shortlist at or above', 1],
                ['reject_score_max', 'Preliminary reject below', 1],
                ['confidence_min', 'Confidence floor', 0.05],
                ['disagreement_cap', 'Scorer disagreement cap', 1],
                ['years_conflict_tolerance', 'Years conflict tolerance', 0.5],
              ].map(([key, label, step]) => (
                <div className="grow" key={key}>
                  <label htmlFor={`cfg-${key}`}>{label}</label>
                  <input
                    id={`cfg-${key}`}
                    type="number"
                    step={step}
                    value={config[key]}
                    onChange={(e) => setConfig({ ...config, [key]: Number(e.target.value) })}
                  />
                </div>
              ))}
            </div>
            <div className="spacer" />
            <button type="submit" disabled={busy}>
              Save thresholds
            </button>
          </form>
        </div>
      ) : null}
    </>
  )
}

function SkillEditor({ items, onChange }) {
  const [value, setValue] = useState('')
  return (
    <>
      <div className="row">
        <div className="grow">
          <input
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="Add a requirement and press Add"
          />
        </div>
        <button
          type="button"
          className="secondary"
          onClick={() => {
            if (!value.trim()) return
            onChange([...items, { skill: value.trim() }])
            setValue('')
          }}
        >
          Add
        </button>
      </div>
      <ul className="tight">
        {items.length === 0 ? <li className="muted small">none</li> : null}
        {items.map((item, index) => (
          <li key={`${item.skill}-${index}`}>
            {item.skill}
            {item.canonical && item.canonical !== item.skill ? (
              <span className="muted small"> → {item.canonical}</span>
            ) : null}{' '}
            <button
              type="button"
              className="secondary small"
              onClick={() => onChange(items.filter((_, i) => i !== index))}
            >
              remove
            </button>
          </li>
        ))}
      </ul>
    </>
  )
}
