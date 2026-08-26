import { Link } from 'react-router-dom'
import { api } from '../lib/api.js'
import { ErrorBox, Empty, Loading, useAsync } from '../lib/components.jsx'

const BUCKETS = [
  ['auto_shortlist', 'Shortlisted'],
  ['human_review', 'Needs review'],
  ['preliminary_reject', 'Preliminary rejects'],
]

export default function Dashboard() {
  const jobs = useAsync(() => api.listJobs(), [])

  const totals = (jobs.data ?? []).reduce(
    (acc, job) => {
      for (const [key] of BUCKETS) acc[key] += job.counts[key] ?? 0
      acc.total += job.counts.total ?? 0
      return acc
    },
    { auto_shortlist: 0, human_review: 0, preliminary_reject: 0, total: 0 },
  )

  return (
    <>
      <header>
        <h2>Dashboard</h2>
        <p className="lede">Every job and where its candidates currently sit.</p>
      </header>

      <ErrorBox error={jobs.error} onDismiss={jobs.reload} />
      {jobs.loading ? <Loading /> : null}

      {jobs.data ? (
        <>
          <div className="panel">
            <h3>Across all jobs</h3>
            <div className="stat-grid">
              <div className="stat">
                <div className="value">{jobs.data.length}</div>
                <div className="label">jobs</div>
              </div>
              <div className="stat">
                <div className="value">{totals.total}</div>
                <div className="label">candidates screened</div>
              </div>
              {BUCKETS.map(([key, label]) => (
                <div className="stat" key={key}>
                  <div className="value">{totals[key]}</div>
                  <div className="label">{label}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel">
            <h3>Jobs</h3>
            {jobs.data.length === 0 ? (
              <Empty>
                No jobs yet. <Link to="/jobs">Add a job description</Link> to begin.
              </Empty>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Job</th>
                    <th>Status</th>
                    <th>JD version</th>
                    {BUCKETS.map(([key, label]) => (
                      <th key={key}>{label}</th>
                    ))}
                    <th>Total</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.data.map((job) => (
                    <tr key={job.id}>
                      <td>
                        <Link to={`/results?job=${job.id}`}>{job.title}</Link>
                      </td>
                      <td>
                        {job.approved ? (
                          <span className="badge auto_shortlist">JD approved</span>
                        ) : (
                          <span className="badge human_review">JD not approved</span>
                        )}
                      </td>
                      <td className="muted">{job.active_jd_version ?? '—'}</td>
                      {BUCKETS.map(([key]) => (
                        <td key={key}>{job.counts[key] ?? 0}</td>
                      ))}
                      <td>
                        <strong>{job.counts.total ?? 0}</strong>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      ) : null}
    </>
  )
}
