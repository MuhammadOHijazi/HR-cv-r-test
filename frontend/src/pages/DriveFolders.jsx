import { useEffect, useState } from 'react'
import { api } from '../lib/api.js'
import { ErrorBox, Empty, JobPicker, Loading, useAsync } from '../lib/components.jsx'

export default function DriveFolders() {
  const status = useAsync(() => api.driveStatus(), [])
  const jobs = useAsync(() => api.listJobs(), [])
  const [folders, setFolders] = useState([])
  const [jobId, setJobId] = useState(null)
  const [assigned, setAssigned] = useState([])
  const [syncState, setSyncState] = useState(null)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.listFolders().then(setFolders).catch(setError)
  }, [])

  useEffect(() => {
    if (!jobId) {
      setAssigned([])
      setSyncState(null)
      return
    }
    api.getJobFolders(jobId).then(setAssigned).catch(setError)
    api.syncStatus(jobId).then(setSyncState).catch(setError)
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

  async function refresh() {
    const list = await guard(() => api.refreshFolders(), 'Folder list refreshed from Drive.')
    if (list) setFolders(list)
  }

  async function saveAssignment() {
    await guard(() => api.assignFolders(jobId, assigned), 'Folders assigned to this job.')
  }

  async function syncNow() {
    const report = await guard(() => api.sync(jobId))
    if (report) {
      setNotice(
        `Sync finished: ${report.ingested} new, ${report.duplicates} duplicate, ${report.errors} failed of ${report.total}.`,
      )
      setSyncState(await api.syncStatus(jobId))
      const list = await api.listFolders()
      setFolders(list)
    }
  }

  function toggle(folderId) {
    setAssigned((current) =>
      current.includes(folderId) ? current.filter((f) => f !== folderId) : [...current, folderId],
    )
  }

  const connected = status.data?.connected
  const email = status.data?.service_account_email

  return (
    <>
      <header>
        <h2>Drive Folders</h2>
        <p className="lede">
          Share a Drive folder with the service account, refresh the list, then assign folders to a
          job and sync.
        </p>
      </header>

      <ErrorBox error={error ?? status.error} onDismiss={() => setError(null)} />
      {notice ? <div className="notice">{notice}</div> : null}

      <div className="panel">
        <h3>Connection</h3>
        {status.loading ? (
          <Loading />
        ) : (
          <>
            <p>
              {connected ? (
                <span className="badge auto_shortlist">connected</span>
              ) : (
                <span className="badge preliminary_reject">not connected</span>
              )}{' '}
              {status.data?.mode === 'mock' ? <span className="flag">mock mode</span> : null}
            </p>
            <p className="small">
              Share your CV folders with this service account address:{' '}
              <span className="mono">{email || '(no service account configured)'}</span>
            </p>
            {status.data?.error ? <p className="small muted">{status.data.error}</p> : null}
          </>
        )}
      </div>

      <div className="panel">
        <h3>Folders visible to the service account</h3>
        <button className="secondary" onClick={refresh} disabled={busy}>
          Refresh folder list
        </button>
        <div className="spacer" />
        {folders.length === 0 ? (
          <Empty>
            No folders yet. Share a folder with the service account above, then refresh.
          </Empty>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Assign</th>
                <th>Folder</th>
                <th>Drive ID</th>
                <th>Last synced</th>
              </tr>
            </thead>
            <tbody>
              {folders.map((folder) => (
                <tr key={folder.folder_id}>
                  <td>
                    <input
                      type="checkbox"
                      style={{ width: 'auto' }}
                      disabled={!jobId}
                      checked={assigned.includes(folder.folder_id)}
                      onChange={() => toggle(folder.folder_id)}
                      aria-label={`assign ${folder.name}`}
                    />
                  </td>
                  <td>{folder.name}</td>
                  <td className="mono muted">{folder.folder_id}</td>
                  <td className="muted small">
                    {folder.last_synced_at ? new Date(folder.last_synced_at).toLocaleString() : 'never'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="panel">
        <h3>Assign to a job and sync</h3>
        {jobs.loading ? <Loading /> : null}
        <div className="row">
          <JobPicker jobs={jobs.data ?? []} jobId={jobId} onChange={setJobId} />
          <button className="secondary" onClick={saveAssignment} disabled={!jobId || busy}>
            Save assignment
          </button>
          <button onClick={syncNow} disabled={!jobId || assigned.length === 0 || busy}>
            {busy ? 'Syncing…' : 'Sync from Drive'}
          </button>
        </div>

        {syncState ? (
          <>
            <div className="spacer" />
            <div className="stat-grid">
              <div className="stat">
                <div className="value">{syncState.status}</div>
                <div className="label">last sync</div>
              </div>
              <div className="stat">
                <div className="value">
                  {syncState.processed ?? 0}/{syncState.total ?? 0}
                </div>
                <div className="label">files processed</div>
              </div>
              <div className="stat">
                <div className="value">{syncState.new_files ?? 0}</div>
                <div className="label">newly ingested</div>
              </div>
              <div className="stat">
                <div className="value">{syncState.duplicates ?? 0}</div>
                <div className="label">duplicates skipped</div>
              </div>
            </div>
            {syncState.errors?.length ? (
              <>
                <h4>Files that failed</h4>
                <ul className="tight small">
                  {syncState.errors.map((e) => (
                    <li key={e.file_id}>
                      <span className="mono">{e.filename}</span> — {e.detail}
                    </li>
                  ))}
                </ul>
              </>
            ) : null}
          </>
        ) : null}
      </div>
    </>
  )
}
