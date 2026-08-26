import { api } from '../lib/api.js'
import { ErrorBox, Loading, useAsync } from '../lib/components.jsx'

export default function Settings() {
  const settings = useAsync(() => api.settings(), [])

  return (
    <>
      <header>
        <h2>Settings</h2>
        <p className="lede">Gemini key-pool health and the address to share your Drive folders with.</p>
      </header>

      <ErrorBox error={settings.error} onDismiss={settings.reload} />
      {settings.loading ? <Loading /> : null}

      {settings.data ? (
        <>
          <div className="panel">
            <h3>Google Drive</h3>
            <p>
              {settings.data.drive.connected ? (
                <span className="badge auto_shortlist">connected</span>
              ) : (
                <span className="badge preliminary_reject">not connected</span>
              )}
            </p>
            <p className="small">
              Share your CV folders with this service account:{' '}
              <span className="mono">
                {settings.data.drive.service_account_email || '(none configured)'}
              </span>
            </p>
            {settings.data.drive.error ? (
              <p className="small muted">{settings.data.drive.error}</p>
            ) : null}
          </div>

          <div className="panel">
            <div className="row">
              <h3 className="grow">Gemini key pool</h3>
              <button className="secondary small" onClick={settings.reload}>
                Refresh
              </button>
            </div>
            <p className="small muted">
              Keys are shown by index and last four characters only — a full key never leaves the
              server or appears in a log.
            </p>
            <div className="stat-grid">
              <div className="stat">
                <div className="value">{settings.data.key_pool.size}</div>
                <div className="label">keys in the pool</div>
              </div>
              <div className="stat">
                <div className="value">{settings.data.key_pool.available}</div>
                <div className="label">available now</div>
              </div>
              <div className="stat">
                <div className="value">
                  {settings.data.key_pool.size - settings.data.key_pool.available}
                </div>
                <div className="label">cooling down</div>
              </div>
            </div>
            <div className="spacer" />
            <table>
              <thead>
                <tr>
                  <th>Key</th>
                  <th>Requests</th>
                  <th>Failures</th>
                  <th>Rate-limit hits</th>
                  <th>State</th>
                </tr>
              </thead>
              <tbody>
                {settings.data.key_pool.keys.map((key) => (
                  <tr key={key.index}>
                    <td className="mono">{key.label}</td>
                    <td>{key.requests}</td>
                    <td>{key.failures}</td>
                    <td>{key.rate_limit_hits}</td>
                    <td>
                      {key.cooling_down ? (
                        <span className="badge human_review">
                          cooling down {Math.ceil(key.cooldown_seconds_remaining)}s
                        </span>
                      ) : (
                        <span className="badge auto_shortlist">ready</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="panel">
            <h3>Models</h3>
            <table>
              <tbody>
                {Object.entries(settings.data.models).map(([role, model]) => (
                  <tr key={role}>
                    <td>{role}</td>
                    <td className="mono">{model}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="small muted">
              Mode: {settings.data.mock_mode ? 'mock (no credentials required)' : 'live Gemini'}
            </p>
          </div>

          <div className="panel">
            <h3>Default routing thresholds</h3>
            <p className="small muted">
              These are the defaults for a new job. Each job can override them on the Job
              Descriptions page.
            </p>
            <table>
              <tbody>
                {Object.entries(settings.data.defaults).map(([key, value]) => (
                  <tr key={key}>
                    <td>{key.replace(/_/g, ' ')}</td>
                    <td className="mono">{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </>
  )
}
