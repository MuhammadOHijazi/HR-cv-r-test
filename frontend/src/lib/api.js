/**
 * The single API client module.
 *
 * Every network call the UI makes goes through `request`, so error handling and
 * the base URL live in exactly one place.
 */

const BASE = import.meta.env.VITE_API_BASE ?? ''

export class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === 'string' ? detail : JSON.stringify(detail))
    this.status = status
    this.detail = detail
  }
}

async function request(path, { method = 'GET', body } = {}) {
  const response = await fetch(`${BASE}${path}`, {
    method,
    headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })

  const text = await response.text()
  let payload = null
  if (text) {
    try {
      payload = JSON.parse(text)
    } catch {
      payload = text
    }
  }

  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : payload || response.statusText
    throw new ApiError(response.status, detail)
  }
  return payload
}

function query(params) {
  const entries = Object.entries(params ?? {}).filter(
    ([, value]) => value !== undefined && value !== null && value !== '',
  )
  return entries.length ? `?${new URLSearchParams(entries).toString()}` : ''
}

export const api = {
  health: () => request('/api/health'),
  settings: () => request('/api/settings'),

  // --- jobs and job descriptions -----------------------------------------
  listJobs: () => request('/api/jobs'),
  getJob: (id) => request(`/api/jobs/${id}`),
  createJob: (title, rawJdText) =>
    request('/api/jobs', { method: 'POST', body: { title, raw_jd_text: rawJdText } }),
  structureJd: (id) => request(`/api/jobs/${id}/structure`, { method: 'POST' }),
  listVersions: (id) => request(`/api/jobs/${id}/versions`),
  editVersion: (id, version, structured) =>
    request(`/api/jobs/${id}/versions/${version}`, { method: 'PUT', body: { structured } }),
  approveVersion: (id, version, actor = 'recruiter') =>
    request(`/api/jobs/${id}/versions/${version}/approve`, { method: 'POST', body: { actor } }),
  getActiveJd: (id) => request(`/api/jobs/${id}/jd`),
  getConfig: (id) => request(`/api/jobs/${id}/config`),
  updateConfig: (id, config) => request(`/api/jobs/${id}/config`, { method: 'PUT', body: config }),

  // --- drive ---------------------------------------------------------------
  driveStatus: () => request('/api/drive/status'),
  listFolders: () => request('/api/drive/folders'),
  refreshFolders: () => request('/api/drive/folders/refresh', { method: 'POST' }),
  getJobFolders: (id) => request(`/api/jobs/${id}/folders`),
  assignFolders: (id, folderIds) =>
    request(`/api/jobs/${id}/folders`, { method: 'POST', body: { folder_ids: folderIds } }),
  sync: (id) => request(`/api/jobs/${id}/sync`, { method: 'POST' }),
  syncStatus: (id) => request(`/api/jobs/${id}/sync/status`),

  // --- screening -----------------------------------------------------------
  screen: (id) => request(`/api/jobs/${id}/screen`, { method: 'POST' }),
  results: (id, filters) => request(`/api/jobs/${id}/results${query(filters)}`),
  candidate: (candidateId, jobId) =>
    request(`/api/candidates/${candidateId}${query({ job_id: jobId })}`),

  // --- review --------------------------------------------------------------
  rejectReasons: () => request('/api/review/reasons'),
  reviewQueue: (id) => request(`/api/jobs/${id}/review`),
  approveReview: (entryId, note = '') =>
    request(`/api/review/${entryId}/approve`, { method: 'POST', body: { note } }),
  rejectReview: (entryId, reason, note = '') =>
    request(`/api/review/${entryId}/reject`, { method: 'POST', body: { reason, note } }),
  correctReview: (entryId, corrections, note = '') =>
    request(`/api/review/${entryId}/correct`, { method: 'POST', body: { corrections, note } }),
  preliminaryRejects: (id) => request(`/api/jobs/${id}/preliminary-rejects`),
  confirmRejects: (id, resultIds) =>
    request(`/api/jobs/${id}/confirm-rejects`, { method: 'POST', body: { result_ids: resultIds } }),
}

export const ROUTING_LABELS = {
  auto_shortlist: 'Shortlisted',
  human_review: 'Needs review',
  preliminary_reject: 'Preliminary reject',
  rejected: 'Rejected',
}

export const FLAG_LABELS = {
  low_ocr_quality: 'Low source quality',
  missing_critical_field: 'Missing critical field',
  stated_vs_computed_years_conflict: 'Years contradiction',
  scorer_disagreement: 'Scorers disagree',
  injection_suspicion: 'Possible prompt injection',
  high_score_weak_evidence: 'High score, weak evidence',
  must_have_failure_on_low_confidence_field: 'Must-have gap on unreliable data',
  extraction_failed: 'Extraction failed',
  low_confidence: 'Low confidence',
  borderline_score: 'Borderline score',
  no_dated_work_history: 'No dated work history',
  high_confidence_must_have_failure: 'Must-have not met',
}
