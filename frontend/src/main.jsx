import React from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter, Navigate, NavLink, Route, Routes } from 'react-router-dom'

import Dashboard from './pages/Dashboard.jsx'
import JobDescriptions from './pages/JobDescriptions.jsx'
import DriveFolders from './pages/DriveFolders.jsx'
import Results from './pages/Results.jsx'
import ReviewQueue from './pages/ReviewQueue.jsx'
import Settings from './pages/Settings.jsx'
import './styles.css'

const NAV = [
  ['/dashboard', 'Dashboard'],
  ['/jobs', 'Job Descriptions'],
  ['/drive', 'Drive Folders'],
  ['/results', 'Results'],
  ['/review', 'Review Queue'],
  ['/settings', 'Settings'],
]

function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>
          CV Screening
          <br />
          <span className="muted small">the LLM reads, rules decide, humans judge</span>
        </h1>
        <nav>
          {NAV.map(([path, label]) => (
            <NavLink key={path} to={path} className={({ isActive }) => (isActive ? 'active' : '')}>
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/jobs" element={<JobDescriptions />} />
          <Route path="/drive" element={<DriveFolders />} />
          <Route path="/results" element={<Results />} />
          <Route path="/review" element={<ReviewQueue />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </main>
    </div>
  )
}

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <HashRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <App />
    </HashRouter>
  </React.StrictMode>,
)
