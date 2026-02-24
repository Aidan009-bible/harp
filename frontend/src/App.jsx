import { useState, useRef, useCallback } from 'react'
import './App.css'

const API = import.meta.env.VITE_API_URL || '/api'

export default function App() {
  const [modelFile, setModelFile] = useState(null)
  const [videoFile, setVideoFile] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [method, setMethod] = useState('audio')
  const [mode, setMode] = useState('hybrid')
  const [weightsFile, setWeightsFile] = useState(null)
  const modelInput = useRef(null)
  const videoInput = useRef(null)
  const weightsInput = useRef(null)

  const pollStatus = useCallback(async (id) => {
    try {
      const res = await fetch(`${API}/status/${id}`)
      if (!res.ok) throw new Error('Status check failed')
      const data = await res.json()
      setStatus(data)
      if (data.status === 'done' || data.status === 'error') return
      setTimeout(() => pollStatus(id), 1500)
    } catch {
      setTimeout(() => pollStatus(id), 3000)
    }
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!videoFile) { setError('Please select a video file.'); return }
    if ((method === 'audio' || method === 'both') && !modelFile) {
      setError('Please select a .keras model file.'); return
    }
    setError(null)
    setStatus(null)
    setUploading(true)
    try {
      const form = new FormData()
      form.append('method', method)
      form.append('video', videoFile)
      if (method === 'audio' || method === 'both') {
        form.append('model', modelFile)
        form.append('mode', mode)
      }
      if ((method === 'hand' || method === 'both') && weightsFile) {
        form.append('weights', weightsFile)
      }
      const res = await fetch(`${API}/upload`, { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText)
      }
      const { job_id } = await res.json()
      setJobId(job_id)
      pollStatus(job_id)
    } catch (err) {
      setError(err.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const reset = () => {
    setModelFile(null)
    setVideoFile(null)
    setWeightsFile(null)
    setJobId(null)
    setStatus(null)
    setError(null)
    if (modelInput.current) modelInput.current.value = ''
    if (videoInput.current) videoInput.current.value = ''
    if (weightsInput.current) weightsInput.current.value = ''
  }

  const downloadCsv = (type) => {
    if (!jobId) return
    window.open(type ? `${API}/download/csv/${jobId}?type=${type}` : `${API}/download/csv/${jobId}`, '_blank')
  }

  const downloadVideo = (type) => {
    if (!jobId) return
    window.open(type ? `${API}/download/video/${jobId}?type=${type}` : `${API}/download/video/${jobId}`, '_blank')
  }

  /* ── Helpers ── */
  const needsModel = method === 'audio' || method === 'both'
  const needsWeights = method === 'hand' || method === 'both'
  const isDone = status?.status === 'done'
  const isRunning = status?.status === 'running' || status?.status === 'queued'
  const isError = status?.status === 'error'

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-badge">
          <span className="dot" />
          AI-Powered Detection
        </div>
        <h1>HarpHand</h1>
        <p>Detect harp strings with audio models &amp; computer vision</p>
      </header>

      <main className="main">
        {/* ── Upload Card ── */}
        <section className="card">
          <div className="card-header">
            <div className="card-icon">🎵</div>
            <div>
              <h2>New Detection</h2>
              <p>Upload a video to analyze</p>
            </div>
          </div>

          {/* Method Tabs */}
          <div className="method-tabs">
            <button
              type="button"
              className={`method-tab ${method === 'audio' ? 'active' : ''}`}
              onClick={() => setMethod('audio')}
            >
              <span className="tab-icon">🎧</span>
              Audio
            </button>
            <button
              type="button"
              className={`method-tab ${method === 'hand' ? 'active' : ''}`}
              onClick={() => setMethod('hand')}
            >
              <span className="tab-icon">✋</span>
              Hand
            </button>
            <button
              type="button"
              className={`method-tab ${method === 'both' ? 'active' : ''}`}
              onClick={() => setMethod('both')}
            >
              <span className="tab-icon">⚡</span>
              Both
            </button>
          </div>

          <form onSubmit={handleSubmit} className="form-section">
            {/* Model upload */}
            {needsModel && (
              <div className="field">
                <label className="field-label">Keras Model (.keras)</label>
                <div className={`file-drop ${modelFile ? 'has-file' : ''}`}>
                  <input
                    ref={modelInput}
                    type="file"
                    accept=".keras"
                    onChange={(e) => setModelFile(e.target.files?.[0] ?? null)}
                  />
                  <div className="file-drop-icon">{modelFile ? '✅' : '🧠'}</div>
                  <div className="file-drop-text">
                    {modelFile
                      ? <span className="file-name">📄 {modelFile.name}</span>
                      : <span>Drop your <strong>.keras</strong> model here or click to browse</span>}
                  </div>
                </div>
              </div>
            )}

            {/* Video upload */}
            <div className="field">
              <label className="field-label">Video File</label>
              <div className={`file-drop ${videoFile ? 'has-file' : ''}`}>
                <input
                  ref={videoInput}
                  type="file"
                  accept=".mp4,.mov,.mkv,.avi,.webm"
                  onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)}
                />
                <div className="file-drop-icon">{videoFile ? '✅' : '🎬'}</div>
                <div className="file-drop-text">
                  {videoFile
                    ? <span className="file-name">📄 {videoFile.name}</span>
                    : <span>Drop your <strong>video</strong> here (.mp4, .mov, .mkv, .avi, .webm)</span>}
                </div>
              </div>
            </div>

            {/* YOLO weights */}
            {needsWeights && (
              <div className="field">
                <label className="field-label">YOLO Weights (.pt) — Optional</label>
                <div className={`file-drop ${weightsFile ? 'has-file' : ''}`}>
                  <input
                    ref={weightsInput}
                    type="file"
                    accept=".pt"
                    onChange={(e) => setWeightsFile(e.target.files?.[0] ?? null)}
                  />
                  <div className="file-drop-icon">{weightsFile ? '✅' : '⚙️'}</div>
                  <div className="file-drop-text">
                    {weightsFile
                      ? <span className="file-name">📄 {weightsFile.name}</span>
                      : <span>Optional — uses <strong>best.pt</strong> from server if not provided</span>}
                  </div>
                </div>
              </div>
            )}

            {/* Audio mode selector */}
            {needsModel && (
              <div className="field">
                <label className="field-label">Audio Mode</label>
                <div className="mode-toggle">
                  <button
                    type="button"
                    className={`mode-btn ${mode === 'default' ? 'active' : ''}`}
                    onClick={() => setMode('default')}
                  >
                    Default (model only)
                  </button>
                  <button
                    type="button"
                    className={`mode-btn ${mode === 'hybrid' ? 'active' : ''}`}
                    onClick={() => setMode('hybrid')}
                  >
                    Hybrid (model + YIN)
                  </button>
                </div>
              </div>
            )}

            {error && <div className="error-msg">⚠️ {error}</div>}

            <button type="submit" className="btn-submit" disabled={uploading || isRunning}>
              {uploading ? '⏳ Uploading…' : isRunning ? '⏳ Processing…' : '🚀 Run Detection'}
            </button>
          </form>
        </section>

        {/* ── Status / Results Card ── */}
        {status && (
          <section className="card status-card">
            <div className="card-header">
              <div className="card-icon" style={isDone ? { background: 'linear-gradient(135deg, #00cec9, #55efc4)' } : undefined}>
                {isDone ? '✅' : isError ? '❌' : '⏳'}
              </div>
              <div>
                <h2>{isDone ? 'Detection Complete' : isError ? 'Error' : 'Processing'}</h2>
                <p>{isDone ? 'Your results are ready!' : isError ? 'Something went wrong' : 'Analyzing your video…'}</p>
              </div>
            </div>

            <div className="status-content">
              {/* Running state */}
              {isRunning && (
                <>
                  <div className="loading-bar">
                    <div className="loading-bar-inner" />
                  </div>
                  <div className="status-text">
                    <span className="spinner" />
                    {status.message || 'Processing…'}
                  </div>
                </>
              )}

              {/* Error state */}
              {isError && (
                <div className="error-msg">⚠️ {status.message}</div>
              )}

              {/* Done + BOTH mode */}
              {isDone && status.audio && status.hand && (
                <>
                  <div className="result-summary">
                    <span className="result-icon">🎯</span>
                    <div className="result-info">
                      <h3>Analysis Complete</h3>
                      <p>{status.audio.rows ?? 0} audio onset(s) • {status.hand.rows ?? 0} hand touch(es)</p>
                    </div>
                  </div>
                  {status.combined_error && (
                    <div className="hand-error">⚠️ Combined video: {status.combined_error}</div>
                  )}
                  <div className="download-grid">
                    {status.combined && (
                      <button type="button" className="btn-download combined" onClick={() => downloadVideo('combined')}>
                        📹 Combined Video (hand + audio)
                      </button>
                    )}
                    <button type="button" className="btn-download primary" onClick={() => downloadCsv('audio')}>
                      📊 Audio CSV
                    </button>
                    <button type="button" className="btn-download" onClick={() => downloadVideo('audio')}>
                      🎬 Audio Video
                    </button>
                    <button type="button" className="btn-download primary" onClick={() => downloadCsv('hand')}>
                      📊 Hand CSV
                    </button>
                    <button type="button" className="btn-download" onClick={() => downloadVideo('hand')}>
                      🎬 Hand Video
                    </button>
                  </div>
                </>
              )}

              {/* Done + audio-only from "both" */}
              {isDone && status.audio && !status.hand && (
                <>
                  <div className="result-summary">
                    <span className="result-icon">🎵</span>
                    <div className="result-info">
                      <h3>Audio Analysis Complete</h3>
                      <p>{status.audio.rows ?? 0} onset(s) detected</p>
                    </div>
                  </div>
                  {status.hand_error && (
                    <div className="hand-error">⚠️ Hand detection: {status.hand_error}</div>
                  )}
                  <div className="download-grid">
                    <button type="button" className="btn-download primary" onClick={() => downloadCsv('audio')}>
                      📊 Download CSV
                    </button>
                    <button type="button" className="btn-download" onClick={() => downloadVideo('audio')}>
                      🎬 Download Video
                    </button>
                  </div>
                </>
              )}

              {/* Done + single method (audio or hand) */}
              {isDone && !status.audio && (
                <>
                  <div className="result-summary">
                    <span className="result-icon">{method === 'hand' ? '✋' : '🎵'}</span>
                    <div className="result-info">
                      <h3>Detection Complete</h3>
                      <p>{status.rows ?? 0} {method === 'hand' ? 'touch event(s)' : 'onset(s)'} detected</p>
                    </div>
                  </div>
                  <div className="download-grid">
                    <button type="button" className="btn-download primary" onClick={() => downloadCsv()}>
                      📊 Download CSV
                    </button>
                    <button type="button" className="btn-download" onClick={() => downloadVideo()}>
                      🎬 Download {method === 'hand' ? 'Annotated' : 'Labeled'} Video
                    </button>
                  </div>
                </>
              )}

              {/* New run button */}
              {(isDone || isError) && (
                <button type="button" className="btn-new-run" onClick={reset}>
                  ↻ Start New Detection
                </button>
              )}
            </div>
          </section>
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="footer">
        <p>HarpHand — AI-Powered Harp String Detection</p>
        <div className="footer-links">
          <span>Audio</span>
          <span>•</span>
          <span>Hand</span>
          <span>•</span>
          <span>Both</span>
        </div>
      </footer>
    </div>
  )
}
