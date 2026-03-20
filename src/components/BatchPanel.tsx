/**
 * BatchPanel — Upload multiple videos, track them all, show combined results
 * and group comparison charts.
 */
import { useEffect, useRef, useState } from 'react'
import GroupChart, { type AnimalRow } from './GroupChart'

interface JobEntry {
  job_id: string
  filename: string
  status: 'queued' | 'running' | 'done' | 'error'
  progress: number
  animal_id?: string
  treatment?: string
  error?: string
}

interface Props {
  apiBase?: string
  onAllDone?: (count: number) => void
}

export default function BatchPanel({ apiBase = '', onAllDone }: Props) {
  const [jobs, setJobs] = useState<JobEntry[]>([])
  const [groupName, setGroupName] = useState('')
  const [nAnimals, setNAnimals] = useState(1)
  const [rows, setRows] = useState<AnimalRow[]>([])
  const [submitting, setSubmitting] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Poll all running/queued jobs; fire onAllDone when batch completes
  useEffect(() => {
    if (pollRef.current) clearInterval(pollRef.current)
    const pending = jobs.filter(j => j.status === 'queued' || j.status === 'running')
    if (pending.length === 0) return

    pollRef.current = setInterval(async () => {
      const updated = await Promise.all(
        pending.map(async j => {
          try {
            const r = await fetch(`${apiBase}/api/jobs/${j.job_id}`)
            if (!r.ok) return j
            const st = await r.json()
            return { ...j, status: st.status, progress: st.progress, error: st.message ?? undefined }
          } catch {
            return j
          }
        })
      )
      setJobs(prev => {
        const next = prev.map(j => {
          const u = updated.find(u => u.job_id === j.job_id)
          return u ?? j
        })
        // Fire notification when the last pending job finishes
        const stillPending = next.filter(j => j.status === 'queued' || j.status === 'running')
        if (stillPending.length === 0 && prev.some(j => j.status === 'queued' || j.status === 'running')) {
          const doneCount = next.filter(j => j.status === 'done').length
          onAllDone?.(doneCount)
        }
        return next
      })
    }, 800)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [jobs, apiBase, onAllDone])

  // When a job finishes, fetch its result and add to chart rows
  useEffect(() => {
    const done = jobs.filter(j => j.status === 'done' && !rows.find(r => r.job_id === j.job_id))
    if (done.length === 0) return
    done.forEach(async j => {
      try {
        const r = await fetch(`${apiBase}/api/jobs/${j.job_id}/result`)
        if (!r.ok) return
        const data = await r.json()
        const am = data.animal_meta ?? {}
        const row: AnimalRow = {
          job_id: j.job_id,
          animal_id: am.animal_id || j.filename.replace(/\.[^.]+$/, ''),
          treatment: am.treatment || groupName,
          trial: am.trial || '',
          metrics: data.metrics ?? {},
        }
        setRows(prev => [...prev.filter(r => r.job_id !== j.job_id), row])
      } catch { /* ignore */ }
    })
  }, [jobs, rows, apiBase, groupName])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const files = fileInputRef.current?.files
    if (!files || files.length === 0) return
    setSubmitting(true)
    try {
      const fd = new FormData()
      for (const f of files) fd.append('videos', f)
      fd.append('group_name', groupName)
      fd.append('n_animals', String(nAnimals))
      const r = await fetch(`${apiBase}/api/batch`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error(await r.text())
      const data = await r.json()
      const newJobs: JobEntry[] = data.jobs.map((j: { job_id: string; filename: string }) => ({
        job_id: j.job_id,
        filename: j.filename,
        status: 'queued' as const,
        progress: 0,
        treatment: groupName,
      }))
      setJobs(prev => [...prev, ...newJobs])
      if (fileInputRef.current) fileInputRef.current.value = ''
    } catch (e) {
      alert('Batch submit failed: ' + (e instanceof Error ? e.message : String(e)))
    } finally {
      setSubmitting(false)
    }
  }

  const doneCount = jobs.filter(j => j.status === 'done').length
  const totalJobs = jobs.length

  function downloadCombinedCSV() {
    const doneIds = jobs.filter(j => j.status === 'done').map(j => j.job_id)
    if (doneIds.length === 0) return
    window.open(`${apiBase}/api/batch/csv?job_ids=${doneIds.join(',')}`)
  }

  return (
    <div className="batchPanel">
      <div className="batchHeader">
        <div className="batchTitle">Batch processing</div>
        <div className="batchSub">Process multiple videos with the same arena & settings</div>
      </div>

      <form className="batchForm" onSubmit={handleSubmit}>
        <div className="batchFormRow">
          <label className="batchFormLabel">Videos</label>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            multiple
            className="batchFileInput"
          />
        </div>
        <div className="batchFormRow">
          <label className="batchFormLabel">Group / treatment</label>
          <input
            className="batchInput"
            placeholder="e.g. Control, Drug 10mg/kg"
            value={groupName}
            onChange={e => setGroupName(e.target.value)}
          />
        </div>
        <div className="batchFormRow">
          <label className="batchFormLabel">Animals per video</label>
          <select className="batchInput" value={nAnimals} onChange={e => setNAnimals(Number(e.target.value))}>
            <option value={1}>1 (single)</option>
            <option value={2}>2 (pair)</option>
            <option value={3}>3</option>
          </select>
        </div>
        <button type="submit" className="btnPrimary" disabled={submitting}>
          {submitting ? 'Submitting…' : '▶ Start batch'}
        </button>
      </form>

      {jobs.length > 0 && (
        <div className="batchJobList">
          <div className="batchJobListHeader">
            <span>{doneCount}/{totalJobs} complete</span>
            {doneCount > 0 && (
              <button className="exportBtn" onClick={downloadCombinedCSV}>
                ↓ Combined CSV
              </button>
            )}
          </div>
          {jobs.map(j => (
            <div key={j.job_id} className={`batchJobRow ${j.status}`}>
              <span className="batchJobName">{j.filename}</span>
              <span className="batchJobStatus">
                {j.status === 'done' ? '✓ Done'
                  : j.status === 'error' ? `✕ ${j.error ?? 'Error'}`
                  : j.status === 'running' ? `${(j.progress * 100).toFixed(0)}%`
                  : 'Queued'}
              </span>
              <div className="batchJobBar">
                <div className="batchJobFill" style={{
                  width: j.status === 'done' ? '100%' : `${(j.progress * 100).toFixed(1)}%`,
                  background: j.status === 'error' ? 'var(--red)' : j.status === 'done' ? 'var(--green)' : 'var(--accent)',
                }} />
              </div>
            </div>
          ))}
        </div>
      )}

      {rows.length >= 2 && (
        <div className="batchChartSection">
          <div className="batchChartTitle">Group comparison</div>
          <GroupChart rows={rows} />
        </div>
      )}
    </div>
  )
}
