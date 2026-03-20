import React, { useState, useEffect, useCallback } from 'react'

interface JobSummary {
  job_id: string
  filename?: string
  status?: string
  created_at?: string
  n_frames?: number
}

interface CohortAnimal {
  job_id: string
  animal_id: string
  genotype: string
  sex: string | null
  age_weeks: number | null
  treatment: string | null
}

interface Cohort {
  cohort_id: string
  name: string
  n_animals: number
  created_at: string
  pipeline_status: Record<string, string>
}

interface PipelineStep {
  id: string
  label: string
  endpoint: string
  description: string
  dependsOn?: string
}

const PIPELINE_STEPS: PipelineStep[] = [
  { id: 'pose_features',     label: '1. Pose Features',    endpoint: 'compute_pose_features', description: 'Normalise keypoints → 20-dim vectors' },
  { id: 'motif_discovery',   label: '2. Motif Discovery',  endpoint: 'discover_motifs',       description: 'k-means clustering of behavioral windows', dependsOn: 'pose_features' },
  { id: 'sequence_analysis', label: '3. Sequence Analysis',endpoint: 'compute_sequences',     description: 'Transition matrices & entropy', dependsOn: 'motif_discovery' },
  { id: 'embedding',         label: '4. UMAP Embedding',   endpoint: 'compute_embedding',     description: 'Per-frame behavioral landscape', dependsOn: 'pose_features' },
  { id: 'phenotypes',        label: '5. Phenotypes',       endpoint: 'compute_phenotypes',    description: 'Animal-level feature vectors', dependsOn: 'sequence_analysis' },
  { id: 'group_comparison',  label: '6. Group Comparison', endpoint: 'run_group_comparison',  description: "FDR-corrected statistics & Cohen's d", dependsOn: 'phenotypes' },
  { id: 'classifier',        label: '7. Classifier',       endpoint: 'run_classifier',        description: 'LOOCV genotype prediction', dependsOn: 'group_comparison' },
]

const GENOTYPE_COLORS: Record<string, string> = {
  WT:   '#4ade80',
  BPAN: '#f87171',
  KO:   '#fb923c',
  HET:  '#a78bfa',
}

function genotypeColor(g: string): string {
  return GENOTYPE_COLORS[g.toUpperCase()] ?? '#60a5fa'
}

const STEP_STATUS_ICON: Record<string, string> = {
  done:    '✓',
  running: '⟳',
  error:   '✗',
  pending: '○',
}

interface CohortBuilderProps {
  onCohortSelect?: (cohortId: string) => void
}

export default function CohortBuilder({ onCohortSelect }: CohortBuilderProps = {}) {
  const [cohorts,          setCohorts]         = useState<Cohort[]>([])
  const [selectedCohort,   setSelectedCohort]  = useState<string | null>(null)
  const [cohortDetail,     setCohortDetail]    = useState<any>(null)
  const [jobs,             setJobs]            = useState<JobSummary[]>([])
  const [newCohortName,    setNewCohortName]   = useState('')
  const [creating,         setCreating]        = useState(false)
  const [runningStep,      setRunningStep]     = useState<string | null>(null)
  const [stepResults,      setStepResults]     = useState<Record<string, any>>({})
  const [error,            setError]           = useState<string | null>(null)
  const [taskElapsed,      setTaskElapsed]     = useState<number>(0)
  const taskTimerRef = React.useRef<ReturnType<typeof setInterval> | null>(null)
  const [poseQcSummary,    setPoseQcSummary]   = useState<any>(null)
  const [labelingBusy,     setLabelingBusy]    = useState(false)

  // Form state for adding an animal
  const [addForm, setAddForm] = useState({
    job_id: '', animal_id: '', genotype: 'WT',
    sex: '', age_weeks: '', treatment: '',
  })

  const fetchCohorts = useCallback(async () => {
    try {
      const r = await fetch('/api/cohorts')
      const d = await r.json()
      setCohorts(d.cohorts ?? [])
    } catch { /* ignore */ }
  }, [])

  const fetchJobs = useCallback(async () => {
    try {
      const r = await fetch('/api/jobs')
      const d = await r.json()
      setJobs(d.jobs ?? [])
    } catch { /* ignore */ }
  }, [])

  const fetchCohortDetail = useCallback(async (id: string) => {
    try {
      const r = await fetch(`/api/cohorts/${id}`)
      const d = await r.json()
      setCohortDetail(d)
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    fetchCohorts()
    fetchJobs()
  }, [fetchCohorts, fetchJobs])

  useEffect(() => {
    if (selectedCohort) fetchCohortDetail(selectedCohort)
  }, [selectedCohort, fetchCohortDetail])

  const fetchPoseQcSummary = useCallback(async () => {
    if (!selectedCohort) return
    try {
      const r = await fetch(`/api/cohorts/${selectedCohort}/pose_qc_summary`)
      if (r.ok) setPoseQcSummary(await r.json())
      else setPoseQcSummary(null)
    } catch (e) {
      setPoseQcSummary(null)
    }
  }, [selectedCohort])

  useEffect(() => {
    void fetchPoseQcSummary()
  }, [fetchPoseQcSummary])

  const createCohort = async () => {
    if (!newCohortName.trim()) return
    setCreating(true)
    try {
      const r = await fetch(`/api/cohorts?name=${encodeURIComponent(newCohortName.trim())}`, { method: 'POST' })
      const d = await r.json()
      setNewCohortName('')
      await fetchCohorts()
      setSelectedCohort(d.cohort_id)
      onCohortSelect?.(d.cohort_id)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setCreating(false)
    }
  }

  const addAnimal = async () => {
    if (!selectedCohort || !addForm.job_id || !addForm.animal_id) return
    try {
      const body = {
        job_id: addForm.job_id,
        animal_id: addForm.animal_id,
        genotype: addForm.genotype,
        sex: addForm.sex || null,
        age_weeks: addForm.age_weeks ? parseFloat(addForm.age_weeks) : null,
        treatment: addForm.treatment || null,
      }
      await fetch(`/api/cohorts/${selectedCohort}/animals`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      setAddForm({ job_id: '', animal_id: '', genotype: 'WT', sex: '', age_weeks: '', treatment: '' })
      fetchCohortDetail(selectedCohort)
    } catch (e: any) {
      setError(e.message)
    }
  }

  const removeAnimal = async (job_id: string) => {
    if (!selectedCohort) return
    await fetch(`/api/cohorts/${selectedCohort}/animals/${job_id}`, { method: 'DELETE' })
    fetchCohortDetail(selectedCohort)
  }

  const pollTask = useCallback(async (cohortId: string, taskId: string): Promise<any> => {
    const POLL_INTERVAL = 2000
    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const r = await fetch(`/api/cohorts/${cohortId}/task_status/${taskId}`)
          const d = await r.json()
          if (d.status === 'done') return resolve(d.result)
          if (d.status === 'error') return reject(new Error(d.error ?? 'Task failed'))
          setTimeout(poll, POLL_INTERVAL)
        } catch (e) {
          reject(e)
        }
      }
      poll()
    })
  }, [])

  const runStep = async (step: PipelineStep) => {
    if (!selectedCohort || runningStep) return
    setRunningStep(step.id)
    setTaskElapsed(0)
    setError(null)

    // Start elapsed-time counter
    taskTimerRef.current = setInterval(() => setTaskElapsed(s => s + 1), 1000)

    try {
      const r = await fetch(`/api/cohorts/${selectedCohort}/${step.endpoint}`, { method: 'POST' })
      const d = await r.json()
      if (!r.ok) throw new Error(d.detail ?? 'Unknown error')

      // If the response contains a task_id, poll until completion
      const result = d.task_id ? await pollTask(selectedCohort, d.task_id) : d

      setStepResults(prev => ({ ...prev, [step.id]: result }))
      await fetchCohortDetail(selectedCohort)
      await fetchCohorts()
    } catch (e: any) {
      setError(`${step.label}: ${e.message}`)
    } finally {
      if (taskTimerRef.current) { clearInterval(taskTimerRef.current); taskTimerRef.current = null }
      setRunningStep(null)
    }
  }

  const runAllSteps = async () => {
    for (const step of PIPELINE_STEPS) {
      await runStep(step)
    }
  }

  const pipelineStatus: Record<string, string> = cohortDetail?.pipeline_status ?? {}

  function stepDisabled(step: PipelineStep): boolean {
    if (runningStep !== null) return true
    if (!cohortDetail?.animals?.length) return true
    if (step.dependsOn && pipelineStatus[step.dependsOn] !== 'done') return true
    return false
  }

  return (
    <div style={{ display: 'flex', gap: 20, height: '100%', minHeight: 0 }}>
      {/* Left: Cohort list */}
      <div style={{
        width: 220, flexShrink: 0,
        background: 'rgba(255,255,255,0.04)',
        borderRadius: 12, padding: 16,
        display: 'flex', flexDirection: 'column', gap: 12,
      }}>
        <div style={{ fontWeight: 700, color: '#e2e8f0', fontSize: 14 }}>Cohorts</div>

        {cohorts.map(c => (
          <div key={c.cohort_id}
            onClick={() => { setSelectedCohort(c.cohort_id); onCohortSelect?.(c.cohort_id) }}
            style={{
              padding: '8px 12px', borderRadius: 8, cursor: 'pointer',
              background: selectedCohort === c.cohort_id
                ? 'rgba(99,102,241,0.3)' : 'rgba(255,255,255,0.05)',
              border: `1px solid ${selectedCohort === c.cohort_id ? '#818cf8' : 'transparent'}`,
              transition: 'all 0.15s',
            }}>
            <div style={{ fontWeight: 600, fontSize: 13, color: '#e2e8f0' }}>{c.name}</div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
              {c.n_animals} animal{c.n_animals !== 1 ? 's' : ''}
            </div>
          </div>
        ))}

        {/* New cohort */}
        <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
          <input
            value={newCohortName}
            onChange={e => setNewCohortName(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && createCohort()}
            placeholder="New cohort name…"
            style={{
              background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: 6, padding: '6px 10px', color: '#e2e8f0', fontSize: 12,
            }}
          />
          <button onClick={createCohort} disabled={creating || !newCohortName.trim()}
            style={{
              background: '#6366f1', border: 'none', borderRadius: 6, color: '#fff',
              padding: '6px 0', cursor: 'pointer', fontSize: 12, fontWeight: 600,
              opacity: (creating || !newCohortName.trim()) ? 0.5 : 1,
            }}>
            {creating ? 'Creating…' : '+ Create'}
          </button>
        </div>
      </div>

      {/* Middle: Animals + Add form */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
        {selectedCohort && cohortDetail ? (
          <>
            <div style={{ fontWeight: 700, color: '#e2e8f0', fontSize: 16 }}>
              {cohortDetail.name}
              <span style={{ fontSize: 12, color: '#94a3b8', marginLeft: 12 }}>
                {cohortDetail.animals?.length ?? 0} animals
              </span>
            </div>

            {/* Animal table */}
            <div style={{
              background: 'rgba(255,255,255,0.04)', borderRadius: 10,
              overflow: 'hidden', flexShrink: 0,
            }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: 'rgba(255,255,255,0.06)' }}>
                    {['Job ID', 'Animal ID', 'Genotype', 'Sex', 'Age (wks)', 'Treatment', ''].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: '#94a3b8', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(cohortDetail.animals ?? []).map((a: CohortAnimal) => (
                    <tr key={a.job_id} style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                      <td style={{ padding: '8px 12px', color: '#94a3b8', fontFamily: 'monospace', fontSize: 11 }}>
                        {a.job_id.slice(0, 8)}…
                      </td>
                      <td style={{ padding: '8px 12px', color: '#e2e8f0' }}>{a.animal_id}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <span style={{
                          background: genotypeColor(a.genotype) + '22',
                          color: genotypeColor(a.genotype),
                          border: `1px solid ${genotypeColor(a.genotype)}44`,
                          borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600,
                        }}>{a.genotype}</span>
                      </td>
                      <td style={{ padding: '8px 12px', color: '#94a3b8' }}>{a.sex ?? '—'}</td>
                      <td style={{ padding: '8px 12px', color: '#94a3b8' }}>{a.age_weeks ?? '—'}</td>
                      <td style={{ padding: '8px 12px', color: '#94a3b8' }}>{a.treatment ?? '—'}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <button onClick={() => removeAnimal(a.job_id)}
                          style={{ background: 'rgba(239,68,68,0.15)', border: 'none', borderRadius: 4,
                            color: '#f87171', cursor: 'pointer', padding: '3px 8px', fontSize: 11 }}>
                          ✕
                        </button>
                      </td>
                    </tr>
                  ))}
                  {!(cohortDetail.animals?.length) && (
                    <tr><td colSpan={7} style={{ padding: '20px 12px', color: '#475569', textAlign: 'center' }}>
                      No animals yet. Add sessions below.
                    </td></tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Add animal form */}
            <div style={{
              background: 'rgba(255,255,255,0.04)', borderRadius: 10, padding: 16,
            }}>
              <div style={{ fontWeight: 600, color: '#94a3b8', fontSize: 12, marginBottom: 10 }}>
                Add Animal
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'flex-end' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <label style={{ fontSize: 10, color: '#64748b' }}>Session (Job ID)</label>
                  <select
                    value={addForm.job_id}
                    onChange={e => {
                      const jid = e.target.value
                      const job = jobs.find(j => j.job_id === jid)
                      setAddForm(f => ({ ...f, job_id: jid, animal_id: job?.filename?.replace(/\.[^.]+$/, '') ?? jid.slice(0, 8) }))
                    }}
                    style={inputStyle}>
                    <option value="">— select session —</option>
                    {jobs.map(j => (
                      <option key={j.job_id} value={j.job_id}>
                        {j.filename ?? j.job_id.slice(0, 8)}
                      </option>
                    ))}
                  </select>
                </div>

                <FormField label="Animal ID" value={addForm.animal_id}
                  onChange={v => setAddForm(f => ({ ...f, animal_id: v }))} />

                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <label style={{ fontSize: 10, color: '#64748b' }}>Genotype</label>
                  <select value={addForm.genotype}
                    onChange={e => setAddForm(f => ({ ...f, genotype: e.target.value }))}
                    style={inputStyle}>
                    {['WT', 'BPAN', 'KO', 'HET', 'custom'].map(g => <option key={g}>{g}</option>)}
                  </select>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <label style={{ fontSize: 10, color: '#64748b' }}>Sex</label>
                  <select value={addForm.sex}
                    onChange={e => setAddForm(f => ({ ...f, sex: e.target.value }))}
                    style={inputStyle}>
                    <option value="">—</option>
                    <option value="M">M</option>
                    <option value="F">F</option>
                  </select>
                </div>

                <FormField label="Age (weeks)" value={addForm.age_weeks}
                  type="number" onChange={v => setAddForm(f => ({ ...f, age_weeks: v }))} width={80} />
                <FormField label="Treatment" value={addForm.treatment}
                  onChange={v => setAddForm(f => ({ ...f, treatment: v }))} />

                <button onClick={addAnimal}
                  disabled={!addForm.job_id || !addForm.animal_id}
                  style={{
                    background: '#6366f1', border: 'none', borderRadius: 6,
                    color: '#fff', padding: '7px 16px', cursor: 'pointer',
                    fontSize: 12, fontWeight: 600, alignSelf: 'flex-end',
                    opacity: (!addForm.job_id || !addForm.animal_id) ? 0.5 : 1,
                  }}>
                  + Add
                </button>
              </div>
            </div>

            {/* Cohort pose QC summary */}
            <div style={{
              background: 'rgba(99,102,241,0.08)', borderRadius: 10, padding: 14,
              border: '1px solid rgba(99,102,241,0.2)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <span style={{ fontWeight: 700, color: '#a5b4fc', fontSize: 13 }}>Pose QC (cohort)</span>
                <button type="button" onClick={() => void fetchPoseQcSummary()}
                  style={{ background: 'rgba(255,255,255,0.08)', border: 'none', borderRadius: 4, color: '#94a3b8', fontSize: 11, cursor: 'pointer', padding: '2px 8px' }}>
                  Refresh
                </button>
              </div>
              {poseQcSummary?.per_animal && poseQcSummary.per_animal.length > 0 && (
                <>
                  <div style={{ fontSize: 11, color: poseQcSummary.tracking_quality_differs_by_genotype ? '#f87171' : '#4ade80', marginBottom: 8 }}>
                    Tracking quality differs by genotype (Mann–Whitney on valid fraction):{' '}
                    <strong>{poseQcSummary.tracking_quality_differs_by_genotype ? 'Yes — check confounds' : 'No clear difference'}</strong>
                    {poseQcSummary.valid_fraction_mannwhitney_p != null && (
                      <span style={{ color: '#94a3b8' }}> (p={poseQcSummary.valid_fraction_mannwhitney_p.toFixed(4)})</span>
                    )}
                  </div>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                    <thead>
                      <tr style={{ color: '#64748b' }}>
                        <th style={{ textAlign: 'left', padding: 4 }}>Animal</th>
                        <th style={{ textAlign: 'left', padding: 4 }}>Geno</th>
                        <th style={{ textAlign: 'left', padding: 4 }}>Tier</th>
                        <th style={{ textAlign: 'left', padding: 4 }}>Valid%</th>
                      </tr>
                    </thead>
                    <tbody>
                      {poseQcSummary.per_animal.map((row: any) => (
                        <tr key={row.job_id} style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                          <td style={{ padding: 4, color: '#e2e8f0' }}>{row.animal_id}</td>
                          <td style={{ padding: 4, color: '#94a3b8' }}>{row.genotype}</td>
                          <td style={{ padding: 4 }}>{row.qc_available ? row.tier : '—'}</td>
                          <td style={{ padding: 4 }}>
                            {row.qc_available && row.valid_frame_fraction_all7 != null
                              ? `${(row.valid_frame_fraction_all7 * 100).toFixed(0)}%`
                              : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div style={{ marginTop: 10, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    <button
                      type="button"
                      disabled={labelingBusy}
                      onClick={async () => {
                        if (!selectedCohort) return
                        setLabelingBusy(true)
                        try {
                          const r = await fetch(`/api/cohorts/${selectedCohort}/export_labeling_frames`, { method: 'POST' })
                          const d = await r.json()
                          if (!r.ok) throw new Error(d.detail ?? 'failed')
                          alert(`Exported ${d.n_selected} frames. Manifest: ${d.manifest_path}. ${d.recommended_first_round}`)
                        } catch (e: any) {
                          alert(e.message)
                        } finally {
                          setLabelingBusy(false)
                        }
                      }}
                      style={{
                        background: 'rgba(251,191,36,0.2)', border: '1px solid rgba(251,191,36,0.4)',
                        borderRadius: 6, color: '#fcd34d', fontSize: 11, padding: '6px 10px', cursor: 'pointer',
                      }}
                    >
                      {labelingBusy ? '…' : 'Export labeling manifest (stratified frames)'}
                    </button>
                  </div>
                </>
              )}
              {poseQcSummary && !poseQcSummary.per_animal?.length && (
                <div style={{ fontSize: 11, color: '#64748b' }}>No animals in cohort.</div>
              )}
              {!poseQcSummary && (
                <div style={{ fontSize: 11, color: '#64748b' }}>Run per-session Pose QC from results, then refresh.</div>
              )}
            </div>
          </>
        ) : (
          <div style={{ color: '#475569', textAlign: 'center', marginTop: 60, fontSize: 14 }}>
            Select or create a cohort to get started.
          </div>
        )}
      </div>

      {/* Right: Pipeline */}
      {selectedCohort && (
        <div style={{
          width: 260, flexShrink: 0,
          background: 'rgba(255,255,255,0.04)',
          borderRadius: 12, padding: 16,
          display: 'flex', flexDirection: 'column', gap: 10,
        }}>
          <div style={{ fontWeight: 700, color: '#e2e8f0', fontSize: 14 }}>Analysis Pipeline</div>

          {PIPELINE_STEPS.map(step => {
            const status = pipelineStatus[step.id] ?? 'pending'
            const disabled = stepDisabled(step)
            const isRunning = runningStep === step.id
            return (
              <div key={step.id} style={{
                background: status === 'done' ? 'rgba(74,222,128,0.08)'
                  : status === 'error' ? 'rgba(248,113,113,0.08)'
                  : status === 'running' ? 'rgba(99,102,241,0.12)'
                  : 'rgba(255,255,255,0.04)',
                border: `1px solid ${
                  status === 'done' ? 'rgba(74,222,128,0.3)'
                  : status === 'error' ? 'rgba(248,113,113,0.3)'
                  : status === 'running' ? 'rgba(99,102,241,0.4)'
                  : 'rgba(255,255,255,0.08)'}`,
                borderRadius: 8, padding: '10px 12px',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{
                      fontSize: 12, fontWeight: 600,
                      color: status === 'done' ? '#4ade80' : status === 'error' ? '#f87171' : '#e2e8f0',
                    }}>
                      {STEP_STATUS_ICON[status] ?? '○'} {step.label}
                    </div>
                    <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>{step.description}</div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    {isRunning && (
                      <span style={{ fontSize: 10, color: '#a5b4fc', fontVariantNumeric: 'tabular-nums' }}>
                        {taskElapsed}s
                      </span>
                    )}
                    <button
                      onClick={() => runStep(step)}
                      disabled={disabled}
                      style={{
                        background: disabled ? 'rgba(255,255,255,0.06)' : 'rgba(99,102,241,0.8)',
                        border: 'none', borderRadius: 5,
                        color: disabled ? '#475569' : '#fff',
                        padding: '4px 8px', cursor: disabled ? 'not-allowed' : 'pointer',
                        fontSize: 11, fontWeight: 600, flexShrink: 0,
                      }}>
                      {isRunning ? '⟳' : '▶'}
                    </button>
                  </div>
                </div>
                {stepResults[step.id] && status === 'done' && (
                  <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 6 }}>
                    {step.id === 'motif_discovery' && stepResults[step.id]?.k &&
                      `k=${stepResults[step.id].k}, stability=${stepResults[step.id].stability_score?.toFixed(2)}`}
                    {step.id === 'pose_features' &&
                      `${stepResults[step.id]?.results?.filter((r: any) => r.status === 'ok').length ?? 0} animals processed`}
                    {step.id === 'phenotypes' &&
                      `${stepResults[step.id]?.n_animals ?? 0} phenotype vectors`}
                    {step.id === 'classifier' &&
                      `LR AUC: ${stepResults[step.id]?.logistic_regression?.roc_auc ?? '—'}`}
                  </div>
                )}
              </div>
            )
          })}

          <button onClick={runAllSteps} disabled={runningStep !== null || !cohortDetail?.animals?.length}
            style={{
              marginTop: 4,
              background: 'linear-gradient(135deg,#6366f1,#8b5cf6)',
              border: 'none', borderRadius: 8, color: '#fff',
              padding: '10px 0', cursor: 'pointer', fontSize: 13,
              fontWeight: 700, width: '100%',
              opacity: (runningStep !== null || !cohortDetail?.animals?.length) ? 0.5 : 1,
            }}>
            {runningStep ? `Running ${runningStep}… (${taskElapsed}s)` : '▶▶ Run All Steps'}
          </button>

          {error && (
            <div style={{
              background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
              borderRadius: 6, padding: '8px 10px', fontSize: 11, color: '#fca5a5',
            }}>
              {error}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function FormField({
  label, value, onChange, type = 'text', width = 110,
}: {
  label: string, value: string, onChange: (v: string) => void,
  type?: string, width?: number,
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <label style={{ fontSize: 10, color: '#64748b' }}>{label}</label>
      <input
        type={type} value={value} onChange={e => onChange(e.target.value)}
        style={{ ...inputStyle, width }}
      />
    </div>
  )
}

const inputStyle: React.CSSProperties = {
  background: 'rgba(255,255,255,0.07)',
  border: '1px solid rgba(255,255,255,0.15)',
  borderRadius: 6, padding: '6px 10px',
  color: '#e2e8f0', fontSize: 12, minWidth: 70,
}
