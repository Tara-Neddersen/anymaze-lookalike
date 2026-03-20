import { useCallback, useEffect, useState } from 'react'

type QcDecision = {
  tier: string
  label: string
  recommendation: string
  reasons?: string[]
  metric_grades?: Record<string, string>
}

type PoseQCReport = {
  schema_version?: string
  confidence_threshold?: number
  likelihood_assumed?: boolean
  computed_at?: string
  per_keypoint: Record<string, { missing_fraction: number; mean_likelihood_when_present: number }>
  session: {
    n_frames: number
    valid_frame_fraction_all7: number
    body_length_px: { median: number; iqr: number; cv: number }
    body_length_outlier_fraction: number
    jitter_px: {
      mean_of_median_per_kp: number
      per_keypoint: Record<string, number>
      jitter_norm_vs_body_length: number
    }
    max_gap_frames: number
    gaps_ge_5_frames: number
  }
  decision: QcDecision
}

export default function PoseQCPanel({
  jobId,
  hasPose,
}: {
  jobId: string | null
  hasPose: boolean
}) {
  const [qc, setQc] = useState<PoseQCReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const fetchQc = useCallback(async () => {
    if (!jobId || !hasPose) return
    setLoading(true)
    setErr(null)
    try {
      const r = await fetch(`/api/jobs/${jobId}/pose_qc`)
      if (r.status === 404) {
        setQc(null)
        return
      }
      if (!r.ok) throw new Error(await r.text())
      setQc(await r.json())
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [jobId, hasPose])

  useEffect(() => { void fetchQc() }, [fetchQc])

  const runCompute = async () => {
    if (!jobId) return
    setLoading(true)
    setErr(null)
    try {
      const r = await fetch(`/api/jobs/${jobId}/pose_qc`, { method: 'POST' })
      if (!r.ok) throw new Error(await r.text())
      setQc(await r.json())
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  if (!hasPose || !jobId) {
    return (
      <div style={{ padding: 16, color: '#94a3b8', fontSize: 13 }}>
        Pose QC is available when tracking uses DeepLabCut or SLEAP with a pose file.
      </div>
    )
  }

  const tier = qc?.decision?.tier ?? '?'
  const tierColor =
    tier === 'A' ? '#4ade80' : tier === 'B' ? '#fbbf24' : tier === 'C' ? '#f87171' : '#94a3b8'

  return (
    <div style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
        <div style={{ fontWeight: 700, color: '#e2e8f0', fontSize: 15 }}>Pose quality control</div>
        <button
          type="button"
          onClick={() => void runCompute()}
          disabled={loading}
          style={{
            background: 'rgba(99,102,241,0.85)', border: 'none', borderRadius: 6,
            color: '#fff', padding: '6px 12px', cursor: loading ? 'wait' : 'pointer', fontSize: 12, fontWeight: 600,
          }}
        >
          {loading ? '…' : qc ? 'Recompute QC' : 'Run pose QC'}
        </button>
      </div>

      {err && (
        <div style={{ background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.35)', borderRadius: 6, padding: 8, fontSize: 12, color: '#fca5a5' }}>
          {err}
        </div>
      )}

      {!qc && !loading && (
        <p style={{ fontSize: 12, color: '#94a3b8', margin: 0 }}>
          Compute QC to see valid-frame fraction, per-keypoint confidence, body-length stability, jitter, and a tier (A/B/C) recommendation.
        </p>
      )}

      {qc && (
        <>
          <div style={{
            borderRadius: 10,
            border: `2px solid ${tierColor}`,
            background: 'rgba(15,23,42,0.6)',
            padding: '12px 14px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
              <span style={{
                fontSize: 22, fontWeight: 800, color: tierColor, fontFamily: 'monospace',
              }}>{tier}</span>
              <span style={{ fontWeight: 600, color: '#e2e8f0', textTransform: 'capitalize' }}>{qc.decision.label}</span>
            </div>
            <p style={{ margin: 0, fontSize: 13, color: '#cbd5e1', lineHeight: 1.45 }}>{qc.decision.recommendation}</p>
            {qc.likelihood_assumed && (
              <p style={{ margin: '8px 0 0', fontSize: 11, color: '#fbbf24' }}>
                Some keypoints had no likelihood in the file; assumed 1.0 for QC.
              </p>
            )}
          </div>

          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ color: '#64748b', textAlign: 'left' }}>
                <th style={{ padding: '4px 6px' }}>Metric</th>
                <th style={{ padding: '4px 6px' }}>Value</th>
              </tr>
            </thead>
            <tbody style={{ color: '#e2e8f0' }}>
              <tr><td style={{ padding: 4 }}>Valid frames (all 7 KPs ≥ τ)</td><td>{(qc.session.valid_frame_fraction_all7 * 100).toFixed(1)}%</td></tr>
              <tr><td style={{ padding: 4 }}>τ (confidence)</td><td>{qc.confidence_threshold ?? '—'}</td></tr>
              <tr><td style={{ padding: 4 }}>Body length CV</td><td>{qc.session.body_length_px.cv.toFixed(4)}</td></tr>
              <tr><td style={{ padding: 4 }}>Body length outliers</td><td>{(qc.session.body_length_outlier_fraction * 100).toFixed(1)}%</td></tr>
              <tr><td style={{ padding: 4 }}>Jitter (mean of KP medians, px)</td><td>{qc.session.jitter_px.mean_of_median_per_kp.toFixed(2)}</td></tr>
              <tr><td style={{ padding: 4 }}>Jitter / body length</td><td>{qc.session.jitter_px.jitter_norm_vs_body_length.toFixed(4)}</td></tr>
              <tr><td style={{ padding: 4 }}>Max gap (frames)</td><td>{qc.session.max_gap_frames}</td></tr>
              <tr><td style={{ padding: 4 }}>Gaps ≥ 5 frames</td><td>{qc.session.gaps_ge_5_frames}</td></tr>
            </tbody>
          </table>

          <div style={{ fontWeight: 600, color: '#94a3b8', fontSize: 12 }}>Per keypoint</div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ color: '#64748b', textAlign: 'left' }}>
                <th style={{ padding: 4 }}>KP</th>
                <th style={{ padding: 4 }}>Missing</th>
                <th style={{ padding: 4 }}>Mean lik (present)</th>
              </tr>
            </thead>
            <tbody style={{ color: '#e2e8f0' }}>
              {Object.entries(qc.per_keypoint).map(([k, v]) => (
                <tr key={k}>
                  <td style={{ padding: 4 }}>{k}</td>
                  <td style={{ padding: 4 }}>{(v.missing_fraction * 100).toFixed(1)}%</td>
                  <td style={{ padding: 4 }}>{v.mean_likelihood_when_present.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <a
              href={`/api/jobs/${jobId}/pose_qc`}
              download={`pose_qc_${jobId.slice(0, 8)}.json`}
              style={{ fontSize: 12, color: '#818cf8' }}
            >Download pose_qc.json</a>
            <a
              href={`/api/jobs/${jobId}/pose_qc/mask`}
              download="pose_valid_mask.npy"
              style={{ fontSize: 12, color: '#818cf8' }}
            >Download valid mask (.npy)</a>
          </div>
        </>
      )}
    </div>
  )
}
