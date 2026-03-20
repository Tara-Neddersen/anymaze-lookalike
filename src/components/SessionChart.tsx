import { useMemo, useState } from 'react'

export interface SessionRow {
  job_id: string
  animal_id: string
  treatment: string
  session: string       // e.g. "Day 1", "Day 7"
  metrics: Record<string, number | null>
}

interface Props {
  rows: SessionRow[]
}

const METRIC_OPTIONS: { key: string; label: string; unit: string }[] = [
  { key: 'total_distance_cm', label: 'Total Distance', unit: 'cm' },
  { key: 'total_distance_px', label: 'Total Distance (px)', unit: 'px' },
  { key: 'mean_speed_cm_s', label: 'Mean Speed', unit: 'cm/s' },
  { key: 'mean_speed_px_s', label: 'Mean Speed (px/s)', unit: 'px/s' },
  { key: 'total_time_mobile_s', label: 'Time Mobile', unit: 's' },
  { key: 'total_time_immobile_s', label: 'Time Immobile', unit: 's' },
  { key: 'total_time_freezing_s', label: 'Time Freezing', unit: 's' },
  { key: 'thigmotaxis_fraction', label: 'Thigmotaxis', unit: '' },
  { key: 'path_efficiency', label: 'Path Efficiency', unit: '' },
  { key: 'clockwise_rotations', label: 'Clockwise Rotations', unit: '' },
  { key: 'total_time_rearing_s', label: 'Time Rearing', unit: 's' },
]

const COLORS = ['#00d9c8', '#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#c77dff', '#ff9f43']

export default function SessionChart({ rows }: Props) {
  const [metric, setMetric] = useState('total_distance_cm')

  const metaOpt = METRIC_OPTIONS.find(m => m.key === metric) ?? METRIC_OPTIONS[0]

  // Sorted unique sessions
  const sessions = useMemo(() => {
    const s = new Set(rows.map(r => r.session).filter(Boolean))
    return [...s].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
  }, [rows])

  // Group by animal_id
  const animals = useMemo(() => {
    const map = new Map<string, { treatment: string; pts: Map<string, number | null> }>()
    for (const r of rows) {
      if (!r.animal_id || !r.session) continue
      if (!map.has(r.animal_id)) map.set(r.animal_id, { treatment: r.treatment, pts: new Map() })
      map.get(r.animal_id)!.pts.set(r.session, r.metrics[metric] ?? null)
    }
    return [...map.entries()].map(([id, v], i) => ({
      id, treatment: v.treatment, pts: v.pts,
      color: COLORS[i % COLORS.length],
    }))
  }, [rows, metric])

  if (rows.length === 0) {
    return (
      <div className="schart-empty">
        <p>No session data yet.</p>
        <p className="schart-hint">
          Fill in the <strong>Session</strong> field (e.g. "Day 1", "Week 3") and run
          multiple analyses for the same Animal ID to build a longitudinal curve.
        </p>
      </div>
    )
  }

  if (sessions.length < 2) {
    return (
      <div className="schart-empty">
        <p>Need at least 2 sessions to draw a longitudinal chart.</p>
        <p className="schart-hint">Sessions detected: {sessions.join(', ') || 'none'}</p>
      </div>
    )
  }

  // Compute chart bounds
  const allVals = animals.flatMap(a =>
    sessions.map(s => a.pts.get(s) ?? null).filter((v): v is number => v !== null)
  )
  const minV = Math.min(0, ...allVals)
  const maxV = Math.max(...allVals, 0.001)

  const W = 520, H = 240
  const PAD = { top: 20, right: 20, bottom: 45, left: 55 }
  const plotW = W - PAD.left - PAD.right
  const plotH = H - PAD.top - PAD.bottom

  const xPos = (i: number) => PAD.left + (i / Math.max(1, sessions.length - 1)) * plotW
  const yPos = (v: number) => PAD.top + plotH - ((v - minV) / (maxV - minV)) * plotH

  // Y-axis ticks
  const yTicks = useMemo(() => {
    const n = 5
    const step = (maxV - minV) / n
    return Array.from({ length: n + 1 }, (_, i) => minV + i * step)
  }, [minV, maxV])

  return (
    <div className="schart-wrap">
      <div className="schart-toolbar">
        <label className="schart-label">Metric</label>
        <select className="schart-select" value={metric} onChange={e => setMetric(e.target.value)}>
          {METRIC_OPTIONS.map(m => (
            <option key={m.key} value={m.key}>{m.label}</option>
          ))}
        </select>
        <span className="schart-unit">{metaOpt.unit}</span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="schart-svg">
        {/* Grid lines */}
        {yTicks.map((v, i) => (
          <g key={i}>
            <line
              x1={PAD.left} y1={yPos(v)} x2={W - PAD.right} y2={yPos(v)}
              stroke="rgba(255,255,255,0.07)" strokeWidth={1}
            />
            <text x={PAD.left - 6} y={yPos(v) + 4} textAnchor="end"
              fontSize={9} fill="rgba(255,255,255,0.4)">
              {v.toFixed(1)}
            </text>
          </g>
        ))}

        {/* X-axis session labels */}
        {sessions.map((s, i) => (
          <text key={s} x={xPos(i)} y={H - PAD.bottom + 14}
            textAnchor="middle" fontSize={9} fill="rgba(255,255,255,0.5)">
            {s.length > 8 ? s.slice(0, 7) + '…' : s}
          </text>
        ))}

        {/* Axes */}
        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + plotH}
          stroke="rgba(255,255,255,0.2)" strokeWidth={1} />
        <line x1={PAD.left} y1={PAD.top + plotH} x2={W - PAD.right} y2={PAD.top + plotH}
          stroke="rgba(255,255,255,0.2)" strokeWidth={1} />

        {/* Lines per animal */}
        {animals.map(a => {
          const pts = sessions.map((s, i) => {
            const v = a.pts.get(s)
            return v != null ? { x: xPos(i), y: yPos(v), v } : null
          })
          let d = ''
          let prevPt: { x: number; y: number } | null = null
          const segments: string[] = []
          for (const pt of pts) {
            if (!pt) { prevPt = null; continue }
            if (prevPt) segments.push(`M ${prevPt.x} ${prevPt.y} L ${pt.x} ${pt.y}`)
            prevPt = pt
          }
          return (
            <g key={a.id}>
              {segments.map((seg, i) => (
                <path key={i} d={seg} stroke={a.color} strokeWidth={2}
                  fill="none" strokeLinejoin="round" />
              ))}
              {pts.map((pt, i) => pt && (
                <g key={i}>
                  <circle cx={pt.x} cy={pt.y} r={4} fill={a.color} />
                  <title>{a.id} @ {sessions[i]}: {pt.v.toFixed(2)}</title>
                </g>
              ))}
            </g>
          )
          void d
        })}
      </svg>

      {/* Legend */}
      <div className="schart-legend">
        {animals.map(a => (
          <div key={a.id} className="schart-legend-item">
            <span className="schart-legend-dot" style={{ background: a.color }} />
            <span>{a.id}{a.treatment ? ` (${a.treatment})` : ''}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
