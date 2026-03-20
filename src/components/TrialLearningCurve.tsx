/**
 * TrialLearningCurve — escape latency (or any metric) plotted across trials
 * for each treatment group. Useful for MWM, fear conditioning, etc.
 */
import { useMemo } from 'react'

export type TrialRow = {
  animal_id: string
  treatment: string
  trial: string           // "Trial 1", "Day 1", "1", etc.
  metric_value: number    // e.g. latency_first_entry_s or total_time_freezing_s
}

interface Props {
  rows: TrialRow[]
  metricLabel?: string
}

const COLORS = ['#00DDB4', '#ff6b6b', '#ffd93d', '#4d96ff', '#6bcb77', '#c77dff', '#ff9a3c']

function sortTrialLabel(label: string): number {
  const m = label.match(/\d+/)
  return m ? parseInt(m[0]) : 0
}

export default function TrialLearningCurve({ rows, metricLabel = 'Metric' }: Props) {
  const { groups, trialLabels } = useMemo(() => {
    const treatmentSet = new Set(rows.map(r => r.treatment || 'All'))
    const trialSet = new Set(rows.map(r => r.trial))
    const trialLabels = [...trialSet].sort((a, b) => sortTrialLabel(a) - sortTrialLabel(b))

    const groups = [...treatmentSet].map((treatment, i) => {
      const tRows = rows.filter(r => (r.treatment || 'All') === treatment)
      const means = trialLabels.map(tl => {
        const vals = tRows.filter(r => r.trial === tl).map(r => r.metric_value)
        if (!vals.length) return null
        return vals.reduce((a, b) => a + b, 0) / vals.length
      })
      return { treatment, means, color: COLORS[i % COLORS.length] }
    })

    return { groups, trialLabels }
  }, [rows])

  if (!rows.length || trialLabels.length < 2) return (
    <div style={{ fontSize: 12, color: 'var(--textMuted)', padding: '12px 0' }}>
      Need ≥ 2 trials to plot the learning curve. Tag sessions with Trial 1, Trial 2… in the metadata form.
    </div>
  )

  const SVG_W = 560, SVG_H = 200
  const PAD = { top: 16, right: 16, bottom: 36, left: 52 }
  const W = SVG_W - PAD.left - PAD.right
  const H = SVG_H - PAD.top - PAD.bottom

  const allVals = groups.flatMap(g => g.means.filter((v): v is number => v !== null))
  const maxV = Math.max(...allVals, 1)

  const tx = (i: number) => PAD.left + (i / (trialLabels.length - 1)) * W
  const ty = (v: number) => PAD.top + H - (v / maxV) * H

  const yTicks = [0, 0.5, 1.0].map(f => ({ v: maxV * f, label: `${(maxV * f).toFixed(1)}` }))

  return (
    <div>
      <svg width="100%" viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{ display: 'block' }}>
        {yTicks.map(({ v, label }) => (
          <g key={v}>
            <line x1={PAD.left} y1={ty(v)} x2={PAD.left + W} y2={ty(v)}
              stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
            <text x={PAD.left - 6} y={ty(v) + 4} textAnchor="end" fontSize={9} fill="rgba(255,255,255,0.35)">{label}</text>
          </g>
        ))}

        {trialLabels.map((label, i) => (
          <g key={label}>
            <line x1={tx(i)} y1={PAD.top} x2={tx(i)} y2={PAD.top + H}
              stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
            <text x={tx(i)} y={PAD.top + H + 14} textAnchor="middle" fontSize={9} fill="rgba(255,255,255,0.4)">
              {label.length > 8 ? label.slice(0, 8) : label}
            </text>
          </g>
        ))}

        {groups.map(({ treatment, means, color }) => {
          const validPts = means
            .map((v, i) => ({ i, v }))
            .filter((p): p is { i: number; v: number } => p.v !== null)

          if (validPts.length < 1) return null
          const d = validPts.map(({ i, v }, j) =>
            `${j === 0 ? 'M' : 'L'}${tx(i).toFixed(1)},${ty(v).toFixed(1)}`
          ).join(' ')
          return (
            <g key={treatment}>
              <path d={d} fill="none" stroke={color} strokeWidth={2} />
              {validPts.map(({ i, v }) => (
                <circle key={i} cx={tx(i)} cy={ty(v)} r={4} fill={color} />
              ))}
            </g>
          )
        })}

        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + H}
          stroke="rgba(255,255,255,0.2)" strokeWidth={1} />
        <line x1={PAD.left} y1={PAD.top + H} x2={PAD.left + W} y2={PAD.top + H}
          stroke="rgba(255,255,255,0.2)" strokeWidth={1} />

        <text x={PAD.left / 2} y={PAD.top + H / 2} textAnchor="middle" fontSize={9}
          fill="rgba(255,255,255,0.4)"
          transform={`rotate(-90, ${PAD.left / 2}, ${PAD.top + H / 2})`}>
          {metricLabel}
        </text>
      </svg>

      <div style={{ display: 'flex', gap: '6px 14px', flexWrap: 'wrap', marginTop: 4, paddingLeft: PAD.left }}>
        {groups.map(({ treatment, color }) => (
          <div key={treatment} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--textSub)' }}>
            <div style={{ width: 20, height: 3, background: color, borderRadius: 2 }} />
            {treatment}
          </div>
        ))}
      </div>
    </div>
  )
}
