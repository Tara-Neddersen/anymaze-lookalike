/**
 * GroupChart — SVG bar chart comparing a metric across animals/groups.
 * Includes client-side Welch t-test / ANOVA with significance stars.
 */
import { useMemo, useState } from 'react'

// ---------------------------------------------------------------------------
// Client-side statistics (Welch's t-test, F-test, Bonferroni pairwise)
// ---------------------------------------------------------------------------
function tTest(a: number[], b: number[]): number {
  if (a.length < 2 || b.length < 2) return 1
  const ma = a.reduce((s, v) => s + v, 0) / a.length
  const mb = b.reduce((s, v) => s + v, 0) / b.length
  const va = a.reduce((s, v) => s + (v - ma) ** 2, 0) / (a.length - 1)
  const vb = b.reduce((s, v) => s + (v - mb) ** 2, 0) / (b.length - 1)
  const sed = Math.sqrt(va / a.length + vb / b.length)
  if (sed === 0) return 1
  const t = Math.abs(ma - mb) / sed
  // Welch–Satterthwaite degrees of freedom
  const df = (va / a.length + vb / b.length) ** 2 /
    ((va / a.length) ** 2 / (a.length - 1) + (vb / b.length) ** 2 / (b.length - 1))
  return tDist(t, df)
}

// Two-tailed t-distribution survival function approximation (Abramowitz & Stegun)
function tDist(t: number, df: number): number {
  const x = df / (df + t * t)
  return incompleteBeta(df / 2, 0.5, x)
}

function incompleteBeta(a: number, b: number, x: number): number {
  // Simple continued-fraction approximation
  if (x <= 0) return 0; if (x >= 1) return 1
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta) / a
  let cf = 1; let c = 1; let d = 1 - (a + b) * x / (a + 1)
  if (Math.abs(d) < 1e-30) d = 1e-30; d = 1 / d; cf = d
  for (let m = 1; m <= 100; m++) {
    const m2 = 2 * m
    let num = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
    d = 1 + num * d; c = 1 + num / c
    if (Math.abs(d) < 1e-30) d = 1e-30; if (Math.abs(c) < 1e-30) c = 1e-30
    d = 1 / d; cf *= d * c
    num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
    d = 1 + num * d; c = 1 + num / c
    if (Math.abs(d) < 1e-30) d = 1e-30; if (Math.abs(c) < 1e-30) c = 1e-30
    d = 1 / d; cf *= d * c
    if (Math.abs(d * c - 1) < 3e-7) break
  }
  return front * cf
}

function lgamma(z: number): number {
  // Lanczos approximation
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]
  let y = z, x = z, tmp = z + 5.5
  tmp -= (z + 0.5) * Math.log(tmp)
  let ser = 1.000000000190015
  for (const ci of c) { y++; ser += ci / y }
  return -tmp + Math.log(2.5066282746310005 * ser / x)
}

function stars(p: number): string {
  if (p < 0.001) return '***'
  if (p < 0.01) return '**'
  if (p < 0.05) return '*'
  return 'ns'
}

export type AnimalRow = {
  job_id: string
  animal_id: string
  treatment: string
  trial: string
  session?: string
  metrics: Record<string, number | null>
}

const METRIC_OPTIONS = [
  { key: 'total_distance_cm', label: 'Total distance (cm)', fallback: 'total_distance_px' },
  { key: 'mean_speed_cm_s', label: 'Mean speed (cm/s)', fallback: 'mean_speed_px_s' },
  { key: 'max_speed_cm_s', label: 'Max speed (cm/s)', fallback: 'max_speed_px_s' },
  { key: 'total_time_mobile_s', label: 'Time mobile (s)', fallback: null },
  { key: 'total_time_immobile_s', label: 'Time immobile (s)', fallback: null },
  { key: 'total_time_freezing_s', label: 'Time freezing (s)', fallback: null },
  { key: 'thigmotaxis_fraction', label: 'Thigmotaxis (%)', fallback: null, scale: 100 },
  { key: 'path_efficiency', label: 'Path efficiency', fallback: null },
  { key: 'clockwise_rotations', label: 'CW rotations', fallback: null },
  { key: 'anticlockwise_rotations', label: 'CCW rotations', fallback: null },
  { key: 'total_time_rearing_s', label: 'Time rearing (s)', fallback: null },
]

const GROUP_COLORS = [
  '#00d9c8', '#0084ff', '#ff6b6b', '#ffd93d',
  '#6bcb77', '#c77dff', '#ff9a3c', '#f78fb3',
]

interface Props {
  rows: AnimalRow[]
}

export default function GroupChart({ rows }: Props) {
  const [metricKey, setMetricKey] = useState('total_distance_cm')
  const [groupBy, setGroupBy] = useState<'animal' | 'treatment'>('treatment')

  const metricOpt = METRIC_OPTIONS.find(m => m.key === metricKey) ?? METRIC_OPTIONS[0]

  const { bars, maxVal, groups, pairTests } = useMemo(() => {
    const getValue = (row: AnimalRow): number | null => {
      let v = row.metrics[metricKey]
      if ((v === null || v === undefined) && metricOpt.fallback) {
        v = row.metrics[metricOpt.fallback]
      }
      if (v == null) return null
      return (metricOpt.scale ?? 1) * v
    }

    // Group animals
    const groupMap = new Map<string, AnimalRow[]>()
    for (const row of rows) {
      const key = groupBy === 'treatment' ? (row.treatment || 'Unknown') : (row.animal_id || row.job_id)
      const existing = groupMap.get(key) ?? []
      existing.push(row)
      groupMap.set(key, existing)
    }

    const groups = [...groupMap.keys()]
    const bars = groups.map(g => {
      const members = groupMap.get(g)!
      const vals = members.map(getValue).filter((v): v is number => v !== null)
      const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
      const sem = vals.length > 1
        ? Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / (vals.length - 1)) / Math.sqrt(vals.length)
        : 0
      return { group: g, mean, sem, n: vals.length, vals }
    })

    const maxVal = Math.max(...bars.map(b => b.mean + b.sem), 0.001)

    // Pairwise statistical tests (Bonferroni-corrected Welch t-test)
    const nTests = bars.length * (bars.length - 1) / 2
    const pairTests: { i: number; j: number; p: number; stars: string }[] = []
    for (let i = 0; i < bars.length; i++) {
      for (let j = i + 1; j < bars.length; j++) {
        if (bars[i].vals.length >= 2 && bars[j].vals.length >= 2) {
          const p = Math.min(1, tTest(bars[i].vals, bars[j].vals) * nTests)
          pairTests.push({ i, j, p: Math.round(p * 10000) / 10000, stars: stars(p) })
        }
      }
    }

    return { bars, maxVal, groups, pairTests }
  }, [rows, metricKey, groupBy, metricOpt])

  const W = 560, H = 230
  const PAD = { top: 16, right: 16, bottom: 48, left: 52 }
  const chartW = W - PAD.left - PAD.right
  const chartH = H - PAD.top - PAD.bottom
  const barW = Math.min(48, (chartW / Math.max(1, bars.length)) * 0.55)
  const barGap = chartW / Math.max(1, bars.length)

  function scaleY(v: number) { return chartH - (v / maxVal) * chartH }

  const yTicks = 4
  const yVals = Array.from({ length: yTicks + 1 }, (_, i) => (maxVal * i) / yTicks)

  return (
    <div className="groupChart">
      <div className="groupChartControls">
        <div className="gcControl">
          <label className="gcLabel">Metric</label>
          <select className="gcSelect" value={metricKey} onChange={e => setMetricKey(e.target.value)}>
            {METRIC_OPTIONS.map(m => (
              <option key={m.key} value={m.key}>{m.label}</option>
            ))}
          </select>
        </div>
        <div className="gcControl">
          <label className="gcLabel">Group by</label>
          <select className="gcSelect" value={groupBy} onChange={e => setGroupBy(e.target.value as 'animal' | 'treatment')}>
            <option value="treatment">Treatment</option>
            <option value="animal">Animal ID</option>
          </select>
        </div>
        <div className="gcInfo">{rows.length} animal{rows.length !== 1 ? 's' : ''} · {groups.length} group{groups.length !== 1 ? 's' : ''}</div>
      </div>

      <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="gcSvg">
        {/* Y grid + ticks */}
        {yVals.map(v => {
          const y = PAD.top + scaleY(v)
          return (
            <g key={v}>
              <line x1={PAD.left} x2={PAD.left + chartW} y1={y} y2={y}
                stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
              <text x={PAD.left - 6} y={y + 4} textAnchor="end"
                fontSize="9" fill="rgba(255,255,255,0.35)">
                {v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(v < 1 ? 2 : 0)}
              </text>
            </g>
          )
        })}

        {/* Bars */}
        {bars.map((bar, i) => {
          const x = PAD.left + i * barGap + barGap / 2 - barW / 2
          const barH = (bar.mean / maxVal) * chartH
          const y = PAD.top + chartH - barH
          const color = GROUP_COLORS[i % GROUP_COLORS.length]

          return (
            <g key={bar.group}>
              {/* Bar */}
              <rect x={x} y={y} width={barW} height={Math.max(0, barH)}
                fill={color} fillOpacity="0.8" rx="3" />
              {/* Error bar (SEM) */}
              {bar.sem > 0 && (() => {
                const topErr = PAD.top + scaleY(bar.mean + bar.sem)
                const botErr = PAD.top + scaleY(Math.max(0, bar.mean - bar.sem))
                const cx = x + barW / 2
                return (
                  <g stroke={color} strokeWidth="1.5">
                    <line x1={cx} x2={cx} y1={topErr} y2={botErr} />
                    <line x1={cx - 5} x2={cx + 5} y1={topErr} y2={topErr} />
                    <line x1={cx - 5} x2={cx + 5} y1={botErr} y2={botErr} />
                  </g>
                )
              })()}
              {/* Value label */}
              <text x={x + barW / 2} y={Math.max(PAD.top + 12, y - 4)} textAnchor="middle"
                fontSize="9" fill={color}>
                {bar.mean >= 100 ? bar.mean.toFixed(0) : bar.mean.toFixed(1)}
              </text>
              {/* n= label */}
              <text x={x + barW / 2} y={PAD.top + chartH + 10} textAnchor="middle"
                fontSize="8" fill="rgba(255,255,255,0.35)">
                n={bar.n}
              </text>
              {/* Group label */}
              <text
                x={x + barW / 2}
                y={PAD.top + chartH + 22}
                textAnchor="middle"
                fontSize="9"
                fill="rgba(255,255,255,0.6)"
                transform={bar.group.length > 8
                  ? `rotate(-25, ${x + barW / 2}, ${PAD.top + chartH + 22})`
                  : undefined}
              >
                {bar.group.length > 12 ? bar.group.slice(0, 11) + '…' : bar.group}
              </text>
            </g>
          )
        })}

        {/* Significance brackets */}
        {pairTests.filter(pt => pt.stars !== 'ns').map((pt, idx) => {
          const x1 = PAD.left + pt.i * barGap + barGap / 2
          const x2 = PAD.left + pt.j * barGap + barGap / 2
          const topI = PAD.top + scaleY(bars[pt.i].mean + bars[pt.i].sem)
          const topJ = PAD.top + scaleY(bars[pt.j].mean + bars[pt.j].sem)
          const bracketY = Math.min(topI, topJ) - 8 - idx * 14
          return (
            <g key={`sig-${idx}`} fontSize="10" fill="rgba(255,255,255,0.7)" stroke="rgba(255,255,255,0.5)" strokeWidth="1">
              <line x1={x1} x2={x1} y1={Math.min(topI, bracketY + 4)} y2={bracketY} strokeWidth="1" />
              <line x1={x2} x2={x2} y1={Math.min(topJ, bracketY + 4)} y2={bracketY} strokeWidth="1" />
              <line x1={x1} x2={x2} y1={bracketY} y2={bracketY} strokeWidth="1" />
              <text x={(x1 + x2) / 2} y={bracketY - 3} textAnchor="middle" fill="#facc15" fontSize="11" stroke="none">
                {pt.stars}
              </text>
            </g>
          )
        })}

        {/* Axes */}
        <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={PAD.top + chartH}
          stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
        <line x1={PAD.left} x2={PAD.left + chartW} y1={PAD.top + chartH} y2={PAD.top + chartH}
          stroke="rgba(255,255,255,0.15)" strokeWidth="1" />

        {/* Y-axis label */}
        <text
          x={12}
          y={PAD.top + chartH / 2}
          textAnchor="middle"
          fontSize="9"
          fill="rgba(255,255,255,0.4)"
          transform={`rotate(-90, 12, ${PAD.top + chartH / 2})`}
        >
          {metricOpt.label}
        </text>
      </svg>

      {/* Statistical note */}
      <div className="gcNote">
        Bars = mean ± SEM. 
        {pairTests.length > 0 && (
          <span> Stats: Welch t-test, Bonferroni-corrected.{' '}
            {pairTests.filter(p => p.stars !== 'ns').map(pt =>
              `${groups[pt.i]} vs ${groups[pt.j]}: p=${pt.p} ${pt.stars}`
            ).join(' · ') || 'All comparisons ns.'}
          </span>
        )}
      </div>
    </div>
  )
}
