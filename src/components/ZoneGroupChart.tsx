/**
 * ZoneGroupChart — compares a zone metric (time in / entries / latency)
 * across treatment groups for every zone that appears in the dataset.
 */
import { useMemo, useState } from 'react'

export type ZoneRow = {
  animal_id: string
  treatment: string
  zones: { zone_id: string; zone_name: string; time_in_s: number; entries: number; latency_first_entry_s: number | null }[]
}

type ZoneMetric = 'time_in_s' | 'entries' | 'latency_first_entry_s'

const METRIC_LABELS: Record<ZoneMetric, string> = {
  time_in_s: 'Time in zone (s)',
  entries: 'Entries',
  latency_first_entry_s: 'Latency to first entry (s)',
}

const GROUP_COLORS = ['#00d9c8', '#0084ff', '#ff6b6b', '#ffd93d', '#6bcb77', '#c77dff', '#ff9a3c']

function groupColor(idx: number) { return GROUP_COLORS[idx % GROUP_COLORS.length] }

interface Props { rows: ZoneRow[] }

export default function ZoneGroupChart({ rows }: Props) {
  const [metric, setMetric] = useState<ZoneMetric>('time_in_s')

  const { zones, groups, matrix } = useMemo(() => {
    // Collect all unique zone names
    const zoneMap = new Map<string, string>() // id -> name
    for (const r of rows) for (const z of r.zones) zoneMap.set(z.zone_id, z.zone_name)
    const zones = [...zoneMap.entries()].map(([id, name]) => ({ id, name }))

    // Collect treatment groups
    const groupSet = new Set(rows.map(r => r.treatment || 'Unknown'))
    const groups = [...groupSet]

    // matrix[zoneIdx][groupIdx] = values[]
    const matrix: number[][][] = zones.map(() => groups.map(() => []))
    for (const r of rows) {
      const gi = groups.indexOf(r.treatment || 'Unknown')
      for (const z of r.zones) {
        const zi = zones.findIndex(zz => zz.id === z.zone_id)
        if (zi < 0 || gi < 0) continue
        const val = metric === 'latency_first_entry_s'
          ? z.latency_first_entry_s
          : (metric === 'entries' ? z.entries : z.time_in_s)
        if (val != null) matrix[zi][gi].push(val)
      }
    }

    return { zones, groups, matrix }
  }, [rows, metric])

  if (zones.length === 0) {
    return <div className="zgcEmpty">No zone data in this dataset</div>
  }

  const allVals = matrix.flat(2)
  const maxVal = Math.max(...allVals, 0.001)
  const W = Math.max(500, zones.length * groups.length * 22 + 80)
  const H = 200
  const PAD = { top: 14, right: 14, bottom: 50, left: 48 }
  const chartW = W - PAD.left - PAD.right
  const chartH = H - PAD.top - PAD.bottom
  const zoneW = chartW / Math.max(1, zones.length)
  const barW = Math.min(18, (zoneW / Math.max(1, groups.length)) * 0.75)

  function mean(vals: number[]) { return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0 }
  function sem(vals: number[]) {
    if (vals.length < 2) return 0
    const m = mean(vals)
    return Math.sqrt(vals.reduce((a, v) => a + (v - m) ** 2, 0) / (vals.length - 1)) / Math.sqrt(vals.length)
  }

  const yTicks = 4
  const yVals = Array.from({ length: yTicks + 1 }, (_, i) => (maxVal * i) / yTicks)

  return (
    <div className="zgcWrap">
      <div className="zgcToolbar">
        <label className="zgcLabel">Metric</label>
        <select className="zgcSelect" value={metric} onChange={e => setMetric(e.target.value as ZoneMetric)}>
          {(Object.keys(METRIC_LABELS) as ZoneMetric[]).map(k => (
            <option key={k} value={k}>{METRIC_LABELS[k]}</option>
          ))}
        </select>
        {/* Group legend */}
        <div className="zgcLegend">
          {groups.map((g, gi) => (
            <span key={g} className="zgcLegendItem">
              <span className="zgcLegendDot" style={{ background: groupColor(gi) }} />
              {g}
            </span>
          ))}
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <svg width={W} height={H} className="zgcSvg">
          {/* Y grid */}
          {yVals.map(v => {
            const y = PAD.top + chartH - (v / maxVal) * chartH
            return (
              <g key={v}>
                <line x1={PAD.left} x2={PAD.left + chartW} y1={y} y2={y}
                  stroke="rgba(255,255,255,0.05)" strokeWidth={1} />
                <text x={PAD.left - 5} y={y + 4} textAnchor="end" fontSize={8}
                  fill="rgba(255,255,255,0.3)">
                  {v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(v < 1 ? 1 : 0)}
                </text>
              </g>
            )
          })}

          {/* Zone groups */}
          {zones.map((z, zi) => {
            const zoneX = PAD.left + zi * zoneW
            const midX = zoneX + zoneW / 2
            return (
              <g key={z.id}>
                {/* Zone separator */}
                {zi > 0 && (
                  <line x1={zoneX} x2={zoneX} y1={PAD.top} y2={PAD.top + chartH}
                    stroke="rgba(255,255,255,0.06)" strokeWidth={1} strokeDasharray="3,3" />
                )}
                {/* Bars per group */}
                {groups.map((g, gi) => {
                  const vals = matrix[zi][gi]
                  const m = mean(vals)
                  const s = sem(vals)
                  const barH = Math.max(0, (m / maxVal) * chartH)
                  const x = midX - (groups.length / 2 - gi - 0.5) * (barW + 2) - barW / 2
                  const y = PAD.top + chartH - barH
                  const col = groupColor(gi)
                  return (
                    <g key={g}>
                      <rect x={x} y={y} width={barW} height={barH}
                        fill={col} fillOpacity={0.75} rx={2} />
                      {s > 0 && (() => {
                        const cx = x + barW / 2
                        const top = PAD.top + chartH - ((m + s) / maxVal) * chartH
                        const bot = PAD.top + chartH - (Math.max(0, m - s) / maxVal) * chartH
                        return (
                          <g stroke={col} strokeWidth={1.2}>
                            <line x1={cx} x2={cx} y1={top} y2={bot} />
                            <line x1={cx - 3} x2={cx + 3} y1={top} y2={top} />
                            <line x1={cx - 3} x2={cx + 3} y1={bot} y2={bot} />
                          </g>
                        )
                      })()}
                    </g>
                  )
                })}
                {/* Zone label */}
                <text x={midX} y={PAD.top + chartH + 14} textAnchor="middle" fontSize={9}
                  fill="rgba(255,255,255,0.55)"
                  transform={z.name.length > 8
                    ? `rotate(-30, ${midX}, ${PAD.top + chartH + 14})` : undefined}>
                  {z.name.length > 12 ? z.name.slice(0, 11) + '…' : z.name}
                </text>
              </g>
            )
          })}

          {/* Axes */}
          <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={PAD.top + chartH}
            stroke="rgba(255,255,255,0.12)" strokeWidth={1} />
          <line x1={PAD.left} x2={PAD.left + chartW} y1={PAD.top + chartH} y2={PAD.top + chartH}
            stroke="rgba(255,255,255,0.12)" strokeWidth={1} />

          {/* Y-axis label */}
          <text x={12} y={PAD.top + chartH / 2} textAnchor="middle" fontSize={8}
            fill="rgba(255,255,255,0.35)"
            transform={`rotate(-90,12,${PAD.top + chartH / 2})`}>
            {METRIC_LABELS[metric]}
          </text>
        </svg>
      </div>
      <div className="zgcNote">Bars = mean ± SEM per treatment group</div>
    </div>
  )
}
