/**
 * CumulativeZoneChart — line chart showing time-in-zone accumulating over the session.
 * Mirrors the live "zone counter" display in AnyMaze.
 */
import { useMemo } from 'react'

export type RawFrame = {
  t_sec: number
  ok: boolean
  zone_id?: string | null
}

interface Zone { id: string; name: string; color?: string }

interface Props {
  frames: RawFrame[]
  zones: Zone[]
  fps: number
}

const DEFAULT_COLORS = ['#00DDB4', '#ffd93d', '#ff6b6b', '#4d96ff', '#6bcb77', '#c77dff', '#ff9a3c']

export default function CumulativeZoneChart({ frames, zones, fps }: Props) {
  const { series, tMax } = useMemo(() => {
    if (!zones.length || !frames.length) return { series: [], tMax: 0 }

    const dt = fps > 0 ? 1 / fps : 0.04
    // Sample at most 500 points for perf
    const step = Math.max(1, Math.floor(frames.length / 500))

    const acc: Record<string, number> = {}
    zones.forEach(z => { acc[z.id] = 0 })

    const pts: Record<string, Array<{ t: number; v: number }>> = {}
    zones.forEach(z => { pts[z.id] = [] })

    for (let i = 0; i < frames.length; i++) {
      const f = frames[i]
      if (f.ok && f.zone_id && acc[f.zone_id] !== undefined) {
        acc[f.zone_id] += dt
      }
      if (i % step === 0) {
        for (const z of zones) {
          pts[z.id].push({ t: f.t_sec, v: acc[z.id] })
        }
      }
    }

    const tMax = frames[frames.length - 1]?.t_sec ?? 0
    return { series: zones.map((z, i) => ({
      zone: z,
      points: pts[z.id],
      color: z.color || DEFAULT_COLORS[i % DEFAULT_COLORS.length],
    })), tMax }
  }, [frames, zones, fps])

  if (!series.length) return null

  const SVG_W = 580, SVG_H = 220
  const PAD = { top: 16, right: 16, bottom: 40, left: 52 }
  const W = SVG_W - PAD.left - PAD.right
  const H = SVG_H - PAD.top - PAD.bottom

  const maxV = Math.max(...series.flatMap(s => s.points.map(p => p.v)), 1)

  const tx = (t: number) => PAD.left + (t / tMax) * W
  const ty = (v: number) => PAD.top + H - (v / maxV) * H

  const yTicks = [0, 0.25, 0.5, 0.75, 1.0].map(f => ({ v: maxV * f, label: `${(maxV * f).toFixed(0)}s` }))
  const xTicks = [0, 0.25, 0.5, 0.75, 1.0].map(f => ({ t: tMax * f, label: `${(tMax * f).toFixed(0)}s` }))

  return (
    <div style={{ marginTop: 8 }}>
      <svg width="100%" viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{ display: 'block' }}>
        {/* Grid */}
        {yTicks.map(({ v, label }) => (
          <g key={v}>
            <line x1={PAD.left} y1={ty(v)} x2={PAD.left + W} y2={ty(v)}
              stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
            <text x={PAD.left - 6} y={ty(v) + 4} textAnchor="end"
              fontSize={9} fill="rgba(255,255,255,0.35)">{label}</text>
          </g>
        ))}
        {xTicks.map(({ t, label }) => (
          <g key={t}>
            <line x1={tx(t)} y1={PAD.top} x2={tx(t)} y2={PAD.top + H}
              stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
            <text x={tx(t)} y={PAD.top + H + 14} textAnchor="middle"
              fontSize={9} fill="rgba(255,255,255,0.35)">{label}</text>
          </g>
        ))}

        {/* Lines */}
        {series.map(({ zone, points, color }) => {
          if (points.length < 2) return null
          const d = points.map((p, i) =>
            `${i === 0 ? 'M' : 'L'}${tx(p.t).toFixed(1)},${ty(p.v).toFixed(1)}`
          ).join(' ')
          return (
            <path key={zone.id} d={d} fill="none"
              stroke={color} strokeWidth={1.8} opacity={0.9} />
          )
        })}

        {/* Axes */}
        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + H}
          stroke="rgba(255,255,255,0.2)" strokeWidth={1} />
        <line x1={PAD.left} y1={PAD.top + H} x2={PAD.left + W} y2={PAD.top + H}
          stroke="rgba(255,255,255,0.2)" strokeWidth={1} />

        {/* Axis labels */}
        <text x={PAD.left / 2} y={PAD.top + H / 2} textAnchor="middle"
          fontSize={9} fill="rgba(255,255,255,0.4)"
          transform={`rotate(-90, ${PAD.left / 2}, ${PAD.top + H / 2})`}>
          Time in zone (s)
        </text>
        <text x={PAD.left + W / 2} y={SVG_H - 4} textAnchor="middle"
          fontSize={9} fill="rgba(255,255,255,0.4)">
          Session time (s)
        </text>
      </svg>

      {/* Legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 14px', marginTop: 4, paddingLeft: PAD.left }}>
        {series.map(({ zone, color }) => (
          <div key={zone.id} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--textSub)' }}>
            <div style={{ width: 20, height: 3, background: color, borderRadius: 2 }} />
            {zone.name}
          </div>
        ))}
      </div>
    </div>
  )
}
