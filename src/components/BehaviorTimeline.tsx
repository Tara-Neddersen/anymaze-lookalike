import { useMemo, useState } from 'react'

export type BehaviorSegment = {
  behavior: string
  start_t: number
  end_t: number
  duration_s: number
}

export type ZoneEvent = {
  zone_id: string
  zone_name: string
  entry_t: number
  exit_t: number
  duration_s: number
}

export type TTLEvent = {
  t: number
  label: string
}

type Props = {
  behaviorEvents: BehaviorSegment[]
  zoneEvents: ZoneEvent[]
  zoneColors: Record<string, string>
  duration: number
  currentTime: number
  onScrub: (t: number) => void
  ttlEvents?: TTLEvent[]
}

const BEHAVIOR_COLORS: Record<string, string> = {
  mobile:   '#22c55e',
  immobile: '#475569',
  freezing: '#3b82f6',
  rearing:  '#f97316',
}

const BEHAVIOR_LABELS: Record<string, string> = {
  mobile:   'Mobile',
  immobile: 'Immobile',
  freezing: 'Freezing',
  rearing:  'Rearing',
}

const ROW_H = 18
const ROW_GAP = 4
const LABEL_W = 86
const AXIS_H = 22
const MARKER_R = 6

function fmtTime(t: number): string {
  const m = Math.floor(t / 60)
  const s = Math.floor(t % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

export default function BehaviorTimeline({
  behaviorEvents, zoneEvents, zoneColors, duration,
  currentTime, onScrub, ttlEvents = [],
}: Props) {
  const [tooltip, setTooltip] = useState<{ text: string; x: number; y: number } | null>(null)

  const uniqueZones = useMemo(() => {
    const seen = new Set<string>()
    const out: { id: string; name: string }[] = []
    for (const ev of zoneEvents) {
      if (!seen.has(ev.zone_id)) { seen.add(ev.zone_id); out.push({ id: ev.zone_id, name: ev.zone_name }) }
    }
    return out
  }, [zoneEvents])

  // Row definitions: activity rows first, then one row per zone
  const rows = useMemo(() => {
    const r: { key: string; label: string; color: string }[] = []
    const behaviorsSeen = new Set(behaviorEvents.map(e => e.behavior))
    for (const b of ['mobile', 'immobile', 'freezing', 'rearing']) {
      if (behaviorsSeen.has(b)) r.push({ key: b, label: BEHAVIOR_LABELS[b], color: BEHAVIOR_COLORS[b] })
    }
    for (const z of uniqueZones) {
      r.push({ key: `zone:${z.id}`, label: z.name, color: zoneColors[z.id] || '#7c3aed' })
    }
    return r
  }, [behaviorEvents, uniqueZones, zoneColors])

  const totalH = rows.length * (ROW_H + ROW_GAP) + AXIS_H + 8
  const dur = Math.max(duration, 1)

  function handleClick(e: React.MouseEvent<SVGSVGElement>) {
    const rect = e.currentTarget.getBoundingClientRect()
    const svgW = rect.width
    const x = e.clientX - rect.left
    if (x < LABEL_W) return
    const t = ((x - LABEL_W) / (svgW - LABEL_W)) * dur
    onScrub(Math.max(0, Math.min(dur, t)))
  }

  function handleMouseMove(e: React.MouseEvent<SVGElement>, text: string) {
    const rect = e.currentTarget.getBoundingClientRect()
    setTooltip({ text, x: e.clientX - rect.left, y: e.clientY - rect.top - 28 })
  }

  if (!behaviorEvents.length && !zoneEvents.length) {
    return (
      <div style={{ padding: '18px', fontSize: 13, color: 'var(--textSub)', textAlign: 'center' }}>
        No behavioral data available yet
      </div>
    )
  }

  return (
    <div className="behaviorTimeline" style={{ position: 'relative', userSelect: 'none' }}>
      <svg
        width="100%"
        height={totalH}
        style={{ display: 'block', cursor: 'crosshair' }}
        onClick={handleClick}
        onMouseLeave={() => setTooltip(null)}
      >
        {rows.map((row, ri) => {
          const y = AXIS_H + ri * (ROW_H + ROW_GAP)
          const segments = row.key.startsWith('zone:')
            ? zoneEvents.filter(e => e.zone_id === row.key.slice(5))
            : behaviorEvents.filter(e => e.behavior === row.key)

          return (
            <g key={row.key}>
              {/* Row label */}
              <text x={LABEL_W - 6} y={y + ROW_H / 2 + 4}
                textAnchor="end" fontSize={11} fill="var(--textSub)" fontFamily="DM Sans, sans-serif">
                {row.label}
              </text>
              {/* Background track */}
              <rect x={LABEL_W} y={y} width="100%" height={ROW_H}
                fill="rgba(255,255,255,0.03)" rx={3} />
              {/* Segments — use viewBox trick for percentage width */}
              {segments.map((seg, si) => {
                const isZone = row.key.startsWith('zone:')
                const s = isZone ? (seg as ZoneEvent) : (seg as BehaviorSegment)
                const entryT = isZone ? (s as ZoneEvent).entry_t : (s as BehaviorSegment).start_t
                const exitT = isZone ? (s as ZoneEvent).exit_t : (s as BehaviorSegment).end_t
                return (
                  <rect
                    key={si}
                    x={`${(entryT / dur) * 100}%`}
                    y={y}
                    width={`${Math.max(0.2, ((exitT - entryT) / dur) * 100)}%`}
                    height={ROW_H}
                    rx={2}
                    fill={row.color}
                    opacity={0.82}
                    onMouseMove={e => handleMouseMove(e, `${row.label}: ${entryT.toFixed(1)}s – ${exitT.toFixed(1)}s (${(exitT - entryT).toFixed(1)}s)`)}
                    onMouseLeave={() => setTooltip(null)}
                    style={{ cursor: 'pointer' }}
                  />
                )
              })}
            </g>
          )
        })}

        {/* Time axis */}
        {Array.from({ length: Math.min(11, Math.ceil(dur / 10) + 1) }).map((_, i) => {
          const t = (i / 10) * dur
          return (
            <g key={i}>
              <line x1={`${(t / dur) * 100}%`} y1={AXIS_H - 4}
                x2={`${(t / dur) * 100}%`} y2={totalH - 2}
                stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
              <text x={`${(t / dur) * 100}%`} y={AXIS_H - 6}
                textAnchor="middle" fontSize={10} fill="var(--textSub)" fontFamily="DM Sans, sans-serif">
                {fmtTime(t)}
              </text>
            </g>
          )
        })}

        {/* TTL event markers */}
        {ttlEvents.map((ev, i) => (
          <g key={`ttl-${i}`}>
            <line
              x1={`${(ev.t / dur) * 100}%`} y1={AXIS_H}
              x2={`${(ev.t / dur) * 100}%`} y2={totalH}
              stroke="#facc15" strokeWidth={1.5} strokeDasharray="3 3" opacity={0.7} />
            <text
              x={`${(ev.t / dur) * 100}%`} y={AXIS_H - 6}
              textAnchor="middle" fontSize={9} fill="#facc15" fontFamily="DM Sans, sans-serif">
              {ev.label}
            </text>
          </g>
        ))}

        {/* Playhead */}
        <line
          x1={`${(currentTime / dur) * 100}%`} y1={AXIS_H}
          x2={`${(currentTime / dur) * 100}%`} y2={totalH}
          stroke="rgba(255,255,255,0.85)" strokeWidth={1.5} />
        <polygon
          points={`0,-${MARKER_R} ${MARKER_R},0 -${MARKER_R},0`}
          transform={`translate(${(currentTime / dur) * 100}% ${AXIS_H + MARKER_R})`}
          fill="white" />
      </svg>

      {/* Legend */}
      <div className="tlLegend">
        {Object.entries(BEHAVIOR_LABELS).map(([key, label]) => {
          if (!behaviorEvents.some(e => e.behavior === key)) return null
          return (
            <span key={key} className="tlLegendItem">
              <span className="tlLegendDot" style={{ background: BEHAVIOR_COLORS[key] }} />
              {label}
            </span>
          )
        })}
        {uniqueZones.map(z => (
          <span key={z.id} className="tlLegendItem">
            <span className="tlLegendDot" style={{ background: zoneColors[z.id] || '#7c3aed' }} />
            {z.name}
          </span>
        ))}
        {ttlEvents.length > 0 && (
          <span className="tlLegendItem">
            <span className="tlLegendDot" style={{ background: '#facc15' }} />
            TTL events
          </span>
        )}
      </div>

      {tooltip && (
        <div className="tlTooltip" style={{ left: tooltip.x, top: tooltip.y }}>
          {tooltip.text}
        </div>
      )}
    </div>
  )
}
