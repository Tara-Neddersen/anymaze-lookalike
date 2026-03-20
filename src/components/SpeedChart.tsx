import { useRef, useMemo } from 'react'

interface Frame {
  t_sec: number
  ok: boolean
  speed_cm_s?: number | null
  speed_px_s?: number | null
}

interface Props {
  frames: Frame[]
  useCm: boolean
  currentTime: number
  duration: number
  onScrub?: (t: number) => void
}

const IMMOBILE_CM = 2.0
const FREEZE_CM = 0.5
// Must match backend IMMOBILITY_THRESHOLD_PX_S / FREEZING_THRESHOLD_PX_S in metrics.py
const IMMOBILE_PX = 8.0
const FREEZE_PX = 2.0

export default function SpeedChart({ frames, useCm, currentTime, duration, onScrub }: Props) {
  const svgRef = useRef<SVGSVGElement>(null)
  const W = 800
  const H = 100
  const PAD = { top: 8, right: 8, bottom: 20, left: 38 }
  const chartW = W - PAD.left - PAD.right
  const chartH = H - PAD.top - PAD.bottom

  const { points, maxSpeed, immThr, freezeThr } = useMemo(() => {
    const validFrames = frames.filter(f => f.ok && (useCm ? f.speed_cm_s != null : f.speed_px_s != null))
    const speeds = validFrames.map(f => useCm ? (f.speed_cm_s ?? 0) : (f.speed_px_s ?? 0))
    const maxS = speeds.length ? Math.max(...speeds) : 1
    const immThr = useCm ? IMMOBILE_CM : IMMOBILE_PX
    const freezeThr = useCm ? FREEZE_CM : FREEZE_PX
    const pts = validFrames.map(f => ({
      t: f.t_sec,
      v: useCm ? (f.speed_cm_s ?? 0) : (f.speed_px_s ?? 0),
    }))
    return { points: pts, maxSpeed: Math.max(maxS, immThr * 1.5), immThr, freezeThr }
  }, [frames, useCm])

  const tx = (t: number) => PAD.left + (t / Math.max(duration, 0.001)) * chartW
  const ty = (v: number) => PAD.top + chartH - (v / maxSpeed) * chartH

  const polyline = points.map(p => `${tx(p.t).toFixed(1)},${ty(p.v).toFixed(1)}`).join(' ')

  const cursorX = tx(currentTime)

  function handleClick(e: React.MouseEvent<SVGSVGElement>) {
    if (!onScrub) return
    const rect = e.currentTarget.getBoundingClientRect()
    const frac = (e.clientX - rect.left - PAD.left) / chartW
    onScrub(Math.max(0, Math.min(duration, frac * duration)))
  }

  const immY = ty(immThr)
  const freezeY = ty(freezeThr)
  const unit = useCm ? 'cm/s' : 'px/s'

  return (
    <div className="speed-chart-wrap">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        className="speed-svg"
        onClick={handleClick}
      >
        {/* Background */}
        <rect x={PAD.left} y={PAD.top} width={chartW} height={chartH} fill="rgba(255,255,255,0.03)" rx="3" />

        {/* Freezing band */}
        <rect
          x={PAD.left} y={freezeY}
          width={chartW} height={Math.max(0, immY - freezeY)}
          fill="rgba(90,140,255,0.12)"
        />
        {/* Immobile band */}
        <rect
          x={PAD.left} y={immY}
          width={chartW} height={Math.max(0, PAD.top + chartH - immY)}
          fill="rgba(255,255,255,0.06)"
        />

        {/* Threshold lines */}
        <line x1={PAD.left} x2={PAD.left + chartW} y1={immY} y2={immY} stroke="rgba(255,200,100,0.5)" strokeWidth="1" strokeDasharray="4 3" />
        <line x1={PAD.left} x2={PAD.left + chartW} y1={freezeY} y2={freezeY} stroke="rgba(90,140,255,0.6)" strokeWidth="1" strokeDasharray="4 3" />

        {/* Speed line */}
        {points.length > 1 && (
          <polyline
            points={polyline}
            fill="none"
            stroke="rgba(0,255,180,0.85)"
            strokeWidth="1.5"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )}

        {/* Y-axis labels */}
        {[0, immThr, maxSpeed].map(v => (
          <text key={v} x={PAD.left - 4} y={ty(v) + 4} textAnchor="end" fontSize="9" fill="rgba(255,255,255,0.5)">
            {v.toFixed(0)}
          </text>
        ))}
        {/* Unit label */}
        <text x={PAD.left - 2} y={PAD.top + 4} textAnchor="end" fontSize="8" fill="rgba(255,255,255,0.4)">{unit}</text>

        {/* X axis ticks */}
        {[0, 0.25, 0.5, 0.75, 1.0].map(frac => {
          const t = frac * duration
          const x = tx(t)
          return (
            <g key={frac}>
              <line x1={x} x2={x} y1={PAD.top + chartH} y2={PAD.top + chartH + 3} stroke="rgba(255,255,255,0.25)" strokeWidth="1" />
              <text x={x} y={H - 3} textAnchor="middle" fontSize="9" fill="rgba(255,255,255,0.4)">
                {t.toFixed(0)}s
              </text>
            </g>
          )
        })}

        {/* Threshold labels */}
        <text x={PAD.left + chartW - 2} y={immY - 2} textAnchor="end" fontSize="8" fill="rgba(255,200,100,0.7)">immobile</text>
        <text x={PAD.left + chartW - 2} y={freezeY - 2} textAnchor="end" fontSize="8" fill="rgba(90,140,255,0.8)">freeze</text>

        {/* Current time cursor */}
        {currentTime > 0 && (
          <line x1={cursorX} x2={cursorX} y1={PAD.top} y2={PAD.top + chartH} stroke="rgba(255,255,255,0.7)" strokeWidth="1.5" />
        )}
      </svg>
    </div>
  )
}
