import { useEffect, useRef, useState } from 'react'

interface Frame {
  ok: boolean
  centroid?: { x: number; y: number } | null
  speed_px_s?: number | null
  speed_cm_s?: number | null
}

interface Props {
  frames: Frame[]
  arenaW: number
  arenaH: number
  /** URL of first-frame JPEG to render underneath */
  backgroundUrl?: string
  /** 'occupancy' = time spent (default) | 'velocity' = speed-coded */
  mode?: 'occupancy' | 'velocity'
  /** called after render with the real max speed in the dataset */
  onMaxSpeed?: (maxSpeed: number) => void
}

// Plasma-like colormap: maps 0→1 to an RGB tuple
function plasmaColor(t: number): [number, number, number] {
  const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v)))
  const r = clamp(255 * Math.min(1, t * 2.5))
  const g = clamp(255 * Math.max(0, t * 2 - 0.4) * Math.min(1, (1 - t) * 3))
  const b = clamp(255 * Math.max(0, 1 - t * 3) + 80 * Math.max(0, t - 0.6))
  return [r, g, b]
}

// Cool-warm diverging ramp for velocity (blue=slow → red=fast)
function velocityColor(t: number): [number, number, number] {
  const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v)))
  const r = clamp(255 * Math.max(0, 2 * t - 0.5))
  const g = clamp(255 * (t < 0.5 ? 2 * t : 2 * (1 - t)))
  const b = clamp(255 * Math.max(0, 1 - 2 * t))
  return [r, g, b]
}

export default function Heatmap({ frames, arenaW, arenaH, backgroundUrl, mode = 'occupancy', onMaxSpeed }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const bgRef = useRef<HTMLCanvasElement>(null)
  const [actualMaxSpeed, setActualMaxSpeed] = useState<number | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || arenaW <= 0 || arenaH <= 0) return
    const ctx = canvas.getContext('2d')!

    const gridW = 80
    const gridH = Math.round(80 * (arenaH / arenaW))
    const grid = new Float32Array(gridW * gridH)
    const speedSum = new Float32Array(gridW * gridH)
    const speedCount = new Float32Array(gridW * gridH)

    const hasCm = frames.some(f => f.speed_cm_s != null)

    for (const f of frames) {
      if (!f.ok || !f.centroid) continue
      const gx = Math.floor((f.centroid.x / arenaW) * gridW)
      const gy = Math.floor((f.centroid.y / arenaH) * gridH)
      if (gx < 0 || gx >= gridW || gy < 0 || gy >= gridH) continue
      grid[gy * gridW + gx] += 1
      const spd = hasCm ? (f.speed_cm_s ?? 0) : (f.speed_px_s ?? 0)
      speedSum[gy * gridW + gx] += spd
      speedCount[gy * gridW + gx] += 1
    }

    let maxVal = 0
    let maxSpeed = 0
    for (let i = 0; i < gridW * gridH; i++) {
      if (grid[i] > maxVal) maxVal = grid[i]
      if (speedCount[i] > 0) {
        const avg = speedSum[i] / speedCount[i]
        if (avg > maxSpeed) maxSpeed = avg
      }
    }

    if (maxVal === 0) return

    // Expose actual max speed to parent for legend labels
    setActualMaxSpeed(maxSpeed)
    if (onMaxSpeed) onMaxSpeed(maxSpeed)

    canvas.width = gridW
    canvas.height = gridH
    const imgData = ctx.createImageData(gridW, gridH)

    for (let i = 0; i < gridW * gridH; i++) {
      if (grid[i] === 0) {
        imgData.data[i * 4 + 3] = 0
        continue
      }
      let t: number
      let rgb: [number, number, number]
      if (mode === 'velocity') {
        const avgSpeed = speedCount[i] > 0 ? speedSum[i] / speedCount[i] : 0
        t = Math.min(1, avgSpeed / Math.max(maxSpeed, 0.001))
        rgb = velocityColor(t)
      } else {
        t = grid[i] / maxVal
        rgb = plasmaColor(t)
      }
      imgData.data[i * 4 + 0] = rgb[0]
      imgData.data[i * 4 + 1] = rgb[1]
      imgData.data[i * 4 + 2] = rgb[2]
      imgData.data[i * 4 + 3] = Math.round(180 * t + 75)
    }

    ctx.putImageData(imgData, 0, 0)
  }, [frames, arenaW, arenaH, mode])

  // Background image
  useEffect(() => {
    const bg = bgRef.current
    if (!bg || !backgroundUrl) return
    const img = new Image()
    img.onload = () => {
      const ratio = Math.min(bg.parentElement!.clientWidth / img.naturalWidth, 1)
      bg.width = Math.round(img.naturalWidth * ratio)
      bg.height = Math.round(img.naturalHeight * ratio)
      bg.getContext('2d')!.drawImage(img, 0, 0, bg.width, bg.height)
    }
    img.src = backgroundUrl
  }, [backgroundUrl])

  const hasCm = frames.some(f => f.speed_cm_s != null)
  const speedUnit = hasCm ? 'cm/s' : 'px/s'

  const gradientStyle = mode === 'velocity'
    ? 'linear-gradient(to bottom, #e84040, #f5a623, #6eb5ff)'
    : 'linear-gradient(to bottom, #f5d020, #ff6b35, #7b2d8b)'

  const topLabel = mode === 'velocity'
    ? (actualMaxSpeed != null ? `${actualMaxSpeed.toFixed(1)} ${speedUnit}` : 'Fast')
    : 'High'
  const botLabel = mode === 'velocity' ? `0 ${speedUnit}` : 'Low'

  return (
    <div className="heatmap-wrap">
      <canvas ref={bgRef} className="heatmap-bg" />
      <canvas ref={canvasRef} className="heatmap-overlay" />
      <div className="heatmap-legend">
        <div className="heatmap-gradient" style={{ background: gradientStyle }} />
        <div className="heatmap-legend-labels">
          <span>{topLabel}</span>
          <span>{botLabel}</span>
        </div>
      </div>
    </div>
  )
}
