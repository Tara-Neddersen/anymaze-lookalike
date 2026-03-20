/**
 * CalibrationTool — overlay two-point ruler on a video frame.
 * User clicks point A, then point B on an object of known length.
 * Entering the real-world distance in cm auto-computes px/cm.
 */
import { useEffect, useRef, useState } from 'react'

interface Props {
  frameUrl: string       // JPEG/PNG of the first video frame
  onCalibrate: (pxPerCm: number) => void
  onClose: () => void
}

type Pt = { x: number; y: number }

export default function CalibrationTool({ frameUrl, onCalibrate, onClose }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const [pts, setPts] = useState<Pt[]>([])
  const [realCm, setRealCm] = useState('')

  // Load image and draw to canvas
  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      imgRef.current = img
      const canvas = canvasRef.current
      if (!canvas) return
      const maxW = Math.min(img.naturalWidth, window.innerWidth - 120)
      const scale = maxW / img.naturalWidth
      canvas.width = Math.round(img.naturalWidth * scale)
      canvas.height = Math.round(img.naturalHeight * scale)
      redraw(canvas, img, [])
    }
    img.src = frameUrl
  }, [frameUrl])

  function redraw(canvas: HTMLCanvasElement, img: HTMLImageElement, points: Pt[]) {
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    // Dim overlay
    ctx.fillStyle = 'rgba(0,0,0,0.35)'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    if (points.length === 0) return

    const [a, b] = points

    if (points.length >= 1) {
      // Point A
      ctx.beginPath(); ctx.arc(a.x, a.y, 6, 0, Math.PI * 2)
      ctx.fillStyle = '#00d9c8'; ctx.fill()
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke()
      ctx.fillStyle = '#00d9c8'; ctx.font = 'bold 13px DM Sans, sans-serif'
      ctx.fillText('A', a.x + 10, a.y - 8)
    }

    if (points.length >= 2) {
      // Point B
      ctx.beginPath(); ctx.arc(b.x, b.y, 6, 0, Math.PI * 2)
      ctx.fillStyle = '#ffd93d'; ctx.fill()
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke()
      ctx.fillStyle = '#ffd93d'
      ctx.fillText('B', b.x + 10, b.y - 8)

      // Ruler line
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y)
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2
      ctx.setLineDash([6, 3]); ctx.stroke(); ctx.setLineDash([])

      // Tick marks at ends
      const angle = Math.atan2(b.y - a.y, b.x - a.x) + Math.PI / 2
      for (const pt of [a, b]) {
        ctx.beginPath()
        ctx.moveTo(pt.x + 8 * Math.cos(angle), pt.y + 8 * Math.sin(angle))
        ctx.lineTo(pt.x - 8 * Math.cos(angle), pt.y - 8 * Math.sin(angle))
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke()
      }

      // Distance label in pixels
      const dx = b.x - a.x, dy = b.y - a.y
      const dist = Math.round(Math.hypot(dx, dy))
      const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2
      ctx.fillStyle = 'rgba(0,0,0,0.65)'
      ctx.fillRect(mx - 38, my - 24, 76, 22)
      ctx.fillStyle = '#fff'; ctx.font = '12px Space Mono, monospace'
      ctx.textAlign = 'center'
      ctx.fillText(`${dist} px`, mx, my - 8)
      ctx.textAlign = 'left'
    }
  }

  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return
    if (pts.length >= 2) return  // already have both points

    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left) * (canvas.width / rect.width)
    const y = (e.clientY - rect.top) * (canvas.height / rect.height)
    const newPts = [...pts, { x, y }]
    setPts(newPts)
    redraw(canvas, img, newPts)
  }

  function reset() {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (canvas && img) redraw(canvas, img, [])
    setPts([])
    setRealCm('')
  }

  const pixelDist = pts.length === 2
    ? Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y)
    : 0

  // Scale px dist back to original video pixels
  const scaleFactor = imgRef.current
    ? imgRef.current.naturalWidth / (canvasRef.current?.width || 1)
    : 1
  const realPixelDist = pixelDist * scaleFactor

  const pxPerCm = realCm && realPixelDist > 0
    ? realPixelDist / parseFloat(realCm)
    : null

  function confirm() {
    if (pxPerCm && pxPerCm > 0) {
      onCalibrate(Math.round(pxPerCm * 100) / 100)
    }
  }

  const canConfirm = pxPerCm !== null && pxPerCm > 0 && !isNaN(pxPerCm)

  return (
    <div className="calModal">
      <div className="calOverlay" onClick={onClose} />
      <div className="calBox">
        <div className="calHeader">
          <div className="calTitle">Calibration ruler</div>
          <button className="calClose" onClick={onClose}>✕</button>
        </div>

        <div className="calInstructions">
          {pts.length === 0 && 'Click point A on a known-length object (e.g. scale bar, maze wall)'}
          {pts.length === 1 && 'Now click point B at the other end'}
          {pts.length === 2 && 'Enter the real-world distance between A and B, then confirm'}
        </div>

        <div className="calCanvasWrap" style={{ maxHeight: '55vh', overflow: 'auto' }}>
          <canvas
            ref={canvasRef}
            className="calCanvas"
            style={{ cursor: pts.length < 2 ? 'crosshair' : 'default', maxWidth: '100%' }}
            onClick={handleCanvasClick}
          />
        </div>

        {pts.length === 2 && (
          <div className="calControls">
            <div className="calDistRow">
              <span className="calDistLabel">Pixel distance:</span>
              <span className="calDistVal">{Math.round(realPixelDist)} px</span>
            </div>
            <div className="calDistRow">
              <label className="calDistLabel" htmlFor="realDist">Real distance (cm):</label>
              <input
                id="realDist"
                type="number"
                className="calInput"
                min={0.1}
                step={0.1}
                placeholder="e.g. 10"
                value={realCm}
                onChange={e => setRealCm(e.target.value)}
                autoFocus
              />
            </div>
            {pxPerCm != null && !isNaN(pxPerCm) && pxPerCm > 0 && (
              <div className="calResult">
                → <strong>{pxPerCm.toFixed(2)}</strong> px/cm
              </div>
            )}
          </div>
        )}

        <div className="calActions">
          <button className="btnSecondary" onClick={reset}>↺ Reset points</button>
          <button className="btnSecondary" onClick={onClose}>Cancel</button>
          <button className="btnPrimary" disabled={!canConfirm} onClick={confirm}>
            ✓ Apply calibration
          </button>
        </div>
      </div>
    </div>
  )
}
