import { useEffect, useRef, useState } from 'react'
import './ArenaEditor.css'

export type Pt = { x: number; y: number }

export type Zone = {
  id: string
  name: string
  color: string
  points: Pt[]
  closed: boolean
}

export type ArenaSetup = {
  arenaPoly: Pt[]
  zones: Zone[]
  pxPerCm: number
  imgW: number
  imgH: number
}

type Tool = 'arena' | 'zone' | 'scale'

const ZONE_COLORS = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#c77dff', '#ff9a3c', '#f78fb3', '#00b894']

// ---------------------------------------------------------------------------
// Apparatus presets
// ---------------------------------------------------------------------------
type PresetKey = 'custom' | 'open_field_sq' | 'open_field_circ' | 'epm' | 'ymaze' | 'circular' | 'light_dark' | 'novel_object' | 'mwm' | 'fear_cond'

function buildPreset(key: PresetKey, W: number, H: number): { arena: Pt[]; zones: Zone[] } {
  const cx = W / 2, cy = H / 2
  const margin = 0.05  // 5% inset from edges

  const circPoly = (rx: number, ry: number, n = 36): Pt[] =>
    Array.from({ length: n }, (_, i) => {
      const a = (i / n) * 2 * Math.PI
      return { x: cx + rx * Math.cos(a), y: cy + ry * Math.sin(a) }
    })

  const rectPoly = (x0: number, y0: number, x1: number, y1: number): Pt[] => [
    { x: x0, y: y0 }, { x: x1, y: y0 }, { x: x1, y: y1 }, { x: x0, y: y1 }
  ]

  if (key === 'open_field_sq') {
    const x0 = W * margin, y0 = H * margin
    const x1 = W * (1 - margin), y1 = H * (1 - margin)
    const bw = (x1 - x0), bh = (y1 - y0)
    const inset = 0.3  // center zone = inner 30%
    const cx0 = x0 + bw * (0.5 - inset / 2), cy0 = y0 + bh * (0.5 - inset / 2)
    const cx1 = cx0 + bw * inset, cy1 = cy0 + bh * inset
    return {
      arena: rectPoly(x0, y0, x1, y1),
      zones: [
        { id: 'center', name: 'Center', color: '#ffd93d', points: rectPoly(cx0, cy0, cx1, cy1), closed: true },
        { id: 'periphery', name: 'Periphery', color: '#4d96ff', points: rectPoly(x0, y0, x1, y1), closed: true },
      ]
    }
  }

  if (key === 'open_field_circ' || key === 'circular') {
    const rx = W * (0.5 - margin), ry = H * (0.5 - margin)
    const center = circPoly(rx * 0.35, ry * 0.35)
    return {
      arena: circPoly(rx, ry),
      zones: [
        { id: 'center', name: 'Center', color: '#ffd93d', points: center, closed: true },
        { id: 'periphery', name: 'Periphery', color: '#4d96ff', points: circPoly(rx, ry), closed: true },
      ]
    }
  }

  if (key === 'epm') {
    // Cross shape: horizontal open arms + vertical closed arms
    const armW = W * 0.22, armH = H * 0.22
    const armLen = Math.min(W, H) * 0.4
    const cross: Pt[] = [
      { x: cx - armW / 2, y: cy - armLen }, { x: cx + armW / 2, y: cy - armLen },
      { x: cx + armW / 2, y: cy - armH / 2 }, { x: cx + armLen, y: cy - armH / 2 },
      { x: cx + armLen, y: cy + armH / 2 }, { x: cx + armW / 2, y: cy + armH / 2 },
      { x: cx + armW / 2, y: cy + armLen }, { x: cx - armW / 2, y: cy + armLen },
      { x: cx - armW / 2, y: cy + armH / 2 }, { x: cx - armLen, y: cy + armH / 2 },
      { x: cx - armLen, y: cy - armH / 2 }, { x: cx - armW / 2, y: cy - armH / 2 },
    ]
    return {
      arena: cross,
      zones: [
        { id: 'open_n', name: 'Open arm N', color: '#ff6b6b', points: rectPoly(cx - armW / 2, cy - armLen, cx + armW / 2, cy - armH / 2), closed: true },
        { id: 'open_s', name: 'Open arm S', color: '#ff9a3c', points: rectPoly(cx - armW / 2, cy + armH / 2, cx + armW / 2, cy + armLen), closed: true },
        { id: 'closed_e', name: 'Closed arm E', color: '#4d96ff', points: rectPoly(cx + armW / 2, cy - armH / 2, cx + armLen, cy + armH / 2), closed: true },
        { id: 'closed_w', name: 'Closed arm W', color: '#6bcb77', points: rectPoly(cx - armLen, cy - armH / 2, cx - armW / 2, cy + armH / 2), closed: true },
        { id: 'epm_center', name: 'Center', color: '#c77dff', points: rectPoly(cx - armW / 2, cy - armH / 2, cx + armW / 2, cy + armH / 2), closed: true },
      ]
    }
  }

  if (key === 'ymaze') {
    // Y shape: 3 arms at 120° intervals
    const innerR = W * 0.12, outerR = Math.min(W, H) * 0.42
    const arms: Zone[] = ['A', 'B', 'C'].map((name, i) => {
      const angle = (i * 120 - 90) * (Math.PI / 180)
      const perp = angle + Math.PI / 2
      const w = W * 0.10
      const ox = cx + outerR * Math.cos(angle)
      const oy = cy + outerR * Math.sin(angle)
      const ix = cx + innerR * Math.cos(angle)
      const iy = cy + innerR * Math.sin(angle)
      return {
        id: `arm_${name.toLowerCase()}`,
        name: `Arm ${name}`,
        color: ZONE_COLORS[i],
        points: [
          { x: ix + w * Math.cos(perp), y: iy + w * Math.sin(perp) },
          { x: ox + w * Math.cos(perp), y: oy + w * Math.sin(perp) },
          { x: ox - w * Math.cos(perp), y: oy - w * Math.sin(perp) },
          { x: ix - w * Math.cos(perp), y: iy - w * Math.sin(perp) },
        ],
        closed: true,
      }
    })
    const arena = circPoly(Math.min(W, H) * 0.48, Math.min(W, H) * 0.48, 3)
      .map((_, i) => {
        const angle = (i * 120 - 90) * (Math.PI / 180)
        return {
          x: cx + Math.min(W, H) * 0.46 * Math.cos(angle),
          y: cy + Math.min(W, H) * 0.46 * Math.sin(angle)
        }
      })
    return { arena, zones: arms }
  }

  if (key === 'light_dark') {
    const x0 = W * 0.05, y0 = H * 0.05
    const x1 = W * 0.95, y1 = H * 0.95
    const mid = (x0 + x1) / 2
    return {
      arena: rectPoly(x0, y0, x1, y1),
      zones: [
        { id: 'light', name: 'Light', color: '#ffd93d', points: rectPoly(mid, y0, x1, y1), closed: true },
        { id: 'dark', name: 'Dark', color: '#1a1a2e', points: rectPoly(x0, y0, mid, y1), closed: true },
      ]
    }
  }

  if (key === 'novel_object') {
    const x0 = W * 0.05, y0 = H * 0.05
    const x1 = W * 0.95, y1 = H * 0.95
    const bw = x1 - x0, bh = y1 - y0
    const objR = Math.min(bw, bh) * 0.08
    // Two object zones + center exploration zone
    const obj1 = circPoly(objR, objR, 20).map(p => ({ x: p.x + x0 + bw * 0.25, y: p.y + y0 + bh * 0.5 }))
    const obj2 = circPoly(objR, objR, 20).map(p => ({ x: p.x + x0 + bw * 0.75, y: p.y + y0 + bh * 0.5 }))
    return {
      arena: rectPoly(x0, y0, x1, y1),
      zones: [
        { id: 'object_1', name: 'Object 1', color: '#ff6b6b', points: obj1, closed: true },
        { id: 'object_2', name: 'Object 2', color: '#4d96ff', points: obj2, closed: true },
        { id: 'center', name: 'Center', color: '#ffd93d', points: rectPoly(x0 + bw * 0.35, y0 + bh * 0.3, x0 + bw * 0.65, y0 + bh * 0.7), closed: true },
      ]
    }
  }

  if (key === 'mwm') {
    // Morris Water Maze: circular pool, 4 quadrants, platform zone
    const rx = W * (0.5 - margin) * 0.9, ry = H * (0.5 - margin) * 0.9
    const pool = circPoly(rx, ry)
    const platR = rx * 0.08
    const platX = cx + rx * 0.35, platY = cy - ry * 0.35
    const platPoly = circPoly(platR, platR, 16).map(p => ({ x: p.x + platX - cx, y: p.y + platY - cy }))
    return {
      arena: pool,
      zones: [
        { id: 'target_quadrant', name: 'Target Quadrant', color: '#ffd93d', points: [
          { x: cx, y: cy }, { x: cx + rx * 0.1, y: cy - ry * 0.1 }, { x: cx + rx, y: cy - ry * 0.1 }, { x: cx + rx * 0.9, y: cy - ry }, { x: cx + rx * 0.1, y: cy - ry }
        ], closed: true },
        { id: 'opposite_quadrant', name: 'Opposite Quadrant', color: '#4d96ff', points: [
          { x: cx, y: cy }, { x: cx - rx * 0.1, y: cy + ry * 0.1 }, { x: cx - rx, y: cy + ry * 0.1 }, { x: cx - rx * 0.9, y: cy + ry }, { x: cx - rx * 0.1, y: cy + ry }
        ], closed: true },
        { id: 'left_quadrant', name: 'Left Quadrant', color: '#6bcb77', points: [
          { x: cx, y: cy }, { x: cx - rx * 0.1, y: cy - ry * 0.1 }, { x: cx - rx * 0.9, y: cy - ry }, { x: cx - rx, y: cy + ry * 0.1 }, { x: cx - rx * 0.1, y: cy + ry * 0.1 }
        ], closed: true },
        { id: 'right_quadrant', name: 'Right Quadrant', color: '#c77dff', points: [
          { x: cx, y: cy }, { x: cx + rx * 0.1, y: cy + ry * 0.1 }, { x: cx + rx * 0.9, y: cy + ry }, { x: cx + rx, y: cy - ry * 0.1 }, { x: cx + rx * 0.1, y: cy - ry * 0.1 }
        ], closed: true },
        { id: 'platform', name: 'Platform', color: '#ff6b6b', points: platPoly, closed: true },
      ]
    }
  }

  if (key === 'fear_cond') {
    // Fear conditioning: baseline, CS, US/ITI zones as time strips
    const x0 = W * 0.05, y0 = H * 0.05
    const x1 = W * 0.95, y1 = H * 0.95
    const bw = x1 - x0
    const seg = bw / 4
    return {
      arena: rectPoly(x0, y0, x1, y1),
      zones: [
        { id: 'baseline', name: 'Baseline', color: '#6bcb77', points: rectPoly(x0, y0, x0 + seg, y1), closed: true },
        { id: 'cs_tone_1', name: 'CS/Tone 1', color: '#ffd93d', points: rectPoly(x0 + seg, y0, x0 + 2 * seg, y1), closed: true },
        { id: 'iti_1', name: 'ITI 1', color: '#4d96ff', points: rectPoly(x0 + 2 * seg, y0, x0 + 3 * seg, y1), closed: true },
        { id: 'cs_tone_2', name: 'CS/Tone 2', color: '#ff9a3c', points: rectPoly(x0 + 3 * seg, y0, x1, y1), closed: true },
      ]
    }
  }

  return { arena: [], zones: [] }
}

const PRESET_LABELS: Record<PresetKey, string> = {
  custom: 'Custom (draw)',
  open_field_sq: 'Open Field — Square',
  open_field_circ: 'Open Field — Circular',
  epm: 'Elevated Plus Maze',
  ymaze: 'Y-Maze',
  circular: 'Circular Arena',
  light_dark: 'Light/Dark Box',
  novel_object: 'Novel Object',
  mwm: 'Morris Water Maze',
  fear_cond: 'Fear Conditioning',
}

interface Props {
  frameUrl: string
  onConfirm: (setup: ArenaSetup) => void
  onCancel: () => void
}

export default function ArenaEditor({ frameUrl, onConfirm, onCancel }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const [tool, setTool] = useState<Tool>('arena')
  const [preset, setPreset] = useState<PresetKey>('custom')
  const [arenaPts, setArenaPts] = useState<Pt[]>([])
  const [arenaClosed, setArenaClosed] = useState(false)
  const [zones, setZones] = useState<Zone[]>([])
  const [activeZoneId, setActiveZoneId] = useState<string | null>(null)
  const [scalePts, setScalePts] = useState<Pt[]>([])
  const [scaleRealCm, setScaleRealCm] = useState<string>('30')
  const [scaleDone, setScaleDone] = useState(false)
  const [newZoneName, setNewZoneName] = useState('')
  const [cursor, setCursor] = useState<Pt | null>(null)
  const [scale, setScale] = useState({ x: 1, y: 1, natW: 0, natH: 0 })
  const [detecting, setDetecting] = useState(false)
  const [detectConfidence, setDetectConfidence] = useState<number | null>(null)
  const [imgError, setImgError] = useState(false)

  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      imgRef.current = img
      setImgError(false)
      setScale(s => ({ ...s, natW: img.naturalWidth, natH: img.naturalHeight }))
    }
    img.onerror = () => setImgError(true)
    img.src = frameUrl
  }, [frameUrl])

  async function autoDetectArena() {
    setDetecting(true)
    setDetectConfidence(null)
    try {
      const resp = await fetch(frameUrl)
      const blob = await resp.blob()
      const fd = new FormData()
      fd.append('image', blob, 'frame.jpg')
      const r = await fetch('/api/arena/detect', { method: 'POST', body: fd })
      if (!r.ok) throw new Error('Detection failed')
      const data = await r.json() as { polygon: number[][] | null; confidence: number; method: string }
      if (data.polygon && data.polygon.length >= 3) {
        const pts = data.polygon.map(([x, y]) => ({ x, y }))
        setArenaPts(pts)
        setArenaClosed(true)
        setPreset('custom')
        setDetectConfidence(data.confidence)
      } else {
        alert('Could not detect a clear arena boundary. Please draw it manually.')
      }
    } catch {
      alert('Auto-detect failed. Please draw the arena manually.')
    } finally {
      setDetecting(false)
    }
  }

  // Apply preset once image dimensions are known
  function applyPreset(key: PresetKey) {
    setPreset(key)
    if (key === 'custom') {
      setArenaPts([]); setArenaClosed(false); setZones([])
      return
    }
    const W = scale.natW || 640
    const H = scale.natH || 480
    const { arena, zones: zs } = buildPreset(key, W, H)
    setArenaPts(arena)
    setArenaClosed(arena.length >= 3)
    setZones(zs)
    setActiveZoneId(null)
    setTool('scale')
  }

  function toNat(p: Pt): Pt { return { x: p.x / scale.x, y: p.y / scale.y } }
  function fromNat(p: Pt): Pt { return { x: p.x * scale.x, y: p.y * scale.y } }

  function redraw() {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return
    const ctx = canvas.getContext('2d')!

    const parent = canvas.parentElement!
    const maxW = parent.clientWidth - 2
    const maxH = Math.min(window.innerHeight * 0.55, 520)
    const ratio = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight)
    canvas.width = Math.round(img.naturalWidth * ratio)
    canvas.height = Math.round(img.naturalHeight * ratio)
    setScale(s => s.x === ratio ? s : { x: ratio, y: ratio, natW: img.naturalWidth, natH: img.naturalHeight })

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    // Dim outside arena
    if (arenaClosed && arenaPts.length >= 3) {
      ctx.save()
      ctx.fillStyle = 'rgba(0,0,0,0.48)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.globalCompositeOperation = 'destination-out'
      ctx.beginPath()
      arenaPts.forEach((p, i) => {
        const dp = fromNat(p)
        i === 0 ? ctx.moveTo(dp.x, dp.y) : ctx.lineTo(dp.x, dp.y)
      })
      ctx.closePath(); ctx.fill()
      ctx.restore()
    }

    // Arena outline
    if (arenaPts.length > 0) {
      ctx.strokeStyle = 'rgba(255,255,255,0.9)'
      ctx.lineWidth = 2
      ctx.setLineDash([6, 4])
      ctx.beginPath()
      arenaPts.forEach((p, i) => {
        const dp = fromNat(p)
        i === 0 ? ctx.moveTo(dp.x, dp.y) : ctx.lineTo(dp.x, dp.y)
      })
      if (arenaClosed) ctx.closePath()
      ctx.stroke()
      ctx.setLineDash([])
      if (tool === 'arena') {
        arenaPts.forEach(p => {
          const dp = fromNat(p)
          ctx.fillStyle = '#fff'
          ctx.beginPath()
          ctx.arc(dp.x, dp.y, 4, 0, Math.PI * 2)
          ctx.fill()
        })
      }
    }

    // Zones
    zones.forEach(z => {
      if (z.points.length < 2) return
      ctx.strokeStyle = z.color
      ctx.lineWidth = z.id === activeZoneId ? 2.5 : 1.5
      ctx.beginPath()
      z.points.forEach((p, i) => {
        const dp = fromNat(p)
        i === 0 ? ctx.moveTo(dp.x, dp.y) : ctx.lineTo(dp.x, dp.y)
      })
      if (z.closed) {
        ctx.closePath()
        ctx.fillStyle = z.color + '2a'
        ctx.fill()
      }
      ctx.stroke()
      if (z.points.length >= 3) {
        const avgX = z.points.reduce((a, p) => a + p.x, 0) / z.points.length
        const avgY = z.points.reduce((a, p) => a + p.y, 0) / z.points.length
        const dc = fromNat({ x: avgX, y: avgY })
        ctx.font = 'bold 11px sans-serif'
        ctx.fillStyle = z.color
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(z.name, dc.x, dc.y)
      }
    })

    // Scale line
    if (scalePts.length >= 1) {
      ctx.strokeStyle = '#ffd93d'
      ctx.lineWidth = 2
      ctx.setLineDash([4, 3])
      ctx.beginPath()
      const dp0 = fromNat(scalePts[0])
      ctx.moveTo(dp0.x, dp0.y)
      if (scalePts.length >= 2) {
        const dp1 = fromNat(scalePts[1])
        ctx.lineTo(dp1.x, dp1.y)
        ctx.stroke()
        ctx.setLineDash([])
        const pxLen = Math.hypot(dp1.x - dp0.x, dp1.y - dp0.y) / ratio
        ctx.font = '11px sans-serif'
        ctx.fillStyle = '#ffd93d'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'bottom'
        ctx.fillText(`${pxLen.toFixed(0)} px = ${scaleRealCm} cm`, (dp0.x + dp1.x) / 2, (dp0.y + dp1.y) / 2 - 4)
      } else if (cursor) {
        const dc = fromNat(cursor)
        ctx.lineTo(dc.x, dc.y)
        ctx.stroke()
      }
      ctx.setLineDash([])
      scalePts.forEach(p => {
        const dp = fromNat(p)
        ctx.fillStyle = '#ffd93d'
        ctx.beginPath(); ctx.arc(dp.x, dp.y, 4, 0, Math.PI * 2); ctx.fill()
      })
    }

    // Cursor guide
    if (cursor && tool === 'arena' && !arenaClosed && arenaPts.length > 0) {
      const last = fromNat(arenaPts[arenaPts.length - 1])
      const dc = fromNat(cursor)
      ctx.strokeStyle = 'rgba(255,255,255,0.4)'
      ctx.lineWidth = 1
      ctx.setLineDash([4, 4])
      ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(dc.x, dc.y); ctx.stroke()
      ctx.setLineDash([])
    }
  }

  useEffect(() => { redraw() }, [arenaPts, arenaClosed, zones, activeZoneId, scalePts, cursor, tool, scale])

  function getCanvasPt(e: React.MouseEvent<HTMLCanvasElement>): Pt {
    const rect = canvasRef.current!.getBoundingClientRect()
    return toNat({ x: e.clientX - rect.left, y: e.clientY - rect.top })
  }

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const pt = getCanvasPt(e)
    if (tool === 'arena' && !arenaClosed) {
      if (arenaPts.length >= 3) {
        const first = fromNat(arenaPts[0])
        const cur = fromNat(pt)
        if (Math.hypot(cur.x - first.x, cur.y - first.y) < 14) { setArenaClosed(true); return }
      }
      setArenaPts(prev => [...prev, pt])
    } else if (tool === 'zone' && activeZoneId) {
      setZones(prev => prev.map(z =>
        z.id === activeZoneId ? { ...z, points: [...z.points, pt] } : z
      ))
    } else if (tool === 'scale' && !scaleDone) {
      if (scalePts.length === 0) setScalePts([pt])
      else if (scalePts.length === 1) { setScalePts([scalePts[0], pt]); setScaleDone(true) }
    }
  }

  function handleDblClick() {
    if (tool === 'arena' && arenaPts.length >= 3 && !arenaClosed) setArenaClosed(true)
    else if (tool === 'zone' && activeZoneId) {
      setZones(prev => prev.map(z => z.id === activeZoneId ? { ...z, closed: true } : z))
      setActiveZoneId(null)
    }
  }

  function addZone() {
    if (!newZoneName.trim()) return
    const id = `z_${Date.now()}`
    const color = ZONE_COLORS[zones.length % ZONE_COLORS.length]
    setZones(prev => [...prev, { id, name: newZoneName.trim(), color, points: [], closed: false }])
    setActiveZoneId(id); setTool('zone'); setNewZoneName('')
  }

  function removeZone(id: string) {
    setZones(prev => prev.filter(z => z.id !== id))
    if (activeZoneId === id) setActiveZoneId(null)
  }

  function closeActiveZone() {
    if (!activeZoneId) return
    setZones(prev => prev.map(z => z.id === activeZoneId ? { ...z, closed: true } : z))
    setActiveZoneId(null)
  }

  function handleConfirm() {
    let pxPerCm = 0
    if (scalePts.length === 2) {
      const d = Math.hypot(scalePts[1].x - scalePts[0].x, scalePts[1].y - scalePts[0].y)
      const cm = parseFloat(scaleRealCm)
      if (d > 0 && cm > 0) pxPerCm = d / cm
    }
    onConfirm({
      arenaPoly: arenaClosed ? arenaPts : [],
      zones: zones.filter(z => z.closed && z.points.length >= 3),
      pxPerCm,
      imgW: scale.natW,
      imgH: scale.natH,
    })
  }

  const activeZone = zones.find(z => z.id === activeZoneId)

  return (
    <div className="ae-wrap">
      {/* Preset selector */}
      <div className="ae-presets">
        <span className="ae-label" style={{ marginBottom: 0 }}>Apparatus template</span>
        <div className="ae-preset-grid">
          {(Object.keys(PRESET_LABELS) as PresetKey[]).map(k => (
            <button
              key={k}
              className={`ae-preset-btn ${preset === k ? 'active' : ''}`}
              onClick={() => applyPreset(k)}
            >
              {PRESET_LABELS[k]}
            </button>
          ))}
          <button
            className="ae-preset-btn"
            style={{ borderColor: 'rgba(0,217,200,0.5)', color: detecting ? '#888' : '#00d9c8' }}
            onClick={autoDetectArena}
            disabled={detecting}
          >
            {detecting ? '⟳ Detecting…' : '⚡ Auto-detect'}
          </button>
          {detectConfidence !== null && (
            <span style={{ fontSize: 11, color: detectConfidence > 0.5 ? '#22c55e' : '#f59e0b', alignSelf: 'center' }}>
              {detectConfidence > 0.5 ? '✓' : '⚠'} Confidence: {(detectConfidence * 100).toFixed(0)}%
            </span>
          )}
        </div>
      </div>

      {imgError && (
        <div style={{ padding: '12px 14px', fontSize: 12, color: '#f59e0b', background: 'rgba(245,158,11,0.1)', borderRadius: 8, border: '1px solid rgba(245,158,11,0.25)' }}>
          ⚠ Could not load video frame preview. Draw your arena manually using the tools below.
        </div>
      )}

      <div className="ae-toolbar">
        {/* Step 1: Arena */}
        <div className="ae-toolgroup">
          <span className="ae-label">1 · Arena boundary</span>
          <button className={`ae-btn ${tool === 'arena' ? 'active' : ''}`}
            onClick={() => setTool('arena')}>
            ✏ Draw
          </button>
          {arenaClosed && <button className="ae-btn danger-sm" onClick={() => { setArenaPts([]); setArenaClosed(false) }}>Reset</button>}
          <span className="ae-hint">
            {!arenaClosed
              ? preset !== 'custom' ? '✓ Auto-set (adjustable)' : 'Click vertices • double-click or click start to close'
              : '✓ Set'}
          </span>
        </div>

        {/* Step 2: Zones */}
        <div className="ae-toolgroup">
          <span className="ae-label">2 · Zones (optional)</span>
          <div className="ae-zone-row">
            <input className="ae-input" placeholder="Zone name…" value={newZoneName}
              onChange={e => setNewZoneName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && addZone()} />
            <button className="ae-btn" onClick={addZone} disabled={!newZoneName.trim()}>+ Add</button>
          </div>
          <div className="ae-zones-list">
            {zones.map(z => (
              <span key={z.id} className="ae-zone-pill" style={{ borderColor: z.color }}>
                <span className="ae-zone-dot" style={{ background: z.color }}
                  onClick={() => { setActiveZoneId(z.id); setTool('zone') }} />
                {z.name}
                {!z.closed && activeZoneId === z.id && (
                  <button className="ae-btn-xs" onClick={closeActiveZone}>Close</button>
                )}
                <button className="ae-btn-xs danger" onClick={() => removeZone(z.id)}>✕</button>
              </span>
            ))}
          </div>
          {activeZone && !activeZone.closed && (
            <span className="ae-hint">Drawing "{activeZone.name}" — click points, then Close</span>
          )}
        </div>

        {/* Step 3: Scale */}
        <div className="ae-toolgroup">
          <span className="ae-label">3 · Scale (optional)</span>
          <button className={`ae-btn ${tool === 'scale' ? 'active' : ''}`}
            onClick={() => { setTool('scale'); setScalePts([]); setScaleDone(false) }}>
            📏 Draw scale bar
          </button>
          {scaleDone && (
            <>
              <input className="ae-input narrow" type="number" min={0.1} step={0.5}
                value={scaleRealCm} onChange={e => setScaleRealCm(e.target.value)} />
              <span className="ae-hint">cm</span>
              <button className="ae-btn danger-sm" onClick={() => { setScalePts([]); setScaleDone(false) }}>Reset</button>
            </>
          )}
          <span className="ae-hint">
            {scaleDone ? '✓ Scale set' : !scaleDone && tool === 'scale' ? 'Click 2 points on a ruler/object' : ''}
          </span>
        </div>
      </div>

      <canvas ref={canvasRef} className="ae-canvas"
        onClick={handleClick} onDoubleClick={handleDblClick}
        onMouseMove={e => setCursor(getCanvasPt(e))}
        onMouseLeave={() => setCursor(null)} />

      <div className="ae-actions">
        <button className="ae-btn-action secondary" onClick={onCancel}>Cancel</button>
        <button className="ae-btn-action primary" onClick={handleConfirm}>
          {arenaClosed ? '▶ Start tracking' : '▶ Track (full frame)'}
        </button>
      </div>
    </div>
  )
}
