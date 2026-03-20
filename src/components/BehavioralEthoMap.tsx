import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react'

interface FrameMeta {
  job_id: string
  genotype: string
  animal_id: string
  frame_idx: number
}

interface EmbeddingData {
  coords: [number, number][]
  frame_meta: FrameMeta[]
  n_valid: number
  params: Record<string, any>
}

const GENOTYPE_COLORS: Record<string, string> = {
  WT:   '#4ade80',
  BPAN: '#f87171',
  KO:   '#fb923c',
  HET:  '#a78bfa',
}

function gColor(g: string): string {
  return GENOTYPE_COLORS[g?.toUpperCase?.()] ?? '#60a5fa'
}

interface Props {
  cohortId: string | null
}

export default function BehavioralEthoMap({ cohortId }: Props) {
  const canvasRef   = useRef<HTMLCanvasElement>(null)
  const [embedding, setEmbedding]   = useState<EmbeddingData | null>(null)
  const [loading,   setLoading]     = useState(false)
  const [error,     setError]       = useState<string | null>(null)
  const [hovered,   setHovered]     = useState<{ meta: FrameMeta; x: number; y: number } | null>(null)
  const [showDensity, setShowDensity] = useState(true)
  const [selectedGenotypes, setSelectedGenotypes] = useState<Set<string>>(new Set())

  // Map from canvas coords to data coords
  const transformRef = useRef<{ scaleX: number; scaleY: number; offX: number; offY: number } | null>(null)

  useEffect(() => {
    if (!cohortId) return
    setLoading(true)
    setError(null)
    fetch(`/api/cohorts/${cohortId}/results/embedding`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(d => {
        setEmbedding(d)
        const genotypes = new Set<string>(d.frame_meta.map((m: FrameMeta) => m.genotype))
        setSelectedGenotypes(genotypes)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [cohortId])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !embedding) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const W = canvas.width
    const H = canvas.height
    const pad = 32

    ctx.clearRect(0, 0, W, H)

    const coords = embedding.coords
    if (!coords.length) return

    const xs = coords.map(c => c[0])
    const ys = coords.map(c => c[1])
    const xMin = Math.min(...xs), xMax = Math.max(...xs)
    const yMin = Math.min(...ys), yMax = Math.max(...ys)
    const xRange = xMax - xMin || 1
    const yRange = yMax - yMin || 1

    const scaleX = (W - pad * 2) / xRange
    const scaleY = (H - pad * 2) / yRange
    transformRef.current = { scaleX, scaleY, offX: -xMin * scaleX + pad, offY: -yMin * scaleY + pad }

    const toCanvas = (x: number, y: number): [number, number] => [
      x * scaleX + transformRef.current!.offX,
      y * scaleY + transformRef.current!.offY,
    ]

    // Draw background
    ctx.fillStyle = 'rgba(15,15,25,0.95)'
    ctx.fillRect(0, 0, W, H)

    // Draw density heatmap per genotype (if enabled)
    if (showDensity) {
      const genotypes = Array.from(new Set(embedding.frame_meta.map(m => m.genotype)))
      for (const gt of genotypes) {
        if (!selectedGenotypes.has(gt)) continue
        const col = gColor(gt)
        const pts = coords.filter((_, i) => embedding.frame_meta[i].genotype === gt)
        if (pts.length < 5) continue

        // Gaussian density approximation on grid
        const gridN = 40
        const grid = new Float32Array(gridN * gridN)
        const bw = Math.max(xRange, yRange) * 0.04  // bandwidth

        for (const [px, py] of pts) {
          const gx = Math.floor(((px - xMin) / xRange) * (gridN - 1))
          const gy = Math.floor(((py - yMin) / yRange) * (gridN - 1))
          for (let di = -2; di <= 2; di++) {
            for (let dj = -2; dj <= 2; dj++) {
              const ni = gx + di, nj = gy + dj
              if (ni < 0 || ni >= gridN || nj < 0 || nj >= gridN) continue
              const d2 = (di * xRange / gridN) ** 2 + (dj * yRange / gridN) ** 2
              grid[nj * gridN + ni] += Math.exp(-0.5 * d2 / bw ** 2)
            }
          }
        }

        const maxVal = Math.max(...Array.from(grid))
        if (maxVal === 0) continue

        const r = parseInt(col.slice(1, 3), 16)
        const g = parseInt(col.slice(3, 5), 16)
        const b = parseInt(col.slice(5, 7), 16)

        for (let j = 0; j < gridN; j++) {
          for (let i = 0; i < gridN; i++) {
            const v = grid[j * gridN + i] / maxVal
            if (v < 0.05) continue
            const [cx, cy] = toCanvas(
              xMin + (i / (gridN - 1)) * xRange,
              yMin + (j / (gridN - 1)) * yRange,
            )
            const cellW = W / gridN * 1.5
            ctx.fillStyle = `rgba(${r},${g},${b},${v * 0.25})`
            ctx.fillRect(cx - cellW / 2, cy - cellW / 2, cellW, cellW)
          }
        }
      }
    }

    // Draw dots
    const DOT_R = coords.length > 10000 ? 1 : coords.length > 3000 ? 1.5 : 2.5
    for (let i = 0; i < coords.length; i++) {
      const meta = embedding.frame_meta[i]
      if (!selectedGenotypes.has(meta.genotype)) continue
      const [cx, cy] = toCanvas(coords[i][0], coords[i][1])
      ctx.beginPath()
      ctx.arc(cx, cy, DOT_R, 0, Math.PI * 2)
      ctx.fillStyle = gColor(meta.genotype) + 'bb'
      ctx.fill()
    }

    // Axes labels
    ctx.fillStyle = '#475569'
    ctx.font = '11px monospace'
    ctx.fillText('UMAP 1', W / 2 - 20, H - 6)
    ctx.save()
    ctx.translate(12, H / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('UMAP 2', -20, 0)
    ctx.restore()

    // Legend
    const genotypes = Array.from(selectedGenotypes)
    let lx = W - 120, ly = 12
    for (const gt of genotypes) {
      ctx.fillStyle = gColor(gt) + 'cc'
      ctx.fillRect(lx, ly, 10, 10)
      ctx.fillStyle = '#e2e8f0'
      ctx.font = '11px sans-serif'
      ctx.fillText(gt, lx + 14, ly + 9)
      ly += 18
    }
  }, [embedding, showDensity, selectedGenotypes])

  useEffect(() => { draw() }, [draw])

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!embedding || !transformRef.current || !canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const mx = (e.clientX - rect.left) * (canvasRef.current.width / rect.width)
    const my = (e.clientY - rect.top) * (canvasRef.current.height / rect.height)
    const { scaleX, scaleY, offX, offY } = transformRef.current

    // Find nearest point
    let minD = Infinity, nearestIdx = -1
    for (let i = 0; i < embedding.coords.length; i++) {
      const [px, py] = embedding.coords[i]
      const cx = px * scaleX + offX
      const cy = py * scaleY + offY
      const d = (cx - mx) ** 2 + (cy - my) ** 2
      if (d < minD) { minD = d; nearestIdx = i }
    }

    if (nearestIdx >= 0 && minD < 400) {
      setHovered({ meta: embedding.frame_meta[nearestIdx], x: e.clientX, y: e.clientY })
    } else {
      setHovered(null)
    }
  }, [embedding])

  const genotypesInData = useMemo(() => {
    if (!embedding) return []
    return Array.from(new Set(embedding.frame_meta.map(m => m.genotype)))
  }, [embedding])

  const toggleGenotype = (g: string) => {
    setSelectedGenotypes(prev => {
      const next = new Set(prev)
      if (next.has(g)) next.delete(g); else next.add(g)
      return next
    })
  }

  if (!cohortId) {
    return (
      <div style={{ color: '#475569', textAlign: 'center', padding: 40, fontSize: 14 }}>
        Select a cohort and run the embedding step to see the behavioral landscape.
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '100%' }}>
      {/* Toolbar */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <span style={{ fontWeight: 700, color: '#e2e8f0', fontSize: 14 }}>
          Behavioral Landscape (UMAP)
        </span>
        {embedding && (
          <span style={{ fontSize: 11, color: '#64748b' }}>
            {embedding.n_valid.toLocaleString()} frames · {genotypesInData.length} genotypes
          </span>
        )}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          {genotypesInData.map(g => (
            <button key={g}
              onClick={() => toggleGenotype(g)}
              style={{
                background: selectedGenotypes.has(g) ? gColor(g) + '22' : 'rgba(255,255,255,0.05)',
                border: `1px solid ${selectedGenotypes.has(g) ? gColor(g) + '55' : 'rgba(255,255,255,0.1)'}`,
                borderRadius: 5, color: gColor(g), padding: '3px 10px',
                fontSize: 11, cursor: 'pointer', fontWeight: 600,
              }}>
              {g}
            </button>
          ))}
          <button onClick={() => setShowDensity(v => !v)}
            style={{
              background: showDensity ? 'rgba(99,102,241,0.2)' : 'rgba(255,255,255,0.05)',
              border: `1px solid ${showDensity ? '#818cf8' : 'rgba(255,255,255,0.1)'}`,
              borderRadius: 5, color: showDensity ? '#818cf8' : '#64748b',
              padding: '3px 10px', fontSize: 11, cursor: 'pointer',
            }}>
            {showDensity ? '◉ Density' : '○ Density'}
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div style={{ flex: 1, position: 'relative', minHeight: 0 }}>
        {loading && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
            justifyContent: 'center', background: 'rgba(0,0,0,0.5)', borderRadius: 12,
            color: '#94a3b8', fontSize: 14, zIndex: 10,
          }}>
            Computing embedding…
          </div>
        )}
        {error && (
          <div style={{
            padding: 20, color: '#f87171', fontSize: 13,
            background: 'rgba(239,68,68,0.1)', borderRadius: 8,
          }}>
            {error}
            <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 4 }}>
              Run the Analysis Pipeline steps first (Pose Features + Embedding).
            </div>
          </div>
        )}
        {!loading && !error && !embedding && (
          <div style={{ color: '#475569', textAlign: 'center', paddingTop: 60, fontSize: 13 }}>
            No embedding data. Run "Compute Embedding" in the pipeline.
          </div>
        )}
        <canvas
          ref={canvasRef}
          width={800} height={480}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHovered(null)}
          style={{
            width: '100%', height: '100%', borderRadius: 12,
            cursor: 'crosshair', display: embedding ? 'block' : 'none',
          }}
        />

        {/* Tooltip */}
        {hovered && (
          <div style={{
            position: 'fixed', left: hovered.x + 12, top: hovered.y - 10,
            background: 'rgba(15,15,25,0.95)', border: '1px solid rgba(255,255,255,0.15)',
            borderRadius: 8, padding: '8px 12px', pointerEvents: 'none',
            zIndex: 1000, fontSize: 11,
          }}>
            <div style={{ color: gColor(hovered.meta.genotype), fontWeight: 700 }}>
              {hovered.meta.genotype}
            </div>
            <div style={{ color: '#94a3b8' }}>Animal: {hovered.meta.animal_id}</div>
            <div style={{ color: '#94a3b8' }}>Frame: {hovered.meta.frame_idx}</div>
          </div>
        )}
      </div>

      {/* Stats bar */}
      {embedding && (
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          {genotypesInData.map(g => {
            const n = embedding.frame_meta.filter(m => m.genotype === g).length
            const pct = (100 * n / embedding.n_valid).toFixed(1)
            return (
              <div key={g} style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 12 }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: gColor(g), display: 'inline-block' }} />
                <span style={{ color: '#94a3b8' }}>{g}: </span>
                <span style={{ color: '#e2e8f0' }}>{n.toLocaleString()} frames ({pct}%)</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
