import { useEffect, useRef, useState } from 'react'

interface MotifLibrary {
  k: number
  auto_labels: string[]
  centroids: number[][]
  stability_score: number
  silhouette: number
  k_sweep_results?: {
    k_values: number[]
    silhouettes: number[]
    inertias: number[]
  }
}

interface MotifUsage {
  genotype: string
  usage: number[]  // per-motif usage fraction
}

interface Props {
  cohortId: string | null
}

// Canonical KP colors for stick-figure rendering
const KP_COLORS = ['#FF4C4C', '#FFD93D', '#FFD93D', '#6BCB77', '#4D96FF', '#C77DFF']
const SKELETON_PAIRS: [number, number][] = [
  [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 4], [4, 5],
]

const MOTIF_COLORS = [
  '#60a5fa','#f472b6','#34d399','#fbbf24','#a78bfa',
  '#fb923c','#22d3ee','#e879f9','#4ade80','#f87171',
]

export default function MotifGallery({ cohortId }: Props) {
  const [library,  setLibrary]  = useState<MotifLibrary | null>(null)
  const [usages,   setUsages]   = useState<MotifUsage[]>([])
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState<string | null>(null)
  const [selected, setSelected] = useState<number | null>(null)

  useEffect(() => {
    if (!cohortId) return
    setLoading(true)
    setError(null)

    Promise.all([
      fetch(`/api/cohorts/${cohortId}/motif_library`).then(r => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)),
      fetch(`/api/cohorts/${cohortId}/results/phenotypes`).then(r => r.ok ? r.json() : null).catch(() => null),
    ]).then(([lib, ph]) => {
      setLibrary(lib)

      if (ph?.raw_vectors && ph?.animals) {
        const k = lib.k
        const groups: Record<string, number[][]> = {}
        for (let i = 0; i < ph.animals.length; i++) {
          const g = ph.animals[i].genotype
          if (!groups[g]) groups[g] = []
          const usagePct = Array.from({ length: k }, (_, mi) =>
            ph.raw_vectors[i][`motif_${mi}_usage_pct`] ?? 0.0
          )
          groups[g].push(usagePct)
        }

        const usageList: MotifUsage[] = Object.entries(groups).map(([genotype, arrs]) => ({
          genotype,
          usage: Array.from({ length: k }, (_, mi) =>
            arrs.reduce((s, a) => s + (a[mi] ?? 0), 0) / arrs.length
          ),
        }))
        setUsages(usageList)
      }
    }).catch(e => setError(String(e)))
    .finally(() => setLoading(false))
  }, [cohortId])

  if (!cohortId) {
    return <div style={{ color: '#475569', textAlign: 'center', padding: 40, fontSize: 14 }}>
      Select a cohort and run motif discovery to see the gallery.
    </div>
  }

  if (loading) return <div style={{ color: '#94a3b8', padding: 40, textAlign: 'center' }}>Loading motif library…</div>
  if (error) return (
    <div style={{ padding: 20, color: '#f87171', fontSize: 13 }}>
      {error} — Run "Discover Motifs" in the pipeline first.
    </div>
  )
  if (!library) return (
    <div style={{ color: '#475569', padding: 40, textAlign: 'center', fontSize: 13 }}>
      No motif library. Run motif discovery in the pipeline.
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header */}
      <div style={{ display: 'flex', gap: 20, alignItems: 'center' }}>
        <span style={{ fontWeight: 700, color: '#e2e8f0', fontSize: 14 }}>
          Behavioral Motifs (k={library.k})
        </span>
        <span style={{ fontSize: 11, color: '#64748b' }}>
          Stability: {library.stability_score.toFixed(2)} · Silhouette: {library.silhouette.toFixed(3)}
        </span>
        {library.stability_score < 0.75 && (
          <span style={{ fontSize: 11, color: '#fb923c' }}>
            ⚠ Low stability — consider reducing k
          </span>
        )}
      </div>

      {/* Silhouette sweep mini-chart */}
      {library.k_sweep_results && (
        <SilhouetteChart sweep={library.k_sweep_results} chosenK={library.k} />
      )}

      {/* Motif grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
        gap: 12,
      }}>
        {Array.from({ length: library.k }, (_, mi) => (
          <MotifCard
            key={mi}
            motifIdx={mi}
            label={library.auto_labels[mi] ?? `Motif ${mi}`}
            centroid={library.centroids[mi] ?? []}
            usages={usages}
            color={MOTIF_COLORS[mi % MOTIF_COLORS.length]}
            selected={selected === mi}
            onSelect={() => setSelected(selected === mi ? null : mi)}
          />
        ))}
      </div>

      {/* Detail panel for selected motif */}
      {selected !== null && library.centroids[selected] && (
        <MotifDetail
          motifIdx={selected}
          label={library.auto_labels[selected]}
          centroid={library.centroids[selected]}
          color={MOTIF_COLORS[selected % MOTIF_COLORS.length]}
          usages={usages}
        />
      )}
    </div>
  )
}


function MotifCard({
  motifIdx, label, centroid, usages, color, selected, onSelect,
}: {
  motifIdx: number, label: string, centroid: number[], usages: MotifUsage[],
  color: string, selected: boolean, onSelect: () => void,
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Draw average pose stick-figure from centroid (first 12 dims = 6 kp positions)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || centroid.length < 12) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const W = canvas.width, H = canvas.height
    ctx.clearRect(0, 0, W, H)

    // KP positions from normalized vector (add mid_spine at origin)
    // centroid: [nose_x,nose_y, l_ear_x,l_ear_y, r_ear_x,r_ear_y,
    //            neck_x,neck_y, hips_x,hips_y, tail_x,tail_y]
    const kps: [number, number][] = []
    for (let i = 0; i < 6; i++) {
      const nx = centroid[i * 2]     // normalized x
      const ny = centroid[i * 2 + 1] // normalized y
      kps.push([nx, ny])
    }
    // Also add mid_spine = (0, 0) as reference
    kps.push([0, 0])  // index 6

    // Find bounds
    const allX = kps.map(k => k[0])
    const allY = kps.map(k => k[1])
    const xMin = Math.min(...allX), xMax = Math.max(...allX)
    const yMin = Math.min(...allY), yMax = Math.max(...allY)
    const span = Math.max(xMax - xMin, yMax - yMin, 0.5)
    const pad  = 20

    const toC = (x: number, y: number): [number, number] => [
      ((x - xMin) / span) * (W - pad * 2) + pad,
      ((y - yMin) / span) * (H - pad * 2) + pad,
    ]

    // Draw bones
    ctx.lineWidth = 1.5
    ctx.strokeStyle = 'rgba(255,255,255,0.3)'
    for (const [a, b] of SKELETON_PAIRS) {
      if (a >= kps.length || b >= kps.length) continue
      const [ax, ay] = toC(kps[a][0], kps[a][1])
      const [bx, by] = toC(kps[b][0], kps[b][1])
      ctx.beginPath()
      ctx.moveTo(ax, ay)
      ctx.lineTo(bx, by)
      ctx.stroke()
    }

    // Draw keypoints
    for (let i = 0; i < 6; i++) {
      const [cx, cy] = toC(kps[i][0], kps[i][1])
      ctx.beginPath()
      ctx.arc(cx, cy, 4, 0, Math.PI * 2)
      ctx.fillStyle = KP_COLORS[i]
      ctx.fill()
    }

    // Mid-spine marker
    const [mx, my] = toC(0, 0)
    ctx.beginPath()
    ctx.arc(mx, my, 3, 0, Math.PI * 2)
    ctx.fillStyle = '#00DDB4'
    ctx.fill()
  }, [centroid])

  return (
    <div
      onClick={onSelect}
      style={{
        background: selected ? `${color}15` : 'rgba(255,255,255,0.04)',
        border: `1px solid ${selected ? color + '55' : 'rgba(255,255,255,0.08)'}`,
        borderRadius: 10, padding: 12, cursor: 'pointer',
        transition: 'all 0.15s',
      }}>
      {/* Label */}
      <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 8 }}>
        <span style={{
          width: 10, height: 10, borderRadius: '50%', background: color, flexShrink: 0,
        }} />
        <span style={{ fontWeight: 700, color: '#e2e8f0', fontSize: 12 }}>
          Motif {motifIdx}
        </span>
        <span style={{ fontSize: 10, color: '#94a3b8' }}>{label}</span>
      </div>

      {/* Stick figure */}
      <canvas ref={canvasRef} width={160} height={120}
        style={{ width: '100%', height: 120, borderRadius: 6, background: 'rgba(0,0,0,0.3)' }} />

      {/* Usage bars per genotype */}
      <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
        {usages.map(u => (
          <div key={u.genotype} style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <span style={{ fontSize: 9, color: '#64748b', width: 30 }}>{u.genotype}</span>
            <div style={{ flex: 1, height: 4, background: 'rgba(255,255,255,0.08)', borderRadius: 2 }}>
              <div style={{
                width: `${Math.min(100, (u.usage[motifIdx] ?? 0) * 100).toFixed(1)}%`,
                height: '100%', background: color, borderRadius: 2,
              }} />
            </div>
            <span style={{ fontSize: 9, color: '#94a3b8', width: 32, textAlign: 'right' }}>
              {((u.usage[motifIdx] ?? 0) * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}


function MotifDetail({
  motifIdx, label, centroid, color, usages,
}: {
  motifIdx: number, label: string, centroid: number[], color: string, usages: MotifUsage[],
}) {
  const kinematic = centroid.slice(12, 20)
  const kinemLabels = ['Speed', 'AngVel', 'SpineCurv', 'sin(HBA)', 'cos(HBA)', 'EarNorm', 'Grooming', 'Rearing']

  return (
    <div style={{
      background: `${color}10`, border: `1px solid ${color}33`,
      borderRadius: 12, padding: 16,
    }}>
      <div style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 12 }}>
        Motif {motifIdx} — {label}
      </div>
      <div style={{ display: 'flex', gap: 24 }}>
        {/* Kinematic profile */}
        <div>
          <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8 }}>Kinematic signature</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 16px' }}>
            {kinematic.map((val, i) => (
              <div key={i} style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 11 }}>
                <span style={{ color: '#64748b', width: 80 }}>{kinemLabels[i]}:</span>
                <div style={{ flex: 1, height: 4, background: 'rgba(255,255,255,0.08)', borderRadius: 2 }}>
                  <div style={{
                    width: `${Math.min(100, Math.abs(val) * 80)}%`,
                    height: '100%', background: val > 0 ? color : '#f87171', borderRadius: 2,
                  }} />
                </div>
                <span style={{ color: '#94a3b8', width: 40, textAlign: 'right' }}>
                  {val.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Usage per genotype */}
        <div>
          <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8 }}>Usage by genotype</div>
          {usages.map(u => (
            <div key={u.genotype} style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
              <span style={{ fontSize: 12, color: '#e2e8f0', width: 50 }}>{u.genotype}</span>
              <div style={{ width: 150, height: 8, background: 'rgba(255,255,255,0.08)', borderRadius: 4 }}>
                <div style={{
                  width: `${Math.min(100, (u.usage[motifIdx] ?? 0) * 100).toFixed(1)}%`,
                  height: '100%', background: color, borderRadius: 4,
                }} />
              </div>
              <span style={{ fontSize: 11, color: '#94a3b8' }}>
                {((u.usage[motifIdx] ?? 0) * 100).toFixed(2)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}


function SilhouetteChart({ sweep, chosenK }: {
  sweep: { k_values: number[], silhouettes: number[], inertias: number[] },
  chosenK: number,
}) {
  const W = 280, H = 60, pad = 8

  if (!sweep.k_values.length) return null

  const sils = sweep.silhouettes
  const ks   = sweep.k_values
  const maxS = Math.max(...sils), minS = Math.min(...sils)
  const rangeS = maxS - minS || 0.01

  const points = ks.map((k, i) => ({
    x: pad + (i / (ks.length - 1)) * (W - pad * 2),
    y: H - pad - ((sils[i] - minS) / rangeS) * (H - pad * 2),
    k,
  }))

  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ')

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <span style={{ fontSize: 10, color: '#64748b', flexShrink: 0 }}>Silhouette sweep:</span>
      <svg width={W} height={H} style={{ overflow: 'visible' }}>
        <path d={pathD} fill="none" stroke="#818cf8" strokeWidth="1.5" />
        {points.map(p => (
          <circle key={p.k} cx={p.x} cy={p.y} r={p.k === chosenK ? 5 : 2.5}
            fill={p.k === chosenK ? '#6366f1' : '#818cf8'} />
        ))}
        {points.map(p => (
          <text key={p.k} x={p.x} y={H - 1} textAnchor="middle"
            fill="#475569" fontSize="8">{p.k}</text>
        ))}
      </svg>
      <span style={{ fontSize: 10, color: '#64748b' }}>k={chosenK} selected</span>
    </div>
  )
}

