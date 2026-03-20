import { useEffect, useRef, useState, useCallback } from 'react'

interface SequenceProfile {
  transition_matrix: number[][]
  motif_usage_pct: number[]
  transition_entropy: number
  entropy_normalized: number
  self_transition_rate: number
  motif_dwell_mean_s: number[]
  top_trigrams: [number, number, number, number][]
}

interface Props {
  cohortId: string | null
}

const MOTIF_COLORS = [
  '#60a5fa','#f472b6','#34d399','#fbbf24','#a78bfa',
  '#fb923c','#22d3ee','#e879f9','#4ade80','#f87171',
]

export default function TransitionGraph({ cohortId }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [profiles,   setProfiles]   = useState<Record<string, SequenceProfile>>({})
  const [library,    setLibrary]    = useState<{ k: number; auto_labels: string[] } | null>(null)
  const [loading,    setLoading]    = useState(false)
  const [selectedAnimal, setSelectedAnimal] = useState<string | null>(null)
  const [aggregated,     setAggregated]     = useState(true)

  useEffect(() => {
    if (!cohortId) return
    setLoading(true)

    Promise.all([
      fetch(`/api/cohorts/${cohortId}/motif_library`).then(r => r.ok ? r.json() : null).catch(() => null),
      fetch(`/api/cohorts/${cohortId}/results/sequence_profiles`).then(r => r.ok ? r.json() : null).catch(() => null),
    ]).then(([lib, seqProfiles]) => {
      setLibrary(lib)
      if (seqProfiles) setProfiles(seqProfiles)
    }).catch(() => {})
    .finally(() => setLoading(false))
  }, [cohortId])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !library) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const W = canvas.width, H = canvas.height
    const k = library.k

    // Determine which transition matrix to show
    let T: number[][] | null = null
    let usages: number[] | null = null
    let entropy = 0

    const jobIds = Object.keys(profiles)
    if (aggregated && jobIds.length > 0) {
      // Average transition matrices across all animals
      const tSum: number[][] = Array.from({ length: k }, () => new Array(k).fill(0))
      const uSum: number[] = new Array(k).fill(0)
      let entSum = 0

      for (const jid of jobIds) {
        const p = profiles[jid]
        if (!p.transition_matrix) continue
        for (let i = 0; i < k; i++) {
          for (let j = 0; j < k; j++) {
            tSum[i][j] += p.transition_matrix[i]?.[j] ?? 0
          }
          uSum[i] += p.motif_usage_pct[i] ?? 0
        }
        entSum += p.transition_entropy ?? 0
      }

      const n = jobIds.length
      T = tSum.map(row => row.map(v => v / n))
      usages = uSum.map(v => v / n)
      entropy = entSum / n
    } else if (selectedAnimal && profiles[selectedAnimal]) {
      T = profiles[selectedAnimal].transition_matrix
      usages = profiles[selectedAnimal].motif_usage_pct
      entropy = profiles[selectedAnimal].transition_entropy ?? 0
    }

    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = 'rgba(10,10,20,0.95)'
    ctx.fillRect(0, 0, W, H)

    if (!T || !usages) {
      // No data — show placeholder
      ctx.fillStyle = '#334155'
      ctx.font = '13px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('Run "Compute Sequences" in the pipeline', W / 2, H / 2)
      return
    }

    // Layout nodes in a circle
    const cx = W / 2, cy = H / 2
    const radius = Math.min(W, H) * 0.33

    const nodePos: [number, number][] = Array.from({ length: k }, (_, i) => {
      const angle = (2 * Math.PI * i) / k - Math.PI / 2
      return [cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius]
    })

    const maxUsage = Math.max(...usages, 0.01)

    // Draw edges (skip self-loops and very weak transitions)
    const minEdgeT = 0.02
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        if (i === j) continue
        const t = T[i][j]
        if (t < minEdgeT) continue

        const [x1, y1] = nodePos[i]
        const [x2, y2] = nodePos[j]
        const weight = t

        ctx.beginPath()
        ctx.moveTo(x1, y1)

        // Curved edge for bidirectional distinction
        const midX = (x1 + x2) / 2 + (y2 - y1) * 0.15
        const midY = (y1 + y2) / 2 - (x2 - x1) * 0.15
        ctx.quadraticCurveTo(midX, midY, x2, y2)
        ctx.strokeStyle = `rgba(148,163,184,${Math.min(0.8, weight * 3)})`
        ctx.lineWidth = Math.max(0.5, weight * 6)
        ctx.stroke()

        // Arrowhead
        const angle = Math.atan2(y2 - midY, x2 - midX)
        const nodeR = Math.max(8, (usages[j] / maxUsage) * 20)
        const ax = x2 - Math.cos(angle) * nodeR
        const ay = y2 - Math.sin(angle) * nodeR
        ctx.beginPath()
        ctx.moveTo(ax, ay)
        ctx.lineTo(ax - 6 * Math.cos(angle - 0.4), ay - 6 * Math.sin(angle - 0.4))
        ctx.lineTo(ax - 6 * Math.cos(angle + 0.4), ay - 6 * Math.sin(angle + 0.4))
        ctx.closePath()
        ctx.fillStyle = `rgba(148,163,184,${Math.min(0.8, weight * 3)})`
        ctx.fill()
      }
    }

    // Draw self-loops (arc above node)
    for (let i = 0; i < k; i++) {
      const st = T[i][i]
      if (st < 0.1) continue
      const [nx, ny] = nodePos[i]
      ctx.beginPath()
      ctx.arc(nx, ny - 18, 12, 0.3, Math.PI - 0.3)
      ctx.strokeStyle = `rgba(148,163,184,${Math.min(0.7, st)})`
      ctx.lineWidth = Math.max(1, st * 4)
      ctx.stroke()
    }

    // Draw nodes
    for (let i = 0; i < k; i++) {
      const [nx, ny] = nodePos[i]
      const r = Math.max(10, (usages[i] / maxUsage) * 22)
      const col = MOTIF_COLORS[i % MOTIF_COLORS.length]

      ctx.beginPath()
      ctx.arc(nx, ny, r, 0, Math.PI * 2)
      ctx.fillStyle = col + '33'
      ctx.fill()
      ctx.strokeStyle = col
      ctx.lineWidth = 2
      ctx.stroke()

      ctx.fillStyle = '#e2e8f0'
      ctx.font = `bold ${Math.max(9, Math.min(13, r * 0.7))}px sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(String(i), nx, ny)

      // Label
      const label = library.auto_labels[i] ?? `M${i}`
      ctx.fillStyle = '#94a3b8'
      ctx.font = '9px sans-serif'
      ctx.textBaseline = 'alphabetic'
      const labelAngle = Math.atan2(ny - cy, nx - cx)
      ctx.fillText(label, nx + Math.cos(labelAngle) * (r + 12), ny + Math.sin(labelAngle) * (r + 12))
    }

    // Entropy display
    ctx.fillStyle = '#64748b'
    ctx.font = '11px monospace'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'alphabetic'
    ctx.fillText(`H = ${entropy.toFixed(3)} (${(entropy / Math.log(k * k) * 100).toFixed(0)}% max)`, 12, 20)

  }, [profiles, library, selectedAnimal, aggregated])

  useEffect(() => { draw() }, [draw])

  if (!cohortId) return (
    <div style={{ color: '#475569', textAlign: 'center', padding: 40, fontSize: 14 }}>
      Select a cohort and run sequence analysis.
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '100%' }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
        <span style={{ fontWeight: 700, color: '#e2e8f0', fontSize: 14 }}>
          Behavioral Transition Graph
        </span>
        {library && (
          <span style={{ fontSize: 11, color: '#64748b' }}>
            {library.k} motifs
          </span>
        )}
        {Object.keys(profiles).length > 0 && (
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
            <button onClick={() => setAggregated(true)}
              style={tabBtn(aggregated)}>Cohort Mean</button>
            <button onClick={() => setAggregated(false)}
              style={tabBtn(!aggregated)}>Per Animal</button>
          </div>
        )}
      </div>

      {!aggregated && (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {Object.keys(profiles).map(jid => (
            <button key={jid} onClick={() => setSelectedAnimal(jid)}
              style={tabBtn(selectedAnimal === jid)}>
              {jid.slice(0, 8)}
            </button>
          ))}
        </div>
      )}

      <div style={{ flex: 1, position: 'relative', minHeight: 0 }}>
        {loading && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
            justifyContent: 'center', color: '#94a3b8', fontSize: 14,
          }}>
            Loading…
          </div>
        )}
        <canvas ref={canvasRef} width={600} height={480}
          style={{ width: '100%', height: '100%', borderRadius: 12 }} />
      </div>

      {/* Transition entropy bar */}
      {library && Object.keys(profiles).length > 0 && (
        <EntropyBar profiles={profiles} k={library.k} />
      )}
    </div>
  )
}


function EntropyBar({
  profiles, k,
}: {
  profiles: Record<string, SequenceProfile>, k: number, autoLabels?: string[],
}) {
  const maxEnt = Math.log(k * k)
  const sorted = Object.entries(profiles)
    .filter(([, p]) => p.transition_entropy != null)
    .sort(([, a], [, b]) => (b.transition_entropy ?? 0) - (a.transition_entropy ?? 0))

  return (
    <div style={{
      background: 'rgba(255,255,255,0.04)', borderRadius: 8, padding: 12,
    }}>
      <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8 }}>
        Behavioral Entropy by Animal (higher = more flexible behavior)
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {sorted.map(([jid, p]) => (
          <div key={jid} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ fontSize: 10, color: '#64748b', width: 80, fontFamily: 'monospace' }}>
              {jid.slice(0, 8)}
            </span>
            <div style={{ flex: 1, height: 6, background: 'rgba(255,255,255,0.08)', borderRadius: 3 }}>
              <div style={{
                width: `${(p.transition_entropy / maxEnt) * 100}%`,
                height: '100%',
                background: 'linear-gradient(90deg, #60a5fa, #818cf8)',
                borderRadius: 3,
              }} />
            </div>
            <span style={{ fontSize: 10, color: '#94a3b8', width: 40, textAlign: 'right' }}>
              {p.transition_entropy.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

const tabBtn = (active: boolean): React.CSSProperties => ({
  background: active ? 'rgba(99,102,241,0.3)' : 'rgba(255,255,255,0.05)',
  border: `1px solid ${active ? '#818cf8' : 'rgba(255,255,255,0.1)'}`,
  borderRadius: 5, color: active ? '#818cf8' : '#64748b',
  padding: '3px 10px', fontSize: 11, cursor: 'pointer',
})
