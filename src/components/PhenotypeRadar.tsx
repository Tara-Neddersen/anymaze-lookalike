import { useEffect, useState } from 'react'

interface RadarData {
  features: string[]
  labels: string[]
  WT_means?: number[]
  WT_sems?: number[]
  BPAN_means?: number[]
  BPAN_sems?: number[]
  significance?: string[]
  [key: string]: any
}

interface Props {
  cohortId: string | null
}

const GENOTYPE_COLORS: Record<string, string> = {
  WT:   '#4ade80',
  BPAN: '#f87171',
  KO:   '#fb923c',
  HET:  '#a78bfa',
}

function gColor(g: string) { return GENOTYPE_COLORS[g.toUpperCase()] ?? '#60a5fa' }

export default function PhenotypeRadar({ cohortId }: Props) {
  const [radarData,  setRadarData]  = useState<RadarData | null>(null)
  const [stats,      setStats]      = useState<any>(null)
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState<string | null>(null)

  useEffect(() => {
    if (!cohortId) return
    setLoading(true)
    setError(null)

    Promise.all([
      fetch(`/api/cohorts/${cohortId}/results/statistics`).then(r => r.ok ? r.json() : null).catch(() => null),
      fetch(`/api/cohorts/${cohortId}/results/phenotypes`).then(r => r.ok ? r.json() : null).catch(() => null),
    ]).then(([s, p]) => {
      setStats(s)

      // Extract radar_data from statistics results
      if (s?.radar_data) {
        setRadarData(s.radar_data)
      } else if (p) {
        // Build radar data manually from phenotype vectors
        setRadarData(buildRadarFromPhenotypes(p))
      }
    }).catch(e => setError(e.message))
    .finally(() => setLoading(false))
  }, [cohortId])

  if (!cohortId) return (
    <div style={{ color: '#475569', textAlign: 'center', padding: 40, fontSize: 14 }}>
      Select a cohort and run group comparison to see the phenotype radar.
    </div>
  )

  if (loading) return <div style={{ color: '#94a3b8', padding: 40, textAlign: 'center' }}>Loading…</div>
  if (error) return <div style={{ padding: 20, color: '#f87171', fontSize: 13 }}>{error}</div>
  if (!radarData) return (
    <div style={{ color: '#475569', padding: 40, textAlign: 'center', fontSize: 13 }}>
      No phenotype data. Run the full pipeline (steps 1–6) first.
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ fontWeight: 700, color: '#e2e8f0', fontSize: 14 }}>
        Phenotype Radar — Group Comparison
      </div>

      <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        {/* Radar chart SVG */}
        <RadarSVG data={radarData} />

        {/* Statistics table */}
        {stats && <StatsTable stats={stats} />}
      </div>

      {/* Feature table */}
      {stats && <FeatureTable stats={stats} />}
    </div>
  )
}


function RadarSVG({ data }: { data: RadarData }) {
  const size    = 300
  const cx      = size / 2, cy = size / 2
  const r       = 110
  const labels  = data.labels ?? data.features ?? []
  const n       = labels.length
  if (n === 0) return null

  // Detect group names dynamically
  const groupKeys = Object.keys(data).filter(k => k.endsWith('_means'))
  const groups    = groupKeys.map(k => k.replace('_means', ''))

  const angle = (i: number) => (2 * Math.PI * i) / n - Math.PI / 2

  const toXY = (i: number, frac: number): [number, number] => {
    const a = angle(i)
    return [cx + Math.cos(a) * r * frac, cy + Math.sin(a) * r * frac]
  }

  // Grid rings
  const gridRings = [0.25, 0.5, 0.75, 1.0]

  // Determine scale: max absolute value across all groups
  const allMeans: number[] = groups.flatMap(g => (data[`${g}_means`] ?? []) as number[])
  const maxVal = Math.max(...allMeans.map(Math.abs), 0.01)

  const polygonPath = (means: number[]) => {
    return means.map((m, i) => {
      const frac = Math.min(1, Math.abs(m) / maxVal)
      const [x, y] = toXY(i, frac)
      return `${i === 0 ? 'M' : 'L'}${x},${y}`
    }).join(' ') + 'Z'
  }

  return (
    <svg width={size} height={size} style={{ flexShrink: 0 }}>
      {/* Background */}
      <rect width={size} height={size} fill="rgba(10,10,20,0.8)" rx="12" />

      {/* Grid rings */}
      {gridRings.map(frac => (
        <polygon key={frac}
          points={Array.from({ length: n }, (_, i) => {
            const [x, y] = toXY(i, frac)
            return `${x},${y}`
          }).join(' ')}
          fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
      ))}

      {/* Axis spokes */}
      {Array.from({ length: n }, (_, i) => {
        const [x, y] = toXY(i, 1)
        return <line key={i} x1={cx} y1={cy} x2={x} y2={y}
          stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
      })}

      {/* Group polygons */}
      {groups.map((g) => {
        const means = (data[`${g}_means`] ?? []) as number[]
        if (!means.length) return null
        const col = gColor(g)
        return (
          <g key={g}>
            <path d={polygonPath(means)}
              fill={col + '25'} stroke={col} strokeWidth="2" />
          </g>
        )
      })}

      {/* Axis labels with significance */}
      {labels.map((label, i) => {
        const [x, y] = toXY(i, 1.18)
        const sig = data.significance?.[i] ?? ''
        return (
          <text key={i} x={x} y={y + 4}
            textAnchor="middle" fill="#94a3b8" fontSize="9" fontFamily="sans-serif">
            {label}
            {sig && <tspan fill="#fbbf24" fontWeight="bold"> {sig}</tspan>}
          </text>
        )
      })}

      {/* Legend */}
      {groups.map((g, i) => (
        <g key={g} transform={`translate(8,${size - 14 - i * 16})`}>
          <rect width="10" height="10" fill={gColor(g) + '44'} stroke={gColor(g)} rx="2" />
          <text x="14" y="9" fill="#e2e8f0" fontSize="10" fontFamily="sans-serif">{g}</text>
        </g>
      ))}
    </svg>
  )
}


function StatsTable({ stats }: { stats: any }) {
  const compKeys = Object.keys(stats).filter(k => k.includes('_vs_'))
  if (!compKeys.length) return null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {compKeys.map(key => {
        const comp = stats[key]
        const summary = comp?.summary
        if (!summary) return null
        return (
          <div key={key} style={{
            background: 'rgba(255,255,255,0.04)', borderRadius: 8, padding: 12, fontSize: 12,
          }}>
            <div style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 6 }}>
              {summary.group_a} vs {summary.group_b}
            </div>
            <div style={{ color: '#94a3b8' }}>
              n = {summary.n_a} + {summary.n_b} · {summary.test_method} · {summary.fdr_correction}
            </div>
            <div style={{ color: '#60a5fa', marginTop: 4 }}>
              {summary.n_significant_fdr} / {summary.n_features_tested} features significant (FDR &lt; 5%)
            </div>
            {summary.top_discriminating_features?.length > 0 && (
              <div style={{ color: '#94a3b8', marginTop: 4, fontSize: 11 }}>
                Top: {summary.top_discriminating_features.join(', ')}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}


function FeatureTable({ stats }: { stats: any }) {
  const compKeys = Object.keys(stats).filter(k => k.includes('_vs_'))
  if (!compKeys.length) return null

  const key  = compKeys[0]
  const comp = stats[key]
  const rows = comp?.feature_results ?? []
  const summary = comp?.summary ?? {}

  if (!rows.length) return null

  // Show only top 15 features by significance
  const topRows = [...rows].sort((a, b) => a.p_fdr - b.p_fdr).slice(0, 15)

  const ga = summary.group_a ?? 'WT'
  const gb = summary.group_b ?? 'BPAN'

  return (
    <div style={{
      background: 'rgba(255,255,255,0.04)', borderRadius: 10, overflow: 'hidden',
    }}>
      <div style={{ padding: '10px 14px', fontWeight: 600, color: '#94a3b8', fontSize: 12,
        borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        Feature-level comparison (top {topRows.length} by FDR p-value)
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
        <thead>
          <tr style={{ background: 'rgba(255,255,255,0.04)' }}>
            {['Feature', ga + ' mean', gb + ' mean', 'Cohen d', 'p (FDR)', 'Sig'].map(h => (
              <th key={h} style={{ padding: '6px 10px', textAlign: 'left', color: '#64748b' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {topRows.map((row: any) => (
            <tr key={row.feature} style={{
              borderTop: '1px solid rgba(255,255,255,0.04)',
              background: row.significant_fdr ? 'rgba(99,102,241,0.06)' : 'transparent',
            }}>
              <td style={{ padding: '5px 10px', color: '#cbd5e1', fontFamily: 'monospace', fontSize: 10 }}>
                {row.feature}
              </td>
              <td style={{ padding: '5px 10px', color: gColor(ga) }}>
                {row[`${ga}_mean`]?.toFixed(3) ?? '—'}
              </td>
              <td style={{ padding: '5px 10px', color: gColor(gb) }}>
                {row[`${gb}_mean`]?.toFixed(3) ?? '—'}
              </td>
              <td style={{ padding: '5px 10px', color: Math.abs(row.cohen_d ?? 0) > 0.8 ? '#fbbf24' : '#94a3b8' }}>
                {row.cohen_d?.toFixed(2) ?? '—'}
              </td>
              <td style={{ padding: '5px 10px', color: row.p_fdr < 0.05 ? '#4ade80' : '#94a3b8' }}>
                {row.p_fdr?.toFixed(4) ?? '—'}
              </td>
              <td style={{ padding: '5px 10px', color: '#fbbf24', fontWeight: 700, fontSize: 13 }}>
                {row.stars}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}


function buildRadarFromPhenotypes(ph: any): RadarData {
  if (!ph?.animals || !ph?.raw_vectors) return { features: [], labels: [] }

  const RADAR_FEATURES = [
    'entropy_normalized', 'behavioral_rigidity', 'freezing_frac',
    'thigmotaxis_frac', 'rearing_frac', 'grooming_frac',
    'mean_spine_curvature', 'mean_speed_cm_s', 'path_efficiency',
  ]
  const RADAR_LABELS = [
    'Entropy', 'Rigidity', 'Freezing', 'Thigmotaxis',
    'Rearing', 'Grooming', 'Spine Curv.', 'Speed', 'Path Eff.',
  ]

  const groups: Record<string, number[][]> = {}
  for (let i = 0; i < ph.animals.length; i++) {
    const g = ph.animals[i].genotype
    if (!groups[g]) groups[g] = []
    groups[g].push(RADAR_FEATURES.map(f => ph.raw_vectors[i]?.[f] ?? 0))
  }

  const result: RadarData = { features: RADAR_FEATURES, labels: RADAR_LABELS }
  for (const [g, arrs] of Object.entries(groups)) {
    result[`${g}_means`] = RADAR_FEATURES.map((_, fi) =>
      arrs.reduce((s, a) => s + (a[fi] ?? 0), 0) / arrs.length
    )
    result[`${g}_sems`] = RADAR_FEATURES.map((_, fi) => {
      const vals = arrs.map(a => a[fi] ?? 0)
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length
      const std  = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(vals.length - 1, 1))
      return std / Math.sqrt(vals.length)
    })
  }

  return result
}
