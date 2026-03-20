import { useState } from 'react'

interface ZoneM {
  zone_id: string
  zone_name: string
  time_in_s: number
  entries: number
  latency_first_entry_s: number | null
  mean_speed_in_zone_cm_s: number | null
  distance_in_zone_cm: number | null
}

interface Props {
  zones: ZoneM[]
  duration_s: number
  zoneColors: Record<string, string>
}

type SortKey = keyof ZoneM
type Dir = 'asc' | 'desc'

export default function ZoneTable({ zones, duration_s, zoneColors }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>('time_in_s')
  const [sortDir, setSortDir] = useState<Dir>('desc')

  function toggleSort(k: SortKey) {
    if (sortKey === k) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(k); setSortDir('desc') }
  }

  const sorted = [...zones].sort((a, b) => {
    const av = a[sortKey]
    const bv = b[sortKey]
    // Nulls always sort to bottom
    if (av == null && bv == null) return 0
    if (av == null) return 1
    if (bv == null) return -1
    return sortDir === 'asc'
      ? (av < bv ? -1 : av > bv ? 1 : 0)
      : (av > bv ? -1 : av < bv ? 1 : 0)
  })

  function pct(t: number) {
    return duration_s > 0 ? ((t / duration_s) * 100).toFixed(1) + '%' : '—'
  }

  function fmt(v: number | null, dec = 2) {
    return v != null ? v.toFixed(dec) : '—'
  }

  function th(label: string, k: SortKey) {
    const active = sortKey === k
    return (
      <th onClick={() => toggleSort(k)} className={active ? 'zt-active' : ''}>
        {label} {active ? (sortDir === 'asc' ? '↑' : '↓') : ''}
      </th>
    )
  }

  if (!zones.length) return <div className="zt-empty">No zones defined.</div>

  return (
    <div className="zt-wrap">
      <table className="zt">
        <thead>
          <tr>
            {th('Zone', 'zone_name')}
            {th('Time (s)', 'time_in_s')}
            <th>Time %</th>
            {th('Entries', 'entries')}
            {th('Latency (s)', 'latency_first_entry_s')}
            {th('Speed (cm/s)', 'mean_speed_in_zone_cm_s')}
            {th('Dist (cm)', 'distance_in_zone_cm')}
          </tr>
        </thead>
        <tbody>
          {sorted.map(z => (
            <tr key={z.zone_id}>
              <td>
                <span className="zone-dot" style={{ background: zoneColors[z.zone_id] || '#888' }} />
                {z.zone_name}
              </td>
              <td>{z.time_in_s.toFixed(2)}</td>
              <td>{pct(z.time_in_s)}</td>
              <td>{z.entries}</td>
              <td>{fmt(z.latency_first_entry_s)}</td>
              <td>{fmt(z.mean_speed_in_zone_cm_s)}</td>
              <td>{fmt(z.distance_in_zone_cm, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
