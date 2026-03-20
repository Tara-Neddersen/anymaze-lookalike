import { useState, useMemo } from 'react'

export type ZoneEventRow = {
  zone_id: string
  zone_name: string
  entry_t: number
  exit_t: number
  duration_s: number
}

type SortKey = 'entry_t' | 'exit_t' | 'duration_s' | 'zone_name'

type Props = {
  events: ZoneEventRow[]
  zoneColors: Record<string, string>
  onScrub?: (t: number) => void
}

function fmtTime(t: number) {
  const m = Math.floor(t / 60)
  const s = (t % 60).toFixed(2)
  return `${m}:${s.padStart(5, '0')}`
}

export default function ZoneEventLog({ events, zoneColors, onScrub }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>('entry_t')
  const [sortAsc, setSortAsc] = useState(true)
  const [filterZone, setFilterZone] = useState<string>('all')
  const [search, setSearch] = useState('')

  const uniqueZones = useMemo(() => {
    const seen = new Map<string, string>()
    for (const e of events) seen.set(e.zone_id, e.zone_name)
    return Array.from(seen.entries()).map(([id, name]) => ({ id, name }))
  }, [events])

  const sorted = useMemo(() => {
    let rows = [...events]
    if (filterZone !== 'all') rows = rows.filter(r => r.zone_id === filterZone)
    if (search) rows = rows.filter(r => r.zone_name.toLowerCase().includes(search.toLowerCase()))
    rows.sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey]
      if (typeof av === 'number' && typeof bv === 'number') return sortAsc ? av - bv : bv - av
      return sortAsc
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av))
    })
    return rows
  }, [events, sortKey, sortAsc, filterZone, search])

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortAsc(a => !a)
    else { setSortKey(key); setSortAsc(true) }
  }

  function colHead(key: SortKey, label: string) {
    const active = sortKey === key
    return (
      <th className={`zelTh ${active ? 'active' : ''}`} onClick={() => toggleSort(key)}>
        {label} {active ? (sortAsc ? '↑' : '↓') : ''}
      </th>
    )
  }

  // Export to CSV
  function exportCsv() {
    const header = 'zone_id,zone_name,entry_t,exit_t,duration_s\n'
    const rows = sorted.map(r =>
      `${r.zone_id},${r.zone_name},${r.entry_t},${r.exit_t},${r.duration_s}`
    ).join('\n')
    const blob = new Blob([header + rows], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = 'zone_events.csv'; a.click()
    URL.revokeObjectURL(url)
  }

  if (!events.length) {
    return <div style={{ padding: 14, fontSize: 13, color: 'var(--textSub)', textAlign: 'center' }}>
      No zone events — add zones in the arena editor to see entry/exit data
    </div>
  }

  return (
    <div className="zelWrap">
      <div className="zelToolbar">
        <input
          className="zelSearch"
          placeholder="Filter by zone name…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        <select
          className="zelFilter"
          value={filterZone}
          onChange={e => setFilterZone(e.target.value)}
        >
          <option value="all">All zones ({events.length})</option>
          {uniqueZones.map(z => (
            <option key={z.id} value={z.id}>
              {z.name} ({events.filter(e => e.zone_id === z.id).length})
            </option>
          ))}
        </select>
        <button className="btnSm btnSecondary" onClick={exportCsv}>↓ CSV</button>
      </div>
      <div className="zelTableWrap">
        <table className="zelTable">
          <thead>
            <tr>
              <th className="zelTh">Zone</th>
              {colHead('entry_t', 'Entry')}
              {colHead('exit_t', 'Exit')}
              {colHead('duration_s', 'Duration')}
            </tr>
          </thead>
          <tbody>
            {sorted.map((ev, i) => (
              <tr
                key={i}
                className="zelRow"
                onClick={() => onScrub?.(ev.entry_t)}
                title="Click to jump to this event"
              >
                <td className="zelZoneCell">
                  <span className="zelDot" style={{ background: zoneColors[ev.zone_id] || '#7c3aed' }} />
                  {ev.zone_name}
                </td>
                <td className="zelTime">{fmtTime(ev.entry_t)}</td>
                <td className="zelTime">{fmtTime(ev.exit_t)}</td>
                <td className="zelDur">{ev.duration_s.toFixed(2)} s</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="zelFooter">
        {sorted.length} events · Total in zones: {sorted.reduce((s, r) => s + r.duration_s, 0).toFixed(1)} s
      </div>
    </div>
  )
}
