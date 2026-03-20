/**
 * Manual behavioral scoring panel.
 * Researcher presses a keyboard key to start/stop timing a custom behavior.
 * All events are recorded with timestamps and can be exported as CSV.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

export type ScoreEvent = {
  id: string
  behavior: string
  start_t: number
  end_t: number | null  // null = currently recording
  duration_s: number | null
}

const DEFAULT_BEHAVIORS = [
  { name: 'Grooming',         key: 'g', color: '#a855f7' },
  { name: 'Social sniffing',  key: 's', color: '#ec4899' },
  { name: 'Digging',          key: 'd', color: '#f97316' },
  { name: 'Jumping',          key: 'j', color: '#eab308' },
  { name: 'Stretch attend',   key: 'a', color: '#06b6d4' },
  { name: 'Rear (manual)',    key: 'r', color: '#22c55e' },
]

type Props = {
  currentTime: number
  onEventsChange?: (events: ScoreEvent[]) => void
}

export default function ScorePanel({ currentTime, onEventsChange }: Props) {
  const [events, setEvents] = useState<ScoreEvent[]>([])
  const [behaviors, setBehaviors] = useState(DEFAULT_BEHAVIORS)
  const [activeKeys, setActiveKeys] = useState<Set<string>>(new Set())
  const [addingBehavior, setAddingBehavior] = useState(false)
  const [newName, setNewName] = useState('')
  const [newKey, setNewKey] = useState('')
  const [newColor, setNewColor] = useState('#60a5fa')
  const currentTimeRef = useRef(currentTime)
  currentTimeRef.current = currentTime

  const toggleBehavior = useCallback((key: string, behaviorName: string) => {
    const now = currentTimeRef.current
    setEvents(prev => {
      const open = prev.find(e => e.end_t === null && e.behavior === behaviorName)
      if (open) {
        // Close this event
        const dur = now - open.start_t
        const updated = prev.map(e =>
          e.id === open.id
            ? { ...e, end_t: Math.round(now * 1000) / 1000, duration_s: Math.round(dur * 1000) / 1000 }
            : e
        )
        setActiveKeys(k => { const s = new Set(k); s.delete(key); return s })
        return updated
      } else {
        // Start new event
        const ev: ScoreEvent = {
          id: `${behaviorName}-${Date.now()}`,
          behavior: behaviorName,
          start_t: Math.round(now * 1000) / 1000,
          end_t: null,
          duration_s: null,
        }
        setActiveKeys(k => new Set([...k, key]))
        return [...prev, ev]
      }
    })
  }, [])

  // Keyboard listener
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      const b = behaviors.find(b => b.key === e.key.toLowerCase())
      if (b) { e.preventDefault(); toggleBehavior(b.key, b.name) }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [behaviors, toggleBehavior])

  useEffect(() => { onEventsChange?.(events) }, [events, onEventsChange])

  function exportCsv() {
    const header = 'behavior,start_t,end_t,duration_s\n'
    const rows = events
      .filter(e => e.end_t !== null)
      .map(e => `${e.behavior},${e.start_t},${e.end_t},${e.duration_s}`)
      .join('\n')
    const blob = new Blob([header + rows], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = 'manual_scores.csv'; a.click()
    URL.revokeObjectURL(url)
  }

  function addBehavior() {
    if (!newName.trim() || !newKey.trim()) return
    setBehaviors(prev => [...prev, { name: newName.trim(), key: newKey.trim().toLowerCase()[0], color: newColor }])
    setNewName(''); setNewKey(''); setAddingBehavior(false)
  }

  const completedEvents = events.filter(e => e.end_t !== null)
  const fmtT = (t: number) => `${Math.floor(t / 60)}:${(t % 60).toFixed(2).padStart(5, '0')}`

  return (
    <div className="scorePanel">
      {/* Behavior buttons */}
      <div className="scoreBehaviors">
        {behaviors.map(b => {
          const isActive = activeKeys.has(b.key)
          return (
            <button
              key={b.key}
              className={`scoreBehaviorBtn ${isActive ? 'active' : ''}`}
              style={{ '--b-color': b.color } as React.CSSProperties}
              onClick={() => toggleBehavior(b.key, b.name)}
              title={`Keyboard: [${b.key.toUpperCase()}]`}
            >
              <span className="scoreKey">[{b.key.toUpperCase()}]</span>
              <span className="scoreName">{b.name}</span>
              {isActive && <span className="scoreRec">● REC</span>}
            </button>
          )
        })}
        <button className="scoreBehaviorBtn scoreAddBtn" onClick={() => setAddingBehavior(true)}>
          <span>+ Add behavior</span>
        </button>
      </div>

      {/* Add behavior form */}
      {addingBehavior && (
        <div className="scoreAddForm">
          <input className="formInput" placeholder="Behavior name" value={newName} onChange={e => setNewName(e.target.value)} />
          <input className="formInput" placeholder="Key (1 char)" maxLength={1} value={newKey} onChange={e => setNewKey(e.target.value)} style={{ width: 60 }} />
          <input type="color" value={newColor} onChange={e => setNewColor(e.target.value)} style={{ width: 36, height: 34, border: 'none', background: 'none', cursor: 'pointer' }} />
          <button className="btnSm btnAccent" onClick={addBehavior}>Add</button>
          <button className="btnSm btnSecondary" onClick={() => setAddingBehavior(false)}>Cancel</button>
        </div>
      )}

      {/* Event log */}
      {completedEvents.length > 0 && (
        <div className="scoreEventLog">
          <div className="scoreLogHeader">
            <span>Scored events ({completedEvents.length})</span>
            <button className="btnSm btnSecondary" onClick={exportCsv}>↓ Export CSV</button>
            <button className="btnSm btnDanger" onClick={() => setEvents([])}>Clear all</button>
          </div>
          <div className="scoreLogTable">
            {completedEvents.slice(-30).reverse().map(ev => (
              <div key={ev.id} className="scoreLogRow">
                <span className="scoreLogBehavior">{ev.behavior}</span>
                <span className="scoreLogTime">{fmtT(ev.start_t)}</span>
                <span className="scoreLogArrow">→</span>
                <span className="scoreLogTime">{fmtT(ev.end_t!)}</span>
                <span className="scoreLogDur">{ev.duration_s!.toFixed(2)}s</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="scoreHint">
        Click a button or press its keyboard shortcut while the video plays to score behaviors in real time.
      </div>
    </div>
  )
}
