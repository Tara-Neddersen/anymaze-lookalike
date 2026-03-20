/**
 * Protocol Manager — save and reload arena/zone/scale configurations.
 * Uses localStorage for persistence and supports JSON export/import.
 */
import { useEffect, useRef, useState } from 'react'
import type { ArenaSetup } from './ArenaEditor'

const STORAGE_KEY = 'neurotrack_protocols_v1'

export type Protocol = {
  id: string
  name: string
  apparatus: string
  setup: ArenaSetup
  savedAt: string
  engine?: string
  n_animals?: number
}

function loadProtocols(): Protocol[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveProtocols(protocols: Protocol[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(protocols))
}

type Props = {
  currentSetup: ArenaSetup | null
  onLoad: (setup: ArenaSetup) => void
}

export default function ProtocolManager({ currentSetup, onLoad }: Props) {
  const [protocols, setProtocols] = useState<Protocol[]>(loadProtocols)
  const [name, setName] = useState('')
  const [apparatus, setApparatus] = useState('')
  const [saving, setSaving] = useState(false)
  const importRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    saveProtocols(protocols)
  }, [protocols])

  function handleSave() {
    if (!currentSetup || !name.trim()) return
    const p: Protocol = {
      id: Date.now().toString(36),
      name: name.trim(),
      apparatus: apparatus.trim() || 'Custom',
      setup: currentSetup,
      savedAt: new Date().toISOString(),
    }
    setProtocols(prev => [p, ...prev])
    setName('')
    setSaving(false)
  }

  function handleDelete(id: string) {
    setProtocols(prev => prev.filter(p => p.id !== id))
  }

  function handleExport(p: Protocol) {
    const blob = new Blob([JSON.stringify(p, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${p.name.replace(/\s+/g, '_')}.protocol.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  function handleImport(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = ev => {
      try {
        const p: Protocol = JSON.parse(ev.target?.result as string)
        if (!p.id) p.id = Date.now().toString(36)
        setProtocols(prev => [p, ...prev.filter(x => x.id !== p.id)])
      } catch {
        alert('Invalid protocol file')
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  return (
    <div className="pmWrap">
      {/* Save current */}
      {currentSetup && (
        <div className="pmSaveSection">
          {!saving ? (
            <button className="btnSm btnAccent" onClick={() => setSaving(true)}>
              + Save current setup
            </button>
          ) : (
            <div className="pmSaveForm">
              <input
                className="pmInput"
                placeholder="Protocol name (e.g. Open Field 40cm)"
                value={name}
                onChange={e => setName(e.target.value)}
                autoFocus
                onKeyDown={e => e.key === 'Enter' && handleSave()}
              />
              <input
                className="pmInput"
                placeholder="Apparatus (optional)"
                value={apparatus}
                onChange={e => setApparatus(e.target.value)}
              />
              <div className="pmSaveActions">
                <button className="btnSm btnSecondary" onClick={() => setSaving(false)}>Cancel</button>
                <button className="btnSm btnAccent" onClick={handleSave} disabled={!name.trim()}>Save</button>
              </div>
            </div>
          )}
          <input
            ref={importRef} type="file" accept=".json"
            style={{ display: 'none' }} onChange={handleImport}
          />
          <button className="btnSm btnSecondary" onClick={() => importRef.current?.click()}>
            ↑ Import JSON
          </button>
        </div>
      )}

      {/* Protocol list */}
      {protocols.length === 0 ? (
        <div className="pmEmpty">No saved protocols yet. Analyse a video and save the setup.</div>
      ) : (
        <div className="pmList">
          {protocols.map(p => (
            <div key={p.id} className="pmItem">
              <div className="pmItemInfo">
                <div className="pmItemName">{p.name}</div>
                <div className="pmItemMeta">
                  {p.apparatus} ·{' '}
                  {p.setup.zones.length} zone{p.setup.zones.length !== 1 ? 's' : ''} ·{' '}
                  {p.setup.pxPerCm > 0 ? `${p.setup.pxPerCm.toFixed(1)} px/cm ·` : 'uncalibrated ·'}{' '}
                  {new Date(p.savedAt).toLocaleDateString()}
                </div>
              </div>
              <div className="pmItemActions">
                <button className="btnSm btnAccent" onClick={() => onLoad(p.setup)}>Load</button>
                <button className="btnSm btnSecondary" onClick={() => handleExport(p)}>↓ JSON</button>
                <button className="btnSm btnDanger" onClick={() => handleDelete(p.id)}>✕</button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
