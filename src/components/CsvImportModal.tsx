/**
 * CsvImportModal — Import pre-tracked x/y CSV data from AnyMaze or any tracker.
 */
import { useRef, useState } from 'react'

interface Props {
  onImport: (file: File, fps: number) => void
  onClose: () => void
  importing?: boolean
}

export default function CsvImportModal({ onImport, onClose, importing = false }: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [fps, setFps] = useState(25)

  const EXAMPLE = `time,x,y,zone
0.000,312,251,
0.040,314,249,center
0.080,318,245,center
0.120,322,241,`

  return (
    <div className="calModal">
      <div className="calOverlay" onClick={onClose} />
      <div className="calBox" style={{ maxWidth: 540 }}>
        <div className="calHeader">
          <div className="calTitle">Import tracking CSV</div>
          <button className="calClose" onClick={onClose}>✕</button>
        </div>

        <div style={{ padding: '14px 20px', overflowY: 'auto' }}>
          <p style={{ fontSize: 13, color: 'var(--textSub)', marginBottom: 14, lineHeight: 1.6 }}>
            Import x/y coordinate data exported from <strong>AnyMaze</strong>, Ethovision,
            or any tracker as a CSV. Tracking is skipped — all metrics are computed directly
            from the trajectory.
          </p>

          <div style={{ marginBottom: 14 }}>
            <div style={{ fontSize: 11, color: 'var(--accent)', fontWeight: 700, marginBottom: 6 }}>
              REQUIRED COLUMNS (case-insensitive)
            </div>
            <div style={{ fontSize: 12, color: 'var(--textSub)', lineHeight: 1.8 }}>
              <code style={{ color: 'var(--text)' }}>time</code> — seconds (or will be inferred from fps)<br />
              <code style={{ color: 'var(--text)' }}>x</code> — horizontal position in pixels<br />
              <code style={{ color: 'var(--text)' }}>y</code> — vertical position in pixels<br />
              <code style={{ color: 'var(--text)' }}>zone</code> — zone name/ID (optional)
            </div>
          </div>

          <div style={{ marginBottom: 14 }}>
            <div style={{ fontSize: 11, color: 'var(--accent)', fontWeight: 700, marginBottom: 6 }}>
              EXAMPLE FORMAT
            </div>
            <pre style={{
              background: 'rgba(0,0,0,0.25)', border: '1px solid var(--border)',
              borderRadius: 6, padding: '8px 12px', fontSize: 11,
              color: 'var(--textSub)', fontFamily: 'Space Mono, monospace',
            }}>{EXAMPLE}</pre>
          </div>

          <div style={{ display: 'flex', gap: 14, marginBottom: 14, alignItems: 'flex-end' }}>
            <label style={{ flex: 1 }}>
              <div style={{ fontSize: 11, color: 'var(--textSub)', marginBottom: 5 }}>
                CSV file (.csv, .txt)
              </div>
              <input
                ref={fileRef}
                type="file"
                accept=".csv,.txt,.tsv"
                onChange={e => setFile(e.target.files?.[0] ?? null)}
                style={{ display: 'none' }}
              />
              <button
                className="btnSecondary"
                style={{ width: '100%', padding: '8px 12px', fontSize: 13 }}
                onClick={() => fileRef.current?.click()}
              >
                {file ? `📄 ${file.name}` : '↑ Choose CSV file'}
              </button>
            </label>

            <label style={{ width: 120 }}>
              <div style={{ fontSize: 11, color: 'var(--textSub)', marginBottom: 5 }}>
                Frame rate (fps)
              </div>
              <input
                type="number"
                className="calInput"
                style={{ width: '100%' }}
                min={1} max={120} step={1}
                value={fps}
                onChange={e => setFps(Math.max(1, Number(e.target.value)))}
              />
            </label>
          </div>
        </div>

        <div className="calActions">
          <button className="btnSecondary" onClick={onClose}>Cancel</button>
          <button
            className="btnPrimary"
            disabled={!file || importing}
            onClick={() => file && onImport(file, fps)}
          >
            {importing ? 'Importing…' : '→ Import & analyse'}
          </button>
        </div>
      </div>
    </div>
  )
}
