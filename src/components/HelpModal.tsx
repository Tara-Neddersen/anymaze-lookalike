/** HelpModal — keyboard shortcuts and quick-reference card. */
interface Props { onClose: () => void }

const SHORTCUTS = [
  { section: 'Video playback', items: [
    { key: 'Space', desc: 'Play / pause' },
    { key: '← →', desc: 'Step one frame back / forward' },
    { key: 'F', desc: 'Toggle fullscreen' },
  ]},
  { section: 'Manual scoring', items: [
    { key: '1 – 9', desc: 'Score behavior category 1–9' },
    { key: '0', desc: 'Score category 10' },
    { key: 'Z', desc: 'Undo last score entry' },
  ]},
  { section: 'Navigation', items: [
    { key: 'Esc', desc: 'Close modals / cancel' },
    { key: '?', desc: 'Open this help panel' },
  ]},
]

const TIPS = [
  'Upload a video in Step 1, fill metadata in Step 2, then draw your arena in Step 3.',
  'Use ⚡ Quick track to skip zone drawing and get results immediately.',
  'Draw a scale bar in the Arena editor (Step 3 → Scale) to get cm-calibrated metrics.',
  'Zones named "open" / "closed" automatically trigger EPM (Elevated Plus Maze) analysis.',
  'Zones named "novel" / "familiar" automatically trigger NOR (Novel Object Recognition) analysis.',
  'Use ↩ New video / same setup to reuse your arena and zones for the next animal.',
  'Batch CSV export includes per-zone breakdown for every animal.',
  'The Longitudinal panel appears once multiple sessions are added to the comparison table.',
  'Click any Zone events row to jump the video to that entry.',
]

export default function HelpModal({ onClose }: Props) {
  return (
    <div className="helpModalWrap">
      <div className="helpOverlay" onClick={onClose} />
      <div className="helpBox">
        <div className="helpHeader">
          <div className="helpTitle">Keyboard shortcuts &amp; tips</div>
          <button className="helpClose" onClick={onClose}>✕</button>
        </div>

        <div className="helpBody">
          <div className="helpShortcutsCol">
            {SHORTCUTS.map(s => (
              <div key={s.section} className="helpSection">
                <div className="helpSectionTitle">{s.section}</div>
                {s.items.map(item => (
                  <div key={item.key} className="helpRow">
                    <kbd className="helpKbd">{item.key}</kbd>
                    <span className="helpDesc">{item.desc}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>

          <div className="helpTipsCol">
            <div className="helpSectionTitle">Tips</div>
            <ul className="helpTips">
              {TIPS.map((tip, i) => (
                <li key={i} className="helpTip">{tip}</li>
              ))}
            </ul>
          </div>
        </div>

        <div className="helpFooter">
          Press <kbd className="helpKbd">?</kbd> anytime to reopen · Press <kbd className="helpKbd">Esc</kbd> to close
        </div>
      </div>
    </div>
  )
}
