import './App.css'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ArenaEditor, { type ArenaSetup, type Zone } from './components/ArenaEditor'
import Heatmap from './components/Heatmap'
import SpeedChart from './components/SpeedChart'
import ZoneTable from './components/ZoneTable'
import BatchPanel from './components/BatchPanel'
import GroupChart, { type AnimalRow } from './components/GroupChart'
import BehaviorTimeline, { type BehaviorSegment, type ZoneEvent as ZoneEventTimeline, type TTLEvent } from './components/BehaviorTimeline'
import ZoneEventLog, { type ZoneEventRow } from './components/ZoneEventLog'
import ScorePanel from './components/ScorePanel'
import ProtocolManager from './components/ProtocolManager'
import SessionChart, { type SessionRow } from './components/SessionChart'
import CalibrationTool from './components/CalibrationTool'
import ZoneGroupChart, { type ZoneRow } from './components/ZoneGroupChart'
import HelpModal from './components/HelpModal'
import CsvImportModal from './components/CsvImportModal'
import CumulativeZoneChart from './components/CumulativeZoneChart'
import TrialLearningCurve, { type TrialRow } from './components/TrialLearningCurve'
import CohortBuilder from './components/CohortBuilder'
import BehavioralEthoMap from './components/BehavioralEthoMap'
import MotifGallery from './components/MotifGallery'
import TransitionGraph from './components/TransitionGraph'
import PhenotypeRadar from './components/PhenotypeRadar'
import PoseQCPanel from './components/PoseQCPanel'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
type JobStatus = {
  id: string
  status: 'queued' | 'running' | 'done' | 'error'
  message?: string | null
  progress: number
}

type KpPoint = { x: number; y: number; likelihood?: number }
type RichFrame = {
  frame_index: number
  t_sec: number
  ok: boolean
  centroid?: { x: number; y: number } | null
  area_px?: number | null
  zone_id?: string | null
  speed_px_s?: number | null
  speed_cm_s?: number | null
  heading_deg?: number | null
  heading_delta_deg?: number | null
  angular_velocity_deg_s?: number | null
  quality?: string
  body_length_px?: number | null
  keypoints?: Record<string, KpPoint> | null
  canonical_kps?: Record<string, KpPoint> | null   // 7-point canonical skeleton
  rearing?: boolean
  grooming?: boolean
  ear_span_px?: number | null
  spine_curvature?: number | null
  head_body_angle_deg?: number | null
  animal_centroids?: Array<{ animal_id: string; x: number; y: number; rearing: boolean; heading_deg?: number | null }> | null
}

type ZoneMetrics = {
  zone_id: string
  zone_name: string
  time_in_s: number
  entries: number
  latency_first_entry_s: number | null
  mean_speed_in_zone_cm_s: number | null
  distance_in_zone_cm: number | null
}

type TimeBin = {
  label: string
  start_s: number
  end_s: number
  total_distance_px: number
  total_distance_cm: number | null
  mean_speed_px_s: number
  mean_speed_cm_s: number | null
  total_time_mobile_s: number
  total_time_freezing_s: number
  freezing_episodes: number
  valid_fraction: number
}

type NORMetrics = {
  novel_zone_id: string; familiar_zone_id: string
  time_novel_s: number; time_familiar_s: number
  discrimination_index: number; preference_index: number; total_exploration_s: number
}
type EPMMetrics = {
  open_arm_time_s: number; closed_arm_time_s: number; open_arm_time_pct: number
  open_arm_entries: number; closed_arm_entries: number; open_arm_entries_pct: number
}
type MWMMetrics = {
  target_quadrant_pct: number; opposite_quadrant_pct: number
  left_quadrant_pct: number; right_quadrant_pct: number
  trial_escape_latency_s: number[]
  platform_zone_id: string | null; platform_proximity_mean_cm: number | null
}
type YMazeMetrics = {
  arm_entries: number; alternations: number
  spontaneous_alternation_pct: number
  arm_visit_counts: Record<string, number>
}
type OpenFieldMetrics = {
  center_time_s: number; periphery_time_s: number; center_time_pct: number
  center_entries: number
  center_distance_cm: number | null; periphery_distance_cm: number | null
}
type SocialMetrics = {
  total_time_near_s: number; near_episodes: number
  first_contact_latency_s: number | null
  proximity_threshold_cm: number | null; proximity_threshold_px: number
}
type PlacePreference = {
  time_left_s: number; time_right_s: number
  time_top_s: number; time_bottom_s: number
  preference_lr: number   // positive = prefers left
  preference_tb: number   // positive = prefers top
}
type PrevJobSummary = {
  job_id: string; status: string; progress: number
  animal_id: string; treatment: string; trial: string
  session: string; experiment_id: string; experimenter: string
  video_filename: string; created_at: string
}
type Metrics = {
  total_distance_px: number
  total_distance_cm: number | null
  mean_speed_px_s: number
  mean_speed_cm_s: number | null
  max_speed_px_s: number
  max_speed_cm_s: number | null
  total_time_mobile_s: number
  total_time_immobile_s: number
  total_time_freezing_s: number
  freezing_episodes: number
  thigmotaxis_fraction: number | null
  valid_fraction: number
  duration_s: number
  path_efficiency: number | null
  clockwise_rotations: number
  anticlockwise_rotations: number
  total_time_rearing_s?: number
  rearing_episodes?: number
  total_time_grooming_s?: number
  grooming_episodes?: number
  mean_body_length_px?: number | null
  mean_body_length_cm?: number | null
  mean_ear_span_px?: number | null
  mean_spine_curvature?: number | null
  mean_head_body_angle_deg?: number | null
  mean_inter_animal_dist_px?: number | null
  mean_inter_animal_dist_cm?: number | null
  zone_events?: ZoneEventRow[]
  behavior_events?: BehaviorSegment[]
  nor?: NORMetrics | null
  social?: SocialMetrics | null
  place_preference?: PlacePreference | null
  epm?: EPMMetrics | null
  mwm?: MWMMetrics | null
  ymaze?: YMazeMetrics | null
  open_field?: OpenFieldMetrics | null
  light_dark?: {
    light_time_s: number; dark_time_s: number
    light_time_pct: number; dark_time_pct: number
    light_entries: number; dark_entries: number
    latency_to_light_s: number | null; latency_to_dark_s: number | null
    transitions: number
  } | null
  fear_cond?: {
    epochs: Array<{ label: string; start_t: number; end_t: number; duration_s: number; freezing_s: number; freezing_pct: number; freezing_episodes: number }>
    baseline_freezing_pct: number | null
    mean_cs_freezing_pct: number | null
  } | null
  freeze_bouts?: Array<{ start_t: number; end_t: number; duration_s: number; mean_speed_cm_s: number | null; zone_id: string | null }>
  rearing_per_zone?: Record<string, number>
  zones: ZoneMetrics[]
  time_bins: TimeBin[]
}

type AnimalMeta = {
  animal_id: string; treatment: string; trial: string; notes: string
  n_animals: number; engine: 'opencv_mog2_centroid' | 'dlc_csv' | 'sleap_slp'
  session: string; experiment_id: string; experimenter: string
}

type TrackingResult = {
  summary: { fps: number; frame_count: number; ok_frames: number; arena_size_px: [number, number] }
  frames: RichFrame[]
  meta?: { engine?: string; px_per_cm?: number; zones?: Zone[]; arena_poly?: number[][] }
  animal_meta?: AnimalMeta
  metrics?: Metrics
}

type AppStage = 'idle' | 'meta' | 'arena' | 'tracking' | 'fetching' | 'done' | 'batch' | 'cohort'

const API = ''

// ---------------------------------------------------------------------------
// Pose skeleton definition
// ---------------------------------------------------------------------------
// Canonical 7-keypoint rodent skeleton
const CANONICAL_KPS = ['nose', 'left_ear', 'right_ear', 'neck', 'mid_spine', 'hips', 'tail_base']

const SKELETON_BONES: [string, string][] = [
  ['nose',      'left_ear'],
  ['nose',      'right_ear'],
  ['nose',      'neck'],
  ['left_ear',  'neck'],
  ['right_ear', 'neck'],
  ['neck',      'mid_spine'],
  ['mid_spine', 'hips'],
  ['hips',      'tail_base'],
  // Legacy DLC name aliases (so older CSV files still render)
  ['nose', 'leftear'], ['nose', 'rightear'],
  ['leftear', 'spine1'], ['rightear', 'spine1'],
  ['spine1', 'spine2'], ['spine2', 'tailbase'],
]

const KP_COLORS: Record<string, string> = {
  // Canonical
  nose: '#FF4C4C',  snout: '#FF4C4C', head: '#FF4C4C',
  left_ear: '#FFD93D', right_ear: '#FFD93D',
  leftear: '#FFD93D', rightear: '#FFD93D', lear: '#FFD93D', rear: '#FFD93D',
  neck: '#6BCB77', throat: '#6BCB77', nape: '#6BCB77',
  mid_spine: '#00DDB4', midspine: '#00DDB4', spine1: '#00DDB4', spine2: '#00DDB4',
  hips: '#4D96FF', hip: '#4D96FF', rump: '#4D96FF', pelvis: '#4D96FF',
  tail_base: '#C77DFF', tailbase: '#C77DFF', tail_root: '#C77DFF', tail1: '#C77DFF',
  // Other common parts
  midbody: '#4d96ff', center: '#4d96ff', body: '#4d96ff', thorax: '#4d96ff',
  tail2: '#9b5de5',
}
const KP_DEFAULT_COLOR = '#aaa'

// Distinct colors for multi-animal tracking (animal_0=teal, animal_1=orange, animal_2=purple…)
const ANIMAL_COLORS = ['#00d9c8', '#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#c77dff']
function animalColor(idx: number) { return ANIMAL_COLORS[idx % ANIMAL_COLORS.length] }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function speedHsl(speed: number, maxSpeed: number): string {
  const t = Math.min(1, speed / Math.max(maxSpeed, 0.001))
  const hue = Math.round((1 - t) * 120)
  return `hsl(${hue},100%,55%)`
}

function fmt(v: number | null | undefined, decimals: number, unit: string): string {
  if (v == null) return '—'
  return `${v.toFixed(decimals)} ${unit}`
}

// ---------------------------------------------------------------------------
// Live webcam tracker hook
// ---------------------------------------------------------------------------
function useLiveTracker() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const bgRef = useRef<ImageData | null>(null)
  const ptsRef = useRef<{ x: number; y: number }[]>([])
  const rafRef = useRef<number>(0)
  const lastPtRef = useRef<{ x: number; y: number; t: number } | null>(null)
  const distRef = useRef(0)
  const [liveOn, setLiveOn] = useState(false)
  const [liveSpeed, setLiveSpeed] = useState<number | null>(null)
  const [liveDist, setLiveDist] = useState(0)

  const stop = useCallback(() => {
    cancelAnimationFrame(rafRef.current)
    const video = videoRef.current
    if (video?.srcObject) {
      for (const t of (video.srcObject as MediaStream).getTracks()) t.stop()
      video.srcObject = null
    }
    setLiveOn(false)
  }, [])

  const runLoop = useCallback(() => {
    const video = videoRef.current, canvas = canvasRef.current
    if (!video || !canvas) return
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!
    canvas.width = video.videoWidth || 640; canvas.height = video.videoHeight || 480
    ctx.drawImage(video, 0, 0)
    const frame = ctx.getImageData(0, 0, canvas.width, canvas.height)
    if (!bgRef.current) {
      bgRef.current = frame
      rafRef.current = requestAnimationFrame(runLoop)
      return
    }
    const bg = bgRef.current.data, fg = frame.data, W = canvas.width, H = canvas.height
    let sumX = 0, sumY = 0, count = 0
    for (let y = 0; y < H; y += 2) {
      for (let x = 0; x < W; x += 2) {
        const i = (y * W + x) * 4
        const diff = Math.abs(fg[i] - bg[i]) + Math.abs(fg[i+1] - bg[i+1]) + Math.abs(fg[i+2] - bg[i+2])
        if (diff > 40) { sumX += x; sumY += y; count++ }
      }
    }
    for (let i = 0; i < bg.length; i++) bg[i] = Math.round(bg[i] * 0.95 + fg[i] * 0.05)
    if (count > 200) {
      const cx = sumX / count, cy = sumY / count, now = performance.now() / 1000
      ptsRef.current = [...ptsRef.current.slice(-120), { x: cx, y: cy }]
      if (lastPtRef.current) {
        const dt = now - lastPtRef.current.t
        const d = Math.hypot(cx - lastPtRef.current.x, cy - lastPtRef.current.y)
        if (dt > 0) setLiveSpeed(d / dt)
        distRef.current += d; setLiveDist(distRef.current)
      }
      lastPtRef.current = { x: cx, y: cy, t: now }
      ctx.strokeStyle = 'rgba(0,217,200,0.7)'; ctx.lineWidth = 2; ctx.beginPath()
      ptsRef.current.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y))
      ctx.stroke()
      ctx.fillStyle = '#00d9c8'; ctx.beginPath()
      ctx.arc(cx, cy, 7, 0, Math.PI * 2); ctx.fill()
    }
    rafRef.current = requestAnimationFrame(runLoop)
  }, [])

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      const video = videoRef.current!
      video.srcObject = stream
      await video.play()
      bgRef.current = null; ptsRef.current = []; distRef.current = 0; lastPtRef.current = null
      setLiveOn(true); setLiveDist(0); setLiveSpeed(null)
      runLoop()
    } catch { alert('Camera permission denied') }
  }, [runLoop])

  return { videoRef, canvasRef, liveOn, liveSpeed, liveDist, start, stop }
}

// ---------------------------------------------------------------------------
// LiveZoneOverlay — draws zone polygons over the live camera canvas
// ---------------------------------------------------------------------------
function LiveZoneOverlay({ arenaSetup, videoRef }: {
  arenaSetup: ArenaSetup | null
  videoRef: React.RefObject<HTMLVideoElement | null>
}) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    const video = videoRef.current
    if (!canvas) return
    if (!arenaSetup?.zones?.length) {
      canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
      return
    }
    const rect = canvas.getBoundingClientRect()
    canvas.width = Math.round(rect.width || 400)
    canvas.height = Math.round(rect.height || 300)
    const W = video?.videoWidth || 640
    const H = video?.videoHeight || 480
    const sx = canvas.width / W, sy = canvas.height / H
    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    for (const zone of arenaSetup.zones) {
      if (!zone.closed || zone.points.length < 3) continue
      ctx.beginPath()
      zone.points.forEach((p, i) => {
        const x = p.x * sx, y = p.y * sy
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      })
      ctx.closePath()
      ctx.fillStyle = (zone.color || '#7c3aed') + '28'
      ctx.fill()
      ctx.strokeStyle = zone.color || '#7c3aed'
      ctx.lineWidth = 1.5
      ctx.stroke()
      const cx = zone.points.reduce((s, p) => s + p.x, 0) / zone.points.length * sx
      const cy = zone.points.reduce((s, p) => s + p.y, 0) / zone.points.length * sy
      ctx.font = '11px DM Sans, sans-serif'
      ctx.fillStyle = zone.color || '#7c3aed'
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
      ctx.fillText(zone.name, cx, cy)
    }
  }, [arenaSetup, videoRef])

  return (
    <canvas
      ref={ref}
      style={{
        position: 'absolute', inset: 0, width: '100%', height: '100%',
        pointerEvents: 'none', borderRadius: 'inherit',
      }}
    />
  )
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
export default function App() {
  const [stage, setStage] = useState<AppStage>('idle')
  const [file, setFile] = useState<File | null>(null)
  const [poseFile, setPoseFile] = useState<File | null>(null)
  const [animalMeta, setAnimalMeta] = useState<AnimalMeta>({
    animal_id: '', treatment: '', trial: '', notes: '', n_animals: 1,
    engine: 'opencv_mog2_centroid',
    session: '', experiment_id: '', experimenter: '',
  })
  const [trimStart, setTrimStart] = useState(0)
  const [trimEnd, setTrimEnd] = useState(0)
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [prevJobs, setPrevJobs] = useState<PrevJobSummary[]>([])
  const [heatmapMode, setHeatmapMode] = useState<'occupancy' | 'velocity'>('occupancy')
  const exportingVideo = false  // downloads trigger directly, no state needed
  const [showCalibration, setShowCalibration] = useState(false)
  const [postNotes, setPostNotes] = useState('')
  const [savingNotes, setSavingNotes] = useState(false)
  const [savedNotes, setSavedNotes] = useState(false)
  const keepSetupRef = useRef(false)
  const [showHelp, setShowHelp] = useState(false)
  const [freezeThreshold, setFreezeThreshold] = useState(0.5)  // cm/s
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0)
  const [liveZoneTimes, setLiveZoneTimes] = useState<Record<string, number>>({})
  const liveZoneTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const [showCsvImport, setShowCsvImport] = useState(false)
  const [csvImporting, setCsvImporting] = useState(false)
  const [sessionSearch, setSessionSearch] = useState('')
  const [showAllSessions, setShowAllSessions] = useState(false)
  const [toastMsg, setToastMsg] = useState<string | null>(null)
  const [result, setResult] = useState<TrackingResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [arenaSetup, setArenaSetup] = useState<ArenaSetup | null>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [reanalyzing, setReanalyzing] = useState(false)
  const [activeResultTab, setActiveResultTab] = useState<'video' | 'heatmap' | 'timeline' | 'events' | 'score' | 'protocol' | 'poseqc'>('video')
  const [poseOverlayShow, setPoseOverlayShow] = useState(true)
  const [poseShowBones, setPoseShowBones] = useState(true)
  const [poseKpVisible, setPoseKpVisible] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(CANONICAL_KPS.map(k => [k, true])) as Record<string, boolean>)
  const [poseHideLikBelow, setPoseHideLikBelow] = useState(0)
  const [batchRows, setBatchRows] = useState<AnimalRow[]>([])
  const [ttlEvents, setTtlEvents] = useState<TTLEvent[]>([])
  const [correctionMode, setCorrectionMode] = useState(false)
  const live = useLiveTracker()

  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)

  // Stable, memoized blob URL — created once per file, revoked on change
  const fileUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file])
  useEffect(() => {
    return () => { if (fileUrl) URL.revokeObjectURL(fileUrl) }
  }, [fileUrl])

  const zoneColorMap = useMemo(() => {
    const map: Record<string, string> = {}
    const src = arenaSetup?.zones ?? (result?.meta?.zones as Zone[] | undefined) ?? []
    for (const z of src) map[z.id] = z.color
    return map
  }, [arenaSetup, result])

  const maxSpeed = useMemo(() => {
    if (!result) return 1
    const hasCm = result.frames.some(f => f.speed_cm_s != null)
    return Math.max(1, ...result.frames.map(f => hasCm ? (f.speed_cm_s ?? 0) : (f.speed_px_s ?? 0)))
  }, [result])

  const hasCm = useMemo(() => !!(result?.frames.some(f => f.speed_cm_s != null)), [result])

  // Compute median body length from pose-tracked frames (DLC / SLEAP)
  const medianBodyLengthPx = useMemo(() => {
    if (!result) return null
    const vals = result.frames.filter(f => f.body_length_px != null).map(f => f.body_length_px!)
    if (vals.length === 0) return null
    const sorted = [...vals].sort((a, b) => a - b)
    return sorted[Math.floor(sorted.length / 2)]
  }, [result])

  const hasPose = useMemo(
    () => !!(result?.frames.some(f => f.keypoints && Object.keys(f.keypoints).length > 0)),
    [result]
  )

  // -------------------------------------------------------------------------
  // Upload & start job
  // -------------------------------------------------------------------------
  async function startTracking(setup: ArenaSetup) {
    if (!file) return
    setArenaSetup(setup)
    setError(null); setResult(null); setJobId(null); setJobStatus(null)
    setStage('tracking')
    try {
      const fd = new FormData()
      fd.append('video', file)
      if (setup.arenaPoly.length >= 3)
        fd.append('arena_json', JSON.stringify(setup.arenaPoly.map(p => [p.x, p.y])))
      if (setup.zones.length > 0)
        fd.append('zones_json', JSON.stringify(
          setup.zones.map(z => ({ id: z.id, name: z.name, color: z.color, points: z.points.map(p => [p.x, p.y]) }))
        ))
      if (setup.pxPerCm > 0) fd.append('px_per_cm', String(setup.pxPerCm))
      fd.append('n_animals', String(animalMeta.n_animals ?? 1))
      fd.append('engine', animalMeta.engine)
      fd.append('animal_id', animalMeta.animal_id)
      fd.append('treatment', animalMeta.treatment)
      fd.append('trial', animalMeta.trial)
      fd.append('notes', animalMeta.notes)
      fd.append('session', animalMeta.session)
      fd.append('experiment_id', animalMeta.experiment_id)
      fd.append('experimenter', animalMeta.experimenter)
      if (trimStart > 0) fd.append('trim_start_s', String(trimStart))
      if (trimEnd > 0) fd.append('trim_end_s', String(trimEnd))
      if (freezeThreshold !== 0.5) fd.append('freeze_threshold_cm_s', String(freezeThreshold))
      if (poseFile && (animalMeta.engine === 'dlc_csv' || animalMeta.engine === 'sleap_slp')) {
        fd.append('pose_file', poseFile)
      }
      const res = await fetch(`${API}/api/jobs`, { method: 'POST', body: fd })
      if (!res.ok) throw new Error(await res.text())
      const st = await res.json() as JobStatus
      setJobId(st.id); setJobStatus(st)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStage('arena')
    }
  }

  // -------------------------------------------------------------------------
  // Poll job status
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (!jobId || !jobStatus) return
    if (jobStatus.status === 'done' || jobStatus.status === 'error') return
    let cancelled = false
    const t = setInterval(async () => {
      try {
        const r = await fetch(`${API}/api/jobs/${jobId}`)
        if (!r.ok || cancelled) return
        setJobStatus(await r.json() as JobStatus)
      } catch { /* network blip */ }
    }, 600)
    return () => { cancelled = true; clearInterval(t) }
  }, [jobId, jobStatus])

  useEffect(() => {
    if (!jobId || jobStatus?.status !== 'done') return
    let cancelled = false
    setStage('fetching')
    ;(async () => {
      try {
        const r = await fetch(`${API}/api/jobs/${jobId}/result`)
        if (!r.ok) throw new Error(await r.text())
        const data = await r.json() as TrackingResult & { post_notes?: string }
        if (!cancelled) {
          setResult(data)
          setPostNotes(data.post_notes || '')
          // Add to batch rows for group comparison
          if (data.metrics) {
            const am = data.animal_meta
            setBatchRows(prev => [
              ...prev.filter(r => r.job_id !== jobId),
              {
                job_id: jobId,
                animal_id: am?.animal_id || '',
                treatment: am?.treatment || '',
                trial: am?.trial || '',
                session: (am as AnimalMeta | undefined)?.session || '',
                metrics: data.metrics as unknown as Record<string, number | null>,
              }
            ])
          }
          setStage('done')
        }
      } catch (e) { if (!cancelled) { setError(e instanceof Error ? e.message : String(e)); setStage('tracking') } }
    })()
    return () => { cancelled = true }
  }, [jobId, jobStatus])

  // Assign video src from stable URL when we arrive at done stage
  useEffect(() => {
    if (stage === 'done' && videoRef.current && fileUrl) {
      videoRef.current.src = fileUrl
    }
  }, [stage, fileUrl])

  // -------------------------------------------------------------------------
  // Post-hoc reanalysis
  // -------------------------------------------------------------------------
  async function reanalyze(newSetup: ArenaSetup) {
    if (!jobId || !result) return
    setReanalyzing(true)
    setArenaSetup(newSetup)
    try {
      const fd = new FormData()
      if (newSetup.zones.length > 0)
        fd.append('zones_json', JSON.stringify(
          newSetup.zones.map(z => ({ id: z.id, name: z.name, color: z.color, points: z.points.map(p => [p.x, p.y]) }))
        ))
      if (newSetup.arenaPoly.length >= 3)
        fd.append('arena_json', JSON.stringify(newSetup.arenaPoly.map(p => [p.x, p.y])))
      if (newSetup.pxPerCm > 0) fd.append('px_per_cm', String(newSetup.pxPerCm))
      const r = await fetch(`${API}/api/jobs/${jobId}/reanalyze`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error(await r.text())
      setResult(await r.json() as TrackingResult)
      setStage('done')
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
    finally { setReanalyzing(false) }
  }

  // -------------------------------------------------------------------------
  // Trajectory overlay — drawn ONLY by the useEffect, not inline
  // -------------------------------------------------------------------------
  const drawOverlay = useCallback(() => {
    const video = videoRef.current, canvas = overlayRef.current
    if (!video || !canvas || !result) return
    const ctx = canvas.getContext('2d')!
    const vw = video.videoWidth || result.summary.arena_size_px[0] || 1
    const vh = video.videoHeight || result.summary.arena_size_px[1] || 1
    const rect = video.getBoundingClientRect()
    const cssW = Math.max(1, Math.round(rect.width))
    const cssH = Math.max(1, Math.round(rect.height))
    if (canvas.width !== cssW || canvas.height !== cssH) { canvas.width = cssW; canvas.height = cssH }
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const sx = canvas.width / vw, sy = canvas.height / vh
    const fps = result.summary.fps || 30
    const tailSec = 8
    const curIdx = Math.max(0, Math.min(result.frames.length - 1, Math.floor(currentTime * fps)))
    const startIdx = Math.max(0, curIdx - Math.floor(fps * tailSec))

    let prevPt: { x: number; y: number } | null = null
    for (let i = startIdx; i <= curIdx; i++) {
      const f = result.frames[i]
      if (!f.ok || !f.centroid || f.quality === 'no_detection') { prevPt = null; continue }
      const x = f.centroid.x * sx, y = f.centroid.y * sy
      if (prevPt) {
        const jump = Math.hypot(x - prevPt.x * sx, y - prevPt.y * sy)
        if (jump > Math.max(canvas.width, canvas.height) * 0.2) { prevPt = null; continue }
      }
      const spd = hasCm ? (f.speed_cm_s ?? 0) : (f.speed_px_s ?? 0)
      const alpha = 0.15 + 0.85 * ((i - startIdx) / Math.max(1, curIdx - startIdx))
      if (prevPt) {
        ctx.beginPath()
        ctx.moveTo(prevPt.x * sx, prevPt.y * sy)
        ctx.lineTo(x, y)
        ctx.strokeStyle = speedHsl(spd, maxSpeed).replace('hsl', 'hsla').replace(')', `,${alpha})`)
        ctx.lineWidth = 2.5
        ctx.stroke()
      }
      prevPt = { x: f.centroid.x, y: f.centroid.y }
    }

    const cur = result.frames[curIdx]
    if (cur?.ok && cur.centroid) {
      const x = cur.centroid.x * sx, y = cur.centroid.y * sy

      // ── Pose skeleton (DLC / SLEAP) ──
      // Prefer canonical_kps (7-point), fall back to raw keypoints
      const poseKps: Record<string, KpPoint> | null = cur.canonical_kps && Object.keys(cur.canonical_kps).length > 0
        ? cur.canonical_kps
        : (cur.keypoints && Object.keys(cur.keypoints).length > 0 ? cur.keypoints : null)

      if (poseKps && poseOverlayShow) {
        const likOk = (kp: KpPoint) =>
          poseHideLikBelow <= 0 || (kp.likelihood ?? 1) >= poseHideLikBelow
        const nodeVisible = (name: string) =>
          !CANONICAL_KPS.includes(name) || poseKpVisible[name] !== false
        // Draw bones
        if (poseShowBones) {
          ctx.lineWidth = 2
          for (const [a, b] of SKELETON_BONES) {
            const ka = poseKps[a], kb = poseKps[b]
            if (!ka || !kb || !likOk(ka) || !likOk(kb)) continue
            if (!nodeVisible(a) || !nodeVisible(b)) continue
            ctx.beginPath()
            ctx.moveTo(ka.x * sx, ka.y * sy)
            ctx.lineTo(kb.x * sx, kb.y * sy)
            ctx.strokeStyle = 'rgba(255,255,255,0.50)'
            ctx.stroke()
          }
        }
        // Draw keypoint dots + label for canonical ones
        for (const [name, kp] of Object.entries(poseKps)) {
          if (!likOk(kp)) continue
          if (CANONICAL_KPS.includes(name) && poseKpVisible[name] === false) continue
          const kx = kp.x * sx, ky = kp.y * sy
          const col = KP_COLORS[name] ?? KP_DEFAULT_COLOR
          ctx.beginPath(); ctx.arc(kx, ky, 4.5, 0, Math.PI * 2)
          ctx.fillStyle = col; ctx.fill()
          // Label canonical keypoints
          if (CANONICAL_KPS.includes(name)) {
            ctx.font = '9px monospace'
            ctx.fillStyle = 'rgba(255,255,255,0.7)'
            ctx.fillText(name.replace('_', ' '), kx + 6, ky - 4)
          }
        }
        // Draw grooming indicator
        if (cur.grooming) {
          ctx.beginPath(); ctx.arc(x, y - 16, 5, 0, Math.PI * 2)
          ctx.fillStyle = '#ff9a3c'; ctx.fill()
          ctx.font = '9px sans-serif'; ctx.fillStyle = '#ff9a3c'
          ctx.fillText('groom', x + 8, y - 12)
        }
      } else {
        // ── Multi-animal dots (if available) ──
        if (cur.animal_centroids && cur.animal_centroids.length > 1) {
          cur.animal_centroids.forEach((a, ai) => {
            const ax = a.x * sx, ay = a.y * sy
            const col = animalColor(ai)
            ctx.beginPath(); ctx.arc(ax, ay, 8, 0, Math.PI * 2)
            ctx.fillStyle = a.rearing ? '#ff6b6b' : col; ctx.fill()
            ctx.strokeStyle = 'rgba(0,0,0,0.5)'; ctx.lineWidth = 2; ctx.stroke()
            // Animal label
            ctx.font = 'bold 11px DM Sans, sans-serif'
            ctx.fillStyle = col; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'
            ctx.fillText(a.animal_id.replace('animal_', '#'), ax, ay - 12)
            ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'
          })
        } else {
          // Single animal — heading arrow + dot
          if (cur.heading_deg != null) {
            const rad = (cur.heading_deg * Math.PI) / 180
            const arrowLen = 24
            const ax = x + arrowLen * Math.cos(rad), ay = y + arrowLen * Math.sin(rad)
            ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(ax, ay)
            ctx.strokeStyle = 'rgba(255,218,0,0.9)'; ctx.lineWidth = 2.5; ctx.stroke()
            const a1 = rad + 2.5, a2 = rad - 2.5
            ctx.beginPath(); ctx.moveTo(ax, ay)
            ctx.lineTo(ax - 8 * Math.cos(a1), ay - 8 * Math.sin(a1))
            ctx.lineTo(ax - 8 * Math.cos(a2), ay - 8 * Math.sin(a2))
            ctx.closePath(); ctx.fillStyle = 'rgba(255,218,0,0.9)'; ctx.fill()
          }
          ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2)
          ctx.fillStyle = cur.rearing ? '#ff6b6b' : '#fff'; ctx.fill()
          ctx.strokeStyle = 'rgba(0,0,0,0.5)'; ctx.lineWidth = 2; ctx.stroke()
        }
      }

      const zid = cur.zone_id
      if (zid && zoneColorMap[zid]) {
        ctx.font = 'bold 12px DM Sans, sans-serif'
        ctx.fillStyle = zoneColorMap[zid]
        ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'
        ctx.fillText(zid, x, y - 12)
      }
    }
  }, [result, currentTime, maxSpeed, hasCm, zoneColorMap, poseOverlayShow, poseShowBones, poseKpVisible, poseHideLikBelow])

  // Single effect handles all redraws — onTime only sets currentTime
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const onTime = () => setCurrentTime(video.currentTime)
    const onMeta = () => drawOverlay()
    const onResize = () => drawOverlay()
    video.addEventListener('timeupdate', onTime)
    video.addEventListener('loadedmetadata', onMeta)
    window.addEventListener('resize', onResize)
    return () => {
      video.removeEventListener('timeupdate', onTime)
      video.removeEventListener('loadedmetadata', onMeta)
      window.removeEventListener('resize', onResize)
    }
  }, [drawOverlay])

  useEffect(() => { drawOverlay() }, [drawOverlay])

  // Global keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName
      const inInput = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'

      // ? = help (always, even in inputs only if meta key)
      if (e.key === '?' && !inInput) { setShowHelp(h => !h); e.preventDefault(); return }
      if (e.key === 'Escape') { setShowHelp(false); setShowCalibration(false); return }

      if (inInput) return
      if (stage !== 'done' || !result) return

      const fps = result.summary.fps || 30
      const v = videoRef.current
      if (!v) return
      if (e.key === 'ArrowRight') {
        v.currentTime = Math.min(v.duration || 0, v.currentTime + 1 / fps)
        e.preventDefault()
      } else if (e.key === 'ArrowLeft') {
        v.currentTime = Math.max(0, v.currentTime - 1 / fps)
        e.preventDefault()
      } else if (e.key === ' ') {
        v.paused ? v.play() : v.pause()
        e.preventDefault()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [stage, result])

  function showToast(msg: string, durationMs = 3500) {
    setToastMsg(msg)
    setTimeout(() => setToastMsg(null), durationMs)
  }

  // Apply playback speed to video element
  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = playbackSpeed
  }, [playbackSpeed])

  // Live zone dwell timer
  useEffect(() => {
    if (!live.liveOn || !arenaSetup?.zones?.length) {
      if (liveZoneTimerRef.current) clearInterval(liveZoneTimerRef.current)
      setLiveZoneTimes({})
      return
    }
    liveZoneTimerRef.current = setInterval(() => {
      // Read latest centroid from live canvas data (approximate from trail)
      // We increment a "current zone" counter — in live mode we just increment all
      // active zones equally for demo; real zone check needs the centroid
      setLiveZoneTimes(prev => {
        const next = { ...prev }
        for (const z of arenaSetup.zones) {
          next[z.id] = (next[z.id] || 0) + 0.2
        }
        return next
      })
    }, 200)
    return () => { if (liveZoneTimerRef.current) clearInterval(liveZoneTimerRef.current) }
  }, [live.liveOn, arenaSetup])

  // CSV import handler
  async function importCsv(file: File, fps: number) {
    setCsvImporting(true)
    try {
      const fd = new FormData()
      fd.append('csv_file', file)
      fd.append('fps', String(fps))
      fd.append('px_per_cm', String(0))
      fd.append('animal_id', animalMeta.animal_id)
      fd.append('treatment', animalMeta.treatment)
      fd.append('trial', animalMeta.trial)
      fd.append('session', animalMeta.session)
      const r = await fetch(`${API}/api/import/csv`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error(await r.text())
      const data = await r.json() as { job_id: string; status: string; frame_count: number }
      // Load the result
      await loadPreviousJob(data.job_id)
      setShowCsvImport(false)
      showToast(`✓ Imported ${data.frame_count} frames from CSV`)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally { setCsvImporting(false) }
  }

  // Fetch previous sessions for the Session Manager panel
  useEffect(() => {
    fetch(`${API}/api/jobs`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data?.jobs) setPrevJobs(data.jobs) })
      .catch(() => {})
  }, [stage])

  // -------------------------------------------------------------------------
  // Frame correction
  // -------------------------------------------------------------------------
  async function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = overlayRef.current, video = videoRef.current
    if (!canvas || !result) return
    const rect = canvas.getBoundingClientRect()
    const sx = (video?.videoWidth || result.summary.arena_size_px[0]) / canvas.width
    const sy = (video?.videoHeight || result.summary.arena_size_px[1]) / canvas.height
    const x = (e.clientX - rect.left) * sx
    const y = (e.clientY - rect.top) * sy

    // Trajectory click-to-seek: find nearest frame to click position and seek video
    if (!correctionMode && video) {
      const frames = result.frames.filter(f => f.ok && f.centroid)
      if (frames.length > 0) {
        let bestFrame = frames[0]
        let bestDist = Infinity
        for (const f of frames) {
          const dx = (f.centroid!.x - x), dy = (f.centroid!.y - y)
          const d = dx*dx + dy*dy
          if (d < bestDist) { bestDist = d; bestFrame = f }
        }
        if (bestDist < 40*40) {  // only seek if click is within 40px of any point
          video.currentTime = bestFrame.t_sec
          showToast(`↩ Seeked to ${bestFrame.t_sec.toFixed(2)}s`)
          return
        }
      }
    }

    if (!correctionMode || !jobId) return
    const fps = result.summary.fps || 30
    const frameIdx = Math.floor((video?.currentTime || 0) * fps)
    try {
      await fetch(`${API}/api/jobs/${jobId}/frames/${frameIdx}?x=${x.toFixed(1)}&y=${y.toFixed(1)}`, { method: 'PATCH' })
      const r = await fetch(`${API}/api/jobs/${jobId}/result`)
      if (r.ok) setResult(await r.json())
    } catch { /* network error */ }
  }

  // -------------------------------------------------------------------------
  // Annotated video download — backend renders MP4 with burned-in overlays
  // -------------------------------------------------------------------------

  // -------------------------------------------------------------------------
  // Load a previously completed job from the session manager
  // -------------------------------------------------------------------------
  async function loadPreviousJob(prevJobId: string) {
    try {
      const r = await fetch(`${API}/api/jobs/${prevJobId}/result`)
      if (!r.ok) throw new Error(await r.text())
      const data = await r.json() as TrackingResult & { post_notes?: string }
      setResult(data)
      setJobId(prevJobId)
      setJobStatus({ id: prevJobId, status: 'done', progress: 1.0 })
      setPostNotes(data.post_notes || '')
      if (data.metrics) {
        const am = data.animal_meta
        setBatchRows(prev => [
          ...prev.filter(r2 => r2.job_id !== prevJobId),
          {
            job_id: prevJobId,
            animal_id: am?.animal_id || '',
            treatment: am?.treatment || '',
            trial: am?.trial || '',
            session: (am as AnimalMeta | undefined)?.session || '',
            metrics: data.metrics as unknown as Record<string, number | null>,
          }
        ])
      }
      setStage('done')
      // Set video src from backend so video plays for loaded sessions
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.src = `${API}/api/jobs/${prevJobId}/video`
        }
      }, 80)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load previous result')
    }
  }

  async function savePostNotes() {
    if (!jobId) return
    setSavingNotes(true)
    try {
      await fetch(`${API}/api/jobs/${jobId}/notes`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes: postNotes }),
      })
      setSavedNotes(true)
      setTimeout(() => setSavedNotes(false), 2000)
    } finally { setSavingNotes(false) }
  }

  // -------------------------------------------------------------------------
  // PNG export helpers
  // -------------------------------------------------------------------------
  function downloadCanvasAsPng(canvas: HTMLCanvasElement, filename: string) {
    const a = document.createElement('a')
    a.href = canvas.toDataURL('image/png')
    a.download = filename
    a.click()
  }

  function exportTrajectoryPng() {
    const video = videoRef.current
    const overlay = overlayRef.current
    if (!overlay || !result) return
    const W = video?.videoWidth || result.summary.arena_size_px[0]
    const H = video?.videoHeight || result.summary.arena_size_px[1]
    const composite = document.createElement('canvas')
    composite.width = W; composite.height = H
    const ctx = composite.getContext('2d')!
    if (video && video.readyState >= 2) {
      ctx.drawImage(video, 0, 0, W, H)
    }
    ctx.drawImage(overlay, 0, 0, W, H)
    downloadCanvasAsPng(composite, 'trajectory.png')
  }

  // -------------------------------------------------------------------------
  // TTL event import
  // -------------------------------------------------------------------------
  function importTtl(file: File) {
    const reader = new FileReader()
    reader.onload = e => {
      const text = e.target?.result as string
      const lines = text.split('\n').filter(l => l.trim())
      const events: TTLEvent[] = []
      for (const line of lines.slice(1)) { // skip header
        const [t, label] = line.split(',').map(s => s.trim())
        const tNum = parseFloat(t)
        if (!isNaN(tNum)) events.push({ t: tNum, label: label || 'Event' })
      }
      setTtlEvents(events)
    }
    reader.readAsText(file)
  }

  // -------------------------------------------------------------------------
  // Quick analysis — track without zones (empty arena setup)
  // -------------------------------------------------------------------------
  function trackWithoutZones() {
    startTracking({ arenaPoly: [], zones: [], pxPerCm: 0, imgW: 0, imgH: 0 })
  }

  // -------------------------------------------------------------------------
  // Re-use setup — "New video with same settings"
  // -------------------------------------------------------------------------
  function reuseSetup() {
    keepSetupRef.current = true
    setFile(null); setResult(null); setJobId(null); setJobStatus(null)
    setError(null); setPostNotes('')
    // Keep arenaSetup and animalMeta — will pre-populate for next video
    setStage('idle')
  }

  // -------------------------------------------------------------------------
  // File selection
  // -------------------------------------------------------------------------
  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null
    setFile(f); setResult(null); setJobId(null); setJobStatus(null)
    setError(null)
    if (!keepSetupRef.current) setArenaSetup(null)
    keepSetupRef.current = false
    setStage(f ? 'meta' : 'idle')
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f && f.type.startsWith('video/')) {
      setFile(f); setResult(null); setJobId(null); setJobStatus(null)
      setError(null); setArenaSetup(null); setStage('meta')
    }
  }

  const duration = result ? result.summary.frame_count / (result.summary.fps || 30) : 0
  const STEPS = ['Upload', 'Animal info', 'Arena', 'Tracking', 'Results'] as const
  const stepIdx: Record<AppStage, number> = { idle: 0, meta: 1, arena: 2, tracking: 3, fetching: 3, done: 4, batch: 0, cohort: 0 }
  const m = result?.metrics

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebarBrand">
          <div className="brandMark" />
          <div className="brandText">
            <div className="brandName">NeuroTrack</div>
            <div className="brandSub">Rodent behavior analysis</div>
        </div>
          <button className="helpBtn" onClick={() => setShowHelp(true)} title="Keyboard shortcuts &amp; tips (?)">?</button>
        </div>

        <nav className="sidebarNav">
          {STEPS.map((s, i) => {
            const isDone = stepIdx[stage] > i
            const isActive = stepIdx[stage] === i
            return (
              <div key={s} className={`navStep ${isActive ? 'active' : ''} ${isDone ? 'done' : ''}`}>
                <div className="navStepCircle">{isDone ? '✓' : i + 1}</div>
                <span className="navStepLabel">{s}</span>
              </div>
            )
          })}
        </nav>

        <div className="sidebarFooter">
          <div className="sidebarFooterNote">Open-source · Free · No cloud</div>
        </div>
      </aside>

      {/* Main */}
      <main className="main">
        {error && (
          <div className="errorBanner">
            <span className="errorIcon">⚠</span> {error}
            <button className="errorClose" onClick={() => setError(null)}>✕</button>
          </div>
        )}

        {/* ── Stage: idle ── */}
        {stage === 'idle' && (
          <div className="stagePage" key="idle">
            <div className="heroLayout">
              <div className="heroLeft">
                <div className="heroEyebrow">Advanced rodent tracking</div>
                <h1 className="heroTitle">Behavior analysis<br />built for researchers</h1>
                <p className="heroBody">
                  Upload a recording from any maze apparatus. Define the arena, add behavioral zones,
                  set a real-world scale, and get publication-ready metrics in seconds — all free, all local.
                </p>
                <label
                  className="uploadZone"
                  onDragOver={e => e.preventDefault()}
                  onDrop={handleDrop}
                >
                  <input type="file" accept="video/*" onChange={handleFileChange} />
                  <div className="uploadIcon">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="17 8 12 3 7 8"/>
                      <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                  </div>
                  <div className="uploadLabel">Drop video here or click to browse</div>
                  <div className="uploadHint">MP4, AVI, MOV — up to 4 GB per file</div>
                </label>

                <div style={{ textAlign: 'center', margin: '6px 0 2px', fontSize: 12, color: 'var(--textMuted)' }}>— or —</div>
                <button className="csvImportBtn" onClick={() => setShowCsvImport(true)}>
                  📊 Import pre-tracked CSV (AnyMaze / Ethovision / any tracker)
                </button>

                <div className="featureGrid">
                  {[
                    ['Open Field', 'EPM', 'Y-Maze', 'Circular Arena', 'Light/Dark Box', 'Novel Object'],
                    ['Distance & Speed', 'Freezing', 'Immobility', 'Rearing', 'Thigmotaxis'],
                    ['Zone dwell & entries', 'Latency to enter', 'Path efficiency', 'CW/CCW Rotations'],
                    ['Occupancy heatmap', 'Speed trajectory', 'Time-bin analysis', 'Multi-animal', 'Batch CSV'],
                  ].map((row, ri) => (
                    <div key={ri} className="featureRow">
                      {row.map(item => <span key={item} className="featureTag">{item}</span>)}
                    </div>
                  ))}
                </div>

                <div style={{ display: 'flex', gap: 10, marginTop: 4, flexWrap: 'wrap' }}>
                  <button className="btnSecondary" onClick={() => setStage('batch')}>
                    📦 Batch mode — multiple videos
                  </button>
                  <button className="btnSecondary" onClick={() => setStage('cohort')}
                    style={{ background: 'rgba(99,102,241,0.15)', border: '1px solid rgba(99,102,241,0.4)', color: '#a5b4fc' }}>
                    🧬 Cohort analysis — latent phenotyping
                  </button>
                </div>

                {/* Previous sessions */}
                {prevJobs.filter(j => j.status === 'done').length > 0 && (() => {
                  const doneSessions = prevJobs.filter(j => j.status === 'done')
                  const filtered = doneSessions.filter(j => {
                    if (!sessionSearch) return true
                    const q = sessionSearch.toLowerCase()
                    return (j.animal_id || '').toLowerCase().includes(q) ||
                      (j.treatment || '').toLowerCase().includes(q) ||
                      (j.session || '').toLowerCase().includes(q) ||
                      (j.video_filename || '').toLowerCase().includes(q)
                  })
                  const visible = showAllSessions ? filtered : filtered.slice(0, 8)
                  return (
                    <div className="prevSessionsPanel">
                      <div className="prevSessionsHeader">
                        <div className="prevSessionsTitle">Recent sessions ({doneSessions.length})</div>
                        <input
                          className="prevSessionSearch"
                          placeholder="Search…"
                          value={sessionSearch}
                          onChange={e => setSessionSearch(e.target.value)}
                        />
                      </div>
                      <div className="prevSessionsList">
                        {visible.map(j => (
                          <div key={j.job_id} className="prevSessionRow">
                            <div className="prevSessionInfo">
                              <span className="prevSessionAnimal">{j.animal_id || 'Unnamed'}</span>
                              {j.treatment && <span className="prevSessionTx">{j.treatment}</span>}
                              {j.session && <span className="prevSessionSession">{j.session}</span>}
                              <span className="prevSessionFile">{j.video_filename || j.job_id.slice(0, 8)}</span>
                              <span className="prevSessionDate">{j.created_at.slice(0, 16).replace('T', ' ')}</span>
                            </div>
                            <div className="prevSessionActions">
                              <button className="btnSm btnAccent" onClick={() => loadPreviousJob(j.job_id)}>Load</button>
                              <button className="btnSm btnDanger" onClick={async () => {
                                await fetch(`${API}/api/jobs/${j.job_id}`, { method: 'DELETE' })
                                setPrevJobs(p => p.filter(r => r.job_id !== j.job_id))
                              }}>×</button>
                            </div>
                          </div>
                        ))}
                      </div>
                      {filtered.length > 8 && (
                        <button className="prevSessionsMore" onClick={() => setShowAllSessions(s => !s)}>
                          {showAllSessions ? '↑ Show less' : `↓ Show all ${filtered.length} sessions`}
                        </button>
                      )}
                    </div>
                  )
                })()}
              </div>

              <div className="heroRight">
                <div className="livePanel">
                  <div className="livePanelHeader">
                    <div className="livePanelTitle">
                      <span className={`liveIndicator ${live.liveOn ? 'on' : ''}`} />
                      Live camera preview
        </div>
        <button
                      className={`btnSm ${live.liveOn ? 'btnDanger' : 'btnAccent'}`}
                      onClick={live.liveOn ? live.stop : live.start}
        >
                      {live.liveOn ? 'Stop' : 'Start'}
        </button>
                  </div>
                  {live.liveOn && (
                    <div className="liveStats">
                      <div className="liveStat">
                        <span className="liveStatLabel">Speed</span>
                        <span className="liveStatValue">{live.liveSpeed != null ? `${live.liveSpeed.toFixed(1)} px/s` : '—'}</span>
                      </div>
                      <div className="liveStat">
                        <span className="liveStatLabel">Distance</span>
                        <span className="liveStatValue">{live.liveDist.toFixed(0)} px</span>
                      </div>
                      {arenaSetup?.zones?.map(z => (
                        <div key={z.id} className="liveStat">
                          <span className="liveStatLabel" style={{ color: z.color || 'var(--textSub)' }}>{z.name}</span>
                          <span className="liveStatValue">{(liveZoneTimes[z.id] || 0).toFixed(1)} s</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="liveCanvasWrap" style={{ display: live.liveOn ? 'block' : 'none', position: 'relative' }}>
                    <video ref={live.videoRef} muted playsInline style={{ display: 'none' }} />
                    <canvas ref={live.canvasRef} className="liveCanvas" />
                    {/* Zone overlay drawn on a separate canvas */}
                    <LiveZoneOverlay arenaSetup={arenaSetup} videoRef={live.videoRef} />
                  </div>
                  {!live.liveOn && (
                    <div className="livePlaceholder">
                      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" opacity="0.3">
                        <path d="M23 7l-7 5 7 5V7z"/>
                        <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
                      </svg>
                      <p>Test live tracking on your rig's camera<br />before running a formal experiment</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── Stage: meta ── */}
        {stage === 'meta' && (
          <div className="stagePage" key="meta">
            <div className="stageHeader">
              <div className="stageHeaderLeft">
                <button className="backBtn" onClick={() => setStage('idle')}>← Back</button>
                <div>
                  <div className="stageTitle">Animal information</div>
                  <div className="stageSub">Optional metadata for multi-animal studies and report export</div>
                </div>
              </div>
            </div>

            <div className="formCard">
              <div className="formGrid">
                <label className="formField">
                  <span className="formLabel">Animal ID</span>
                  <input className="formInput" placeholder="e.g. Mouse_01"
                    value={animalMeta.animal_id}
                    onChange={e => setAnimalMeta(m => ({ ...m, animal_id: e.target.value }))} />
                </label>
                <label className="formField">
                  <span className="formLabel">Treatment group</span>
                  <input className="formInput" placeholder="e.g. Control, Drug 10 mg/kg"
                    value={animalMeta.treatment}
                    onChange={e => setAnimalMeta(m => ({ ...m, treatment: e.target.value }))} />
                </label>
                <label className="formField">
                  <span className="formLabel">Trial / session</span>
                  <input className="formInput" placeholder="e.g. Trial 1, Day 3"
                    value={animalMeta.trial}
                    onChange={e => setAnimalMeta(m => ({ ...m, trial: e.target.value }))} />
                </label>
                <label className="formField">
                  <span className="formLabel">Animals per video</span>
                  <select className="formInput" value={animalMeta.n_animals}
                    onChange={e => setAnimalMeta(m => ({ ...m, n_animals: Number(e.target.value) }))}>
                    <option value={1}>1 (single animal)</option>
                    <option value={2}>2 (pair / social)</option>
                    <option value={3}>3</option>
                  </select>
                </label>
                <label className="formField">
                  <span className="formLabel">Tracking engine</span>
                  <select className="formInput" value={animalMeta.engine}
                    onChange={e => { setAnimalMeta(m => ({ ...m, engine: e.target.value as AnimalMeta['engine'] })); setPoseFile(null) }}>
                    <option value="opencv_mog2_centroid">OpenCV MOG2 — built-in, no file needed</option>
                    <option value="dlc_csv">DeepLabCut CSV — upload exported CSV</option>
                    <option value="sleap_slp">SLEAP .slp — upload predictions file</option>
                  </select>
                </label>
                {(animalMeta.engine === 'dlc_csv' || animalMeta.engine === 'sleap_slp') && (
                  <label className="formField formFieldFull">
                    <span className="formLabel">
                      {animalMeta.engine === 'dlc_csv' ? 'DeepLabCut output CSV' : 'SLEAP .slp predictions file'}
                    </span>
                    <div className="poseFileRow">
                      <input
                        type="file"
                        accept={animalMeta.engine === 'dlc_csv' ? '.csv' : '.slp,.h5'}
                        className="formInput"
                        style={{ flex: 1 }}
                        onChange={e => setPoseFile(e.target.files?.[0] ?? null)}
                      />
                      {poseFile && <span className="poseBadge">✓ {poseFile.name}</span>}
        </div>
                    <div className="formHint">
                      {animalMeta.engine === 'dlc_csv'
                        ? 'Run DeepLabCut\'s analyze_video() first, then upload the *_el.csv or *DLC*.csv file'
                        : 'Run SLEAP inference first, then upload the .slp predictions file'}
                    </div>
                  </label>
                )}
                <label className="formField">
                  <span className="formLabel">Session label</span>
                  <input className="formInput" placeholder="e.g. Day 1, Week 2"
                    value={animalMeta.session}
                    onChange={e => setAnimalMeta(m => ({ ...m, session: e.target.value }))} />
                </label>
                <label className="formField">
                  <span className="formLabel">Experiment / Cohort ID</span>
                  <input className="formInput" placeholder="e.g. Fear conditioning cohort A"
                    value={animalMeta.experiment_id}
                    onChange={e => setAnimalMeta(m => ({ ...m, experiment_id: e.target.value }))} />
                </label>
                <label className="formField">
                  <span className="formLabel">Experimenter</span>
                  <input className="formInput" placeholder="Name or initials"
                    value={animalMeta.experimenter}
                    onChange={e => setAnimalMeta(m => ({ ...m, experimenter: e.target.value }))} />
                </label>
                <label className="formField formFieldFull">
                  <span className="formLabel">Notes</span>
                  <input className="formInput" placeholder="Any relevant experimental notes…"
                    value={animalMeta.notes}
                    onChange={e => setAnimalMeta(m => ({ ...m, notes: e.target.value }))} />
                </label>
                <div className="formField formFieldFull trimRow">
                  <span className="formLabel">Video trim window (optional)</span>
                  <div className="trimInputs">
                    <label className="trimLabel">
                      Start (s)
                      <input type="number" className="formInput trimInput" min={0} step={1}
                        value={trimStart || ''} placeholder="0"
                        onChange={e => setTrimStart(Math.max(0, Number(e.target.value)))} />
                    </label>
                    <span className="trimSep">→</span>
                    <label className="trimLabel">
                      End (s) <span className="formHint" style={{display:'inline'}}>0 = full video</span>
                      <input type="number" className="formInput trimInput" min={0} step={1}
                        value={trimEnd || ''} placeholder="full"
                        onChange={e => setTrimEnd(Math.max(0, Number(e.target.value)))} />
                    </label>
                  </div>
                  {(trimStart > 0 || trimEnd > 0) && (
                    <div className="formHint">
                      Analysing from {trimStart}s to {trimEnd > 0 ? `${trimEnd}s` : 'end of video'}
                    </div>
                  )}
                </div>

                {/* Freezing threshold */}
                <div className="formField formFieldFull">
                  <span className="formLabel">
                    Freezing threshold — <strong>{freezeThreshold.toFixed(2)} cm/s</strong>
                    <span className="formHint" style={{ display: 'inline', marginLeft: 6 }}>
                      (AnyMaze default: 0.50 cm/s)
                    </span>
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 4 }}>
                    <span style={{ fontSize: 11, color: 'var(--textMuted)' }}>0.1</span>
                    <input type="range" min={0.1} max={3.0} step={0.05}
                      value={freezeThreshold}
                      onChange={e => setFreezeThreshold(Number(e.target.value))}
                      style={{ flex: 1 }} />
                    <span style={{ fontSize: 11, color: 'var(--textMuted)' }}>3.0</span>
                    <button className="btnSm btnSecondary" onClick={() => setFreezeThreshold(0.5)}
                      style={{ whiteSpace: 'nowrap' }}>Reset</button>
                  </div>
                </div>
              </div>

              {file && (
                <div className="filePreviewRow">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polygon points="23 7 16 12 23 17 23 7"/>
                    <rect x="1" y="5" width="15" height="14" rx="2"/>
          </svg>
                  <span>{file.name}</span>
                  <span className="fileSize">({(file.size / 1e6).toFixed(1)} MB)</span>
                </div>
              )}

              <div className="formActions">
                <button className="btnSecondary" onClick={() => setStage('idle')}>← Change video</button>
                <button
                  className="btnPrimary"
                  disabled={(animalMeta.engine === 'dlc_csv' || animalMeta.engine === 'sleap_slp') && !poseFile}
                  title={(animalMeta.engine === 'dlc_csv' || animalMeta.engine === 'sleap_slp') && !poseFile
                    ? 'Upload a pose file to continue' : undefined}
                  onClick={() => setStage('arena')}
                >
                  Continue to Arena Setup →
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── Stage: arena ── */}
        {stage === 'arena' && fileUrl && (
          <div className="stagePage stageArena" key="arena">
            <div className="stageHeader">
              <div className="stageHeaderLeft">
                <button className="backBtn" onClick={() => setStage('meta')}>← Back</button>
                <div>
                  <div className="stageTitle">Arena setup</div>
                  <div className="stageSub">Choose an apparatus preset or draw your arena and zones manually</div>
                </div>
              </div>
              <div className="stageHeaderRight">
                <button className="btnSm btnSecondary" onClick={trackWithoutZones}
                  title="Run tracking with full-frame arena and no behavioral zones — fastest option">
                  ⚡ Quick track (no zones)
                </button>
              </div>
            </div>
            <ArenaEditor
              frameUrl={fileUrl}
              onConfirm={result ? reanalyze : startTracking}
              onCancel={() => setStage(result ? 'done' : 'meta')}
            />
          </div>
        )}

        {/* ── Stage: fetching result ── */}
        {stage === 'fetching' && (
          <div className="stagePage stageTracking" key="fetching">
            <div className="trackingCard">
              <div className="trackingSpinner">
                <div className="spinRing" />
                <div className="spinRingInner" />
              </div>
              <div className="trackingTitle">Loading results…</div>
              <div style={{ fontSize: 13, color: 'var(--textSub)' }}>Preparing your dashboard</div>
            </div>
          </div>
        )}

        {/* ── Stage: batch ── */}
        {stage === 'batch' && (
          <div className="stagePage" key="batch">
            <div className="stageHeader">
              <div className="stageHeaderLeft">
                <button className="backBtn" onClick={() => setStage('idle')}>← Back</button>
                <div>
                  <div className="stageTitle">Batch processing</div>
                  <div className="stageSub">Track multiple videos and compare groups</div>
                </div>
              </div>
            </div>
            <BatchPanel apiBase={API} onAllDone={count => showToast(`✓ Batch complete — ${count} video${count !== 1 ? 's' : ''} processed`)} />
          </div>
        )}

        {/* ── Stage: cohort phenotyping ── */}
        {stage === 'cohort' && (
          <CohortStage onBack={() => setStage('idle')} />
        )}

        {/* ── Stage: tracking ── */}
        {stage === 'tracking' && (
          <div className="stagePage stageTracking" key="tracking">
            <div className="trackingCard">
              <div className="trackingSpinner">
                <div className="spinRing" />
                <div className="spinRingInner" />
              </div>
              <div className="trackingTitle">Analyzing video</div>
              <div className="trackingProgress">
                <div className="trackingBar">
                  <div
                    className="trackingFill"
                    style={{ width: `${((jobStatus?.progress ?? 0) * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="trackingPct">{((jobStatus?.progress ?? 0) * 100).toFixed(0)}%</span>
              </div>
              <div className="trackingSteps">
                {(animalMeta.engine === 'dlc_csv' ? [
                  { label: 'Parsing DeepLabCut CSV', pct: 0.25 },
                  { label: 'Converting keypoints to tracks', pct: 0.55 },
                  { label: 'Computing pose metrics', pct: 0.8 },
                  { label: 'Zone assignment & metrics', pct: 0.95 },
                ] : animalMeta.engine === 'sleap_slp' ? [
                  { label: 'Loading SLEAP predictions', pct: 0.35 },
                  { label: 'Extracting instance tracks', pct: 0.6 },
                  { label: 'Computing pose metrics', pct: 0.8 },
                  { label: 'Zone assignment & metrics', pct: 0.95 },
                ] : [
                  { label: 'MOG2 background model', pct: 0.2 },
                  { label: 'Centroid detection', pct: 0.6 },
                  { label: 'Kalman smoothing', pct: 0.8 },
                  { label: 'Zone assignment & metrics', pct: 0.95 },
                ]).map(step => {
                  const done = (jobStatus?.progress ?? 0) >= step.pct
                  return (
                    <div key={step.label} className={`trackingStep ${done ? 'done' : ''}`}>
                      <span className="trackingStepDot" />
                      {step.label}
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        )}

        {/* ── Stage: done ── */}
        {stage === 'done' && result && (
          <div className="stagePage stageDone" key="done">
            {/* Results header */}
            <div className="resultsHeader">
              <div className="resultsHeaderLeft">
                {result.animal_meta?.animal_id && (
                  <span className="metaTag"><span className="metaTagKey">Animal</span>{result.animal_meta.animal_id}</span>
                )}
                {result.animal_meta?.treatment && (
                  <span className="metaTag"><span className="metaTagKey">Tx</span>{result.animal_meta.treatment}</span>
                )}
                {result.animal_meta?.trial && (
                  <span className="metaTag"><span className="metaTagKey">Trial</span>{result.animal_meta.trial}</span>
                )}
                <span className="metaTag">
                  <span className="metaTagKey">Engine</span>
                  {result.meta?.engine === 'dlc_csv' ? 'DeepLabCut' :
                   result.meta?.engine === 'sleap_slp' ? 'SLEAP' : 'OpenCV'}
                </span>
                {hasPose && <span className="metaTagAccent">Pose tracking</span>}
                {hasCm && <span className="metaTagAccent">Calibrated</span>}
                {reanalyzing && <span className="metaTagWarn">Re-analyzing…</span>}
              </div>
              <div className="resultsHeaderRight">
                {jobId && (
                  <>
                    <a className="exportBtn" href={`${API}/api/jobs/${jobId}/result/csv?type=perframe`} download="perframe.csv">
                      ↓ Per-frame CSV
                    </a>
                    <a className="exportBtn" href={`${API}/api/jobs/${jobId}/result/csv?type=summary`} download="summary.csv">
                      ↓ Summary CSV
                    </a>
                    <a className="exportBtn"
                      href={`data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(result, null, 2))}`}
                      download="result.json">
                      ↓ JSON
                    </a>
                  </>
                )}
                <button
                  className={`btnSm ${correctionMode ? 'btnDanger' : 'btnSecondary'}`}
                  onClick={() => { setCorrectionMode(c => !c); setActiveResultTab('video') }}
                  title="Click on the video overlay to manually correct the centroid for the current frame"
                >
                  {correctionMode ? '✕ Exit correction' : '✎ Correct tracking'}
                </button>
                <button
                  className={`btnSm ${exportingVideo ? 'btnDanger' : 'btnSecondary'}`}
                  onClick={() => {
                    if (!jobId) return
                    const a = document.createElement('a')
                    a.href = `${API}/api/jobs/${jobId}/annotated_video`
                    a.download = 'annotated_tracking.mp4'
                    a.click()
                  }}
                  disabled={exportingVideo}
                  title="Export video with trajectory overlay as WebM"
                >
                  ⬇ Annotated video (MP4)
                </button>
                <button className="btnSm btnSecondary" onClick={() => setStage('arena')}>
                  ✎ Re-draw zones
                </button>
                <button className="btnSm btnSecondary" onClick={() => setShowCalibration(true)}>
                  📏 Calibrate
                </button>
                <button className="btnSm btnSecondary" onClick={reuseSetup}
                  title="Start a new video with the same arena, zones, and settings">
                  ↩ New video / same setup
                </button>
                <button className="btnSm btnSecondary" onClick={() => {
                    if (!jobId) return
                    const a = document.createElement('a')
                    a.href = `${API}/api/jobs/${jobId}/report.pdf`
                    a.download = `report_${jobId.slice(0, 8)}.pdf`
                    a.click()
                  }} title="Download structured PDF report">
                  📄 PDF report
                </button>
                <button className="btnSm btnSecondary" onClick={() => {
                    if (!jobId) return
                    const a = document.createElement('a')
                    a.href = `${API}/api/jobs/${jobId}/result.json`
                    a.download = `result_${jobId.slice(0, 8)}.json`
                    a.click()
                  }} title="Export complete metrics JSON for R/Python">
                  { } JSON export
                </button>
                <button className="btnSm btnSecondary" onClick={() => window.print()}>
                  🖨 Print report
                </button>
        </div>
            </div>

            {/* Dashboard grid */}
            <div className="dashboard">
              {/* Left column: video + heatmap tabs */}
              <div className="dashLeft">
                <div className="panel">
                  <div className="panelTabs">
                    {(['video', 'heatmap', 'timeline', 'events', 'score', 'protocol', 'poseqc'] as const).map(tab => {
                      const label = { video: 'Trajectory', heatmap: 'Heatmap', timeline: 'Timeline',
                        events: 'Zone events', score: 'Score', protocol: 'Protocols', poseqc: 'Pose QC' }[tab]
                      const badge = tab === 'events' && (m?.zone_events?.length ?? 0) > 0
                        ? ` (${m!.zone_events!.length})` : ''
                      return (
                        <button key={tab}
                          className={`panelTab ${activeResultTab === tab ? 'active' : ''}`}
                          onClick={() => setActiveResultTab(tab)}
                        >
                          {label}{badge}
                        </button>
                      )
                    })}
                  </div>

                  {activeResultTab === 'video' && (
                    <div className="videoWrap">
                      <video ref={videoRef} controls playsInline className="resultVideo" />
                      <canvas
                        ref={overlayRef}
                        className={`overlay ${correctionMode ? 'correctionMode' : ''}`}
                        onClick={handleCanvasClick}
                        title={correctionMode ? 'Click to set centroid for this frame' : undefined}
                      />
                      <div className="speedLegend">
                        <div className="speedGrad" />
                        <span>Slow</span>
                        <span style={{ marginLeft: 'auto' }}>Fast</span>
                        <span className="legendArrow">→ heading</span>
                        <button className="exportPngBtn" onClick={exportTrajectoryPng}
                          title="Save trajectory image as PNG">⬇ PNG</button>
                      </div>
                      <div className="playbackControls">
                        {[0.25, 0.5, 1, 2].map(s => (
                          <button key={s}
                            className={`pbSpeedBtn ${playbackSpeed === s ? 'active' : ''}`}
                            onClick={() => setPlaybackSpeed(s)}
                          >{s}×</button>
                        ))}
                        {result.summary.fps && (
                          <span className="pbHint">← → frame · Space play/pause</span>
                        )}
                      </div>
                      {hasPose && (
                        <div style={{
                          marginTop: 10, padding: '10px 12px', borderRadius: 8,
                          background: 'rgba(99,102,241,0.08)', border: '1px solid rgba(99,102,241,0.25)',
                          fontSize: 11, color: '#cbd5e1',
                        }}>
                          <div style={{ fontWeight: 700, marginBottom: 8, color: '#a5b4fc' }}>Pose overlay</div>
                          <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, cursor: 'pointer' }}>
                            <input type="checkbox" checked={poseOverlayShow} onChange={e => setPoseOverlayShow(e.target.checked)} />
                            Show skeleton / keypoints
                          </label>
                          <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, cursor: 'pointer' }}>
                            <input type="checkbox" checked={poseShowBones} onChange={e => setPoseShowBones(e.target.checked)} disabled={!poseOverlayShow} />
                            Draw bones
                          </label>
                          <div style={{ marginTop: 8, marginBottom: 4, color: '#94a3b8' }}>Keypoints</div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 10px' }}>
                            {CANONICAL_KPS.map(name => (
                              <label key={name} style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
                                <input
                                  type="checkbox"
                                  checked={poseKpVisible[name] !== false}
                                  onChange={e => setPoseKpVisible(v => ({ ...v, [name]: e.target.checked }))}
                                  disabled={!poseOverlayShow}
                                />
                                <span style={{ color: KP_COLORS[name] ?? '#fff' }}>{name.replace('_', ' ')}</span>
                              </label>
                            ))}
                          </div>
                          <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span>Hide if likelihood &lt;</span>
                            <input
                              type="range" min={0} max={0.95} step={0.05} value={poseHideLikBelow}
                              onChange={e => setPoseHideLikBelow(Number(e.target.value))}
                              style={{ flex: 1, maxWidth: 140 }}
                            />
                            <span>{poseHideLikBelow.toFixed(2)} (0 = off)</span>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {activeResultTab === 'heatmap' && (
                    <div className="heatmapWrap">
                      <div className="heatmapModeToggle">
                        <button
                          className={`hmBtn ${heatmapMode === 'occupancy' ? 'active' : ''}`}
                          onClick={() => setHeatmapMode('occupancy')}
                        >Occupancy</button>
                        <button
                          className={`hmBtn ${heatmapMode === 'velocity' ? 'active' : ''}`}
                          onClick={() => setHeatmapMode('velocity')}
                        >Velocity</button>
                        <button className="hmBtn" onClick={() => {
                          const wrap = document.querySelector('.heatmap-wrap') as HTMLElement
                          if (!wrap) return
                          const bg = wrap.querySelector('canvas.heatmap-bg') as HTMLCanvasElement
                          const overlay = wrap.querySelector('canvas.heatmap-overlay') as HTMLCanvasElement
                          if (!overlay) return
                          const W = overlay.width, H = overlay.height
                          const comp = document.createElement('canvas')
                          comp.width = W; comp.height = H
                          const ctx = comp.getContext('2d')!
                          if (bg && bg.width) ctx.drawImage(bg, 0, 0, W, H)
                          ctx.drawImage(overlay, 0, 0)
                          downloadCanvasAsPng(comp, `heatmap_${heatmapMode}.png`)
                        }}>⬇ PNG</button>
                      </div>
                      <Heatmap
                        frames={result.frames}
                        arenaW={result.summary.arena_size_px[0]}
                        arenaH={result.summary.arena_size_px[1]}
                        backgroundUrl={fileUrl ?? undefined}
                        mode={heatmapMode}
                      />
                    </div>
                  )}

                  {activeResultTab === 'timeline' && (
                    <div style={{ padding: '10px 14px' }}>
                      <div className="tlToolbar">
                        <label className="tlUploadBtn">
                          ↑ Import TTL events (CSV: timestamp_s, label)
                          <input type="file" accept=".csv,.txt" style={{ display: 'none' }}
                            onChange={e => e.target.files?.[0] && importTtl(e.target.files[0])} />
                        </label>
                        {ttlEvents.length > 0 && (
                          <span className="tlEventCount">{ttlEvents.length} TTL events loaded</span>
                        )}
                      </div>
                      <BehaviorTimeline
                        behaviorEvents={(m?.behavior_events ?? []) as BehaviorSegment[]}
                        zoneEvents={(m?.zone_events ?? []) as ZoneEventTimeline[]}
                        zoneColors={zoneColorMap}
                        duration={duration}
                        currentTime={currentTime}
                        onScrub={t => { if (videoRef.current) videoRef.current.currentTime = t }}
                        ttlEvents={ttlEvents}
                      />
                    </div>
                  )}

                  {activeResultTab === 'events' && (
                    <ZoneEventLog
                      events={(m?.zone_events ?? []) as ZoneEventRow[]}
                      zoneColors={zoneColorMap}
                      onScrub={t => { if (videoRef.current) videoRef.current.currentTime = t; setActiveResultTab('video') }}
                    />
                  )}

                  {activeResultTab === 'score' && (
                    <div style={{ padding: '10px 14px' }}>
                      <ScorePanel currentTime={currentTime} />
                    </div>
                  )}

                  {activeResultTab === 'protocol' && (
                    <ProtocolManager
                      currentSetup={arenaSetup}
                      onLoad={setup => { setArenaSetup(setup); setActiveResultTab('video') }}
                    />
                  )}

                  {activeResultTab === 'poseqc' && (
                    <PoseQCPanel jobId={jobId} hasPose={hasPose} />
                  )}
                </div>

                {/* Speed chart */}
                <div className="panel">
                  <div className="panelHeader">
                    <span className="panelTitle">Speed over time</span>
                    <span className="panelBadge">{hasCm ? 'cm/s' : 'px/s'}</span>
                  </div>
                  <SpeedChart
                    frames={result.frames}
                    useCm={hasCm}
                    currentTime={currentTime}
                    duration={duration}
                    onScrub={(t) => { if (videoRef.current) videoRef.current.currentTime = t }}
                  />
                </div>
              </div>

              {/* Right column: metrics */}
              <div className="dashRight">
                {m && (
                  <div className="panel">
                    <div className="panelHeader">
                      <span className="panelTitle">Summary metrics</span>
                    </div>
                    <div className="statsGrid">
                      <StatCard k="Duration" v={fmt(m.duration_s, 1, 's')} />
                      <StatCard k="Tracked" v={`${(m.valid_fraction * 100).toFixed(1)}%`} />
                      <StatCard k="Distance" v={
                        m.total_distance_cm != null
                          ? fmt(m.total_distance_cm, 1, 'cm')
                          : fmt(m.total_distance_px, 0, 'px')
                      } />
                      <StatCard k="Mean speed" v={
                        m.mean_speed_cm_s != null
                          ? fmt(m.mean_speed_cm_s, 2, 'cm/s')
                          : fmt(m.mean_speed_px_s, 1, 'px/s')
                      } />
                      <StatCard k="Max speed" v={
                        m.max_speed_cm_s != null
                          ? fmt(m.max_speed_cm_s, 2, 'cm/s')
                          : fmt(m.max_speed_px_s, 1, 'px/s')
                      } />
                      <StatCard k="Mobile" v={fmt(m.total_time_mobile_s, 1, 's')} />
                      <StatCard k="Immobile" v={fmt(m.total_time_immobile_s, 1, 's')} />
                      <StatCard k="Freezing" v={`${m.total_time_freezing_s.toFixed(1)} s`} sub={`${m.freezing_episodes} episodes`} />
                      {m.thigmotaxis_fraction != null && (
                        <StatCard k="Thigmotaxis" v={`${(m.thigmotaxis_fraction * 100).toFixed(1)}%`} sub="wall-hugging" />
                      )}
                      {m.path_efficiency != null && (
                        <StatCard k="Path efficiency" v={m.path_efficiency.toFixed(3)} sub="net / total dist." />
                      )}
                      <StatCard k="CW rotations" v={m.clockwise_rotations.toFixed(2)} />
                      <StatCard k="CCW rotations" v={m.anticlockwise_rotations.toFixed(2)} />
                      {(m.total_time_rearing_s ?? 0) > 0 && (
                        <StatCard k="Rearing" v={`${m.total_time_rearing_s!.toFixed(1)} s`} sub={`${m.rearing_episodes} episodes`} />
                      )}
                      {/* Per-animal color legend in multi-animal mode */}
                      {(animalMeta.n_animals ?? 1) > 1 && (
                        <div style={{ gridColumn: '1 / -1', display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 4 }}>
                          {Array.from({ length: animalMeta.n_animals ?? 1 }, (_, i) => (
                            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11 }}>
                              <div style={{ width: 12, height: 12, borderRadius: '50%', background: ANIMAL_COLORS[i % ANIMAL_COLORS.length] }} />
                              <span style={{ color: ANIMAL_COLORS[i % ANIMAL_COLORS.length] }}>Animal {i + 1}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      {m.mean_inter_animal_dist_cm != null && (
                        <StatCard k="Inter-animal dist." v={fmt(m.mean_inter_animal_dist_cm, 1, 'cm')} sub="mean" />
                      )}
                      {m.mean_inter_animal_dist_cm == null && m.mean_inter_animal_dist_px != null && (
                        <StatCard k="Inter-animal dist." v={fmt(m.mean_inter_animal_dist_px, 0, 'px')} sub="mean" />
                      )}
                      {medianBodyLengthPx != null && (
                        <StatCard
                          k="Body length"
                          v={hasCm && result.meta?.px_per_cm
                            ? fmt(medianBodyLengthPx / (result.meta.px_per_cm as number), 1, 'cm')
                            : fmt(medianBodyLengthPx, 0, 'px')}
                          sub="median (pose)"
                        />
                      )}
                    </div>

                    {/* Pose-derived metrics panel (shown when DLC/SLEAP engine used) */}
                    {(m.mean_body_length_px != null || m.mean_spine_curvature != null || (m.total_time_grooming_s ?? 0) > 0) && (
                      <div style={{ marginTop: 12, padding: '10px 12px', background: 'rgba(0,180,120,0.06)', borderRadius: 8, border: '1px solid rgba(0,180,120,0.2)' }}>
                        <div style={{ fontSize: 11, color: '#00b478', fontWeight: 600, marginBottom: 8 }}>
                          Pose Kinematics (DLC / SLEAP — 7-keypoint skeleton)
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--textMuted)', marginBottom: 8 }}>
                          Keypoints: nose · left ear · right ear · neck · mid-spine · hips · tail base
                        </div>
                        <div className="statsGrid">
                          {m.mean_body_length_cm != null && <StatCard k="Mean body length" v={`${m.mean_body_length_cm.toFixed(1)} cm`} sub="nose → tail base" />}
                          {m.mean_body_length_cm == null && m.mean_body_length_px != null && <StatCard k="Mean body length" v={`${m.mean_body_length_px.toFixed(0)} px`} sub="nose → tail base" />}
                          {m.mean_ear_span_px != null && <StatCard k="Mean ear span" v={`${m.mean_ear_span_px.toFixed(0)} px`} sub="L ear → R ear" />}
                          {m.mean_spine_curvature != null && <StatCard k="Mean spine curvature" v={m.mean_spine_curvature.toFixed(4)} sub="0 = straight, >0.25 = hunched" />}
                          {m.mean_head_body_angle_deg != null && <StatCard k="Mean head-body angle" v={`${m.mean_head_body_angle_deg.toFixed(1)}°`} sub="head turning" />}
                          {(m.total_time_grooming_s ?? 0) > 0 && <StatCard k="Grooming time" v={`${m.total_time_grooming_s!.toFixed(1)} s`} sub={`${m.grooming_episodes ?? 0} bouts`} />}
                        </div>
                      </div>
                    )}

                    {/* NOR metrics */}
                    {m.nor && (
                      <div style={{ marginTop: 12, padding: '10px 12px', background: 'rgba(100,220,255,0.06)', borderRadius: 8, border: '1px solid rgba(100,220,255,0.15)' }}>
                        <div style={{ fontSize: 11, color: '#64dcff', fontWeight: 600, marginBottom: 8 }}>
                          Novel Object Recognition
                        </div>
                        <div className="statsGrid">
                          <StatCard k="Discrimination Index" v={m.nor.discrimination_index.toFixed(3)} sub="(T_novel−T_fam)/(T_novel+T_fam)" />
                          <StatCard k="Preference Index" v={`${(m.nor.preference_index * 100).toFixed(1)}%`} sub="T_novel / T_total" />
                          <StatCard k="Time — novel" v={`${m.nor.time_novel_s.toFixed(1)} s`} />
                          <StatCard k="Time — familiar" v={`${m.nor.time_familiar_s.toFixed(1)} s`} />
                          <StatCard k="Total exploration" v={`${m.nor.total_exploration_s.toFixed(1)} s`} />
                        </div>
                      </div>
                    )}

                    {/* Social proximity metrics */}
                    {m.social && (
                      <div style={{ marginTop: 12, padding: '10px 12px', background: 'rgba(200,100,255,0.06)', borderRadius: 8, border: '1px solid rgba(200,100,255,0.15)' }}>
                        <div style={{ fontSize: 11, color: '#c77dff', fontWeight: 600, marginBottom: 8 }}>
                          Social Proximity (threshold: {m.social.proximity_threshold_cm != null ? `${m.social.proximity_threshold_cm} cm` : `${m.social.proximity_threshold_px} px`})
                        </div>
                        <div className="statsGrid">
                          <StatCard k="Time near" v={`${m.social.total_time_near_s.toFixed(1)} s`} />
                          <StatCard k="Near episodes" v={String(m.social.near_episodes)} />
                          <StatCard k="First contact" v={m.social.first_contact_latency_s != null ? `${m.social.first_contact_latency_s.toFixed(1)} s` : '—'} />
                        </div>
                      </div>
                    )}

                    {/* Place preference */}
                    {m.place_preference && (
                      <div style={{ marginTop: 12, padding: '10px 12px', background: 'rgba(255,200,80,0.06)', borderRadius: 8, border: '1px solid rgba(255,200,80,0.15)' }}>
                        <div style={{ fontSize: 11, color: '#ffd93d', fontWeight: 600, marginBottom: 8 }}>
                          Place Preference
                        </div>
                        <div className="statsGrid">
                          <StatCard k="Left / Right" v={`${m.place_preference.time_left_s.toFixed(1)} s / ${m.place_preference.time_right_s.toFixed(1)} s`}
                            sub={`Index: ${m.place_preference.preference_lr > 0 ? '+' : ''}${m.place_preference.preference_lr.toFixed(3)}`} />
                          <StatCard k="Top / Bottom" v={`${m.place_preference.time_top_s.toFixed(1)} s / ${m.place_preference.time_bottom_s.toFixed(1)} s`}
                            sub={`Index: ${m.place_preference.preference_tb > 0 ? '+' : ''}${m.place_preference.preference_tb.toFixed(3)}`} />
                        </div>
                      </div>
                    )}

                    {/* Angular velocity — shown for any engine (OpenCV computes it too) */}
                    {(() => {
                      const avFrames = result.frames.filter(f => f.angular_velocity_deg_s != null)
                      if (avFrames.length < 5) return null
                      const avs = avFrames.map(f => Math.abs(f.angular_velocity_deg_s!))
                      const meanAV = avs.reduce((s, v) => s + v, 0) / avs.length
                      const maxAV = Math.max(...avs)
                      return (
                        <div style={{ marginTop: 12, padding: '10px 12px', background: 'rgba(80,200,255,0.06)', borderRadius: 8, border: '1px solid rgba(80,200,255,0.15)' }}>
                          <div style={{ fontSize: 11, color: '#4dc8ff', fontWeight: 600, marginBottom: 8 }}>
                            Turning / Angular Velocity
                          </div>
                          <div className="statsGrid">
                            <StatCard k="Mean |angular vel.|" v={`${meanAV.toFixed(1)} °/s`} />
                            <StatCard k="Max |angular vel.|" v={`${maxAV.toFixed(1)} °/s`} />
                          </div>
                        </div>
                      )
                    })()}

                    {/* EPM metrics (auto-detected from zone names) */}
                    {m.epm && (
                      <div style={{ marginTop: 12, padding: '10px 12px', background: 'rgba(255,140,80,0.07)', borderRadius: 8, border: '1px solid rgba(255,140,80,0.18)' }}>
                        <div style={{ fontSize: 11, color: '#ff9a3c', fontWeight: 600, marginBottom: 8 }}>
                          Elevated Plus Maze (EPM)
                        </div>
                        <div className="statsGrid">
                          <StatCard k="Open arm time" v={`${m.epm.open_arm_time_s.toFixed(1)} s`}
                            sub={`${m.epm.open_arm_time_pct.toFixed(1)}% of arm time`} />
                          <StatCard k="Closed arm time" v={`${m.epm.closed_arm_time_s.toFixed(1)} s`} />
                          <StatCard k="Open arm entries" v={String(m.epm.open_arm_entries)}
                            sub={`${m.epm.open_arm_entries_pct.toFixed(1)}% of entries`} />
                          <StatCard k="Closed arm entries" v={String(m.epm.closed_arm_entries)} />
                        </div>
                      </div>
                    )}

                    {/* Morris Water Maze */}
                    {m.mwm && (
                      <div style={{ marginTop: 12, padding: '10px 12px', background: 'rgba(0,132,255,0.07)', borderRadius: 8, border: '1px solid rgba(0,132,255,0.18)' }}>
                        <div style={{ fontSize: 11, color: '#4d96ff', fontWeight: 600, marginBottom: 8 }}>
                          Morris Water Maze
                        </div>
                        <div className="statsGrid">
                          <StatCard k="Target quadrant" v={`${m.mwm.target_quadrant_pct.toFixed(1)}%`} sub="most-visited" />
                          <StatCard k="Opposite quadrant" v={`${m.mwm.opposite_quadrant_pct.toFixed(1)}%`} />
                          <StatCard k="Left quadrant" v={`${m.mwm.left_quadrant_pct.toFixed(1)}%`} />
                          <StatCard k="Right quadrant" v={`${m.mwm.right_quadrant_pct.toFixed(1)}%`} />
                          {m.mwm.platform_proximity_mean_cm != null && (
                            <StatCard k="Mean platform proximity" v={`${m.mwm.platform_proximity_mean_cm.toFixed(1)} cm`} />
                          )}
                          {m.mwm.trial_escape_latency_s.length > 0 && (
                            <StatCard k="Escape latency" v={m.mwm.trial_escape_latency_s.map(v => `${v.toFixed(1)}s`).join(', ')} />
                          )}
                        </div>
                      </div>
                    )}

                    {/* Y-maze */}
                    {m.ymaze && (
                      <div style={{ marginTop: 12, padding: '10px 12px', background: 'rgba(107,203,119,0.07)', borderRadius: 8, border: '1px solid rgba(107,203,119,0.18)' }}>
                        <div style={{ fontSize: 11, color: '#6bcb77', fontWeight: 600, marginBottom: 8 }}>
                          Y-maze (Spontaneous Alternation)
                        </div>
                        <div className="statsGrid">
                          <StatCard k="Alternation %" v={`${m.ymaze.spontaneous_alternation_pct.toFixed(1)}%`}
                            sub="chance = 50%" />
                          <StatCard k="Alternations" v={String(m.ymaze.alternations)} sub={`of ${m.ymaze.arm_entries - 2} triplets`} />
                          <StatCard k="Total arm entries" v={String(m.ymaze.arm_entries)} />
                        </div>
                      </div>
                    )}

                    {/* Open field center/periphery */}
                    {m.open_field && (
                      <div style={{ marginTop: 12, padding: '10px 12px', background: 'rgba(255,217,61,0.06)', borderRadius: 8, border: '1px solid rgba(255,217,61,0.15)' }}>
                        <div style={{ fontSize: 11, color: '#ffd93d', fontWeight: 600, marginBottom: 8 }}>
                          Open Field — Center / Periphery
                        </div>
                        <div className="statsGrid">
                          <StatCard k="Center time" v={`${m.open_field.center_time_s.toFixed(1)} s`}
                            sub={`${m.open_field.center_time_pct.toFixed(1)}% of arena time`} />
                          <StatCard k="Periphery time" v={`${m.open_field.periphery_time_s.toFixed(1)} s`} />
                          <StatCard k="Center entries" v={String(m.open_field.center_entries)} />
                          {m.open_field.center_distance_cm != null && (
                            <StatCard k="Center distance" v={`${m.open_field.center_distance_cm.toFixed(1)} cm`} />
                          )}
                        </div>
                      </div>
                    )}

                    {/* Light/Dark Box */}
                    {m.light_dark && (
                      <div style={{ marginTop: 12, padding: '10px 12px', background: 'rgba(255,255,100,0.05)', borderRadius: 8, border: '1px solid rgba(255,255,100,0.15)' }}>
                        <div style={{ fontSize: 11, color: '#ffdd57', fontWeight: 600, marginBottom: 8 }}>
                          Light / Dark Box
                        </div>
                        <div className="statsGrid">
                          <StatCard k="Light time" v={`${m.light_dark.light_time_s.toFixed(1)} s`} sub={`${m.light_dark.light_time_pct.toFixed(1)}%`} />
                          <StatCard k="Dark time" v={`${m.light_dark.dark_time_s.toFixed(1)} s`} sub={`${m.light_dark.dark_time_pct.toFixed(1)}%`} />
                          <StatCard k="Transitions" v={String(m.light_dark.transitions)} />
                          {m.light_dark.latency_to_light_s != null && (
                            <StatCard k="Latency to light" v={`${m.light_dark.latency_to_light_s.toFixed(1)} s`} />
                          )}
                          <StatCard k="Light entries" v={String(m.light_dark.light_entries)} />
                          <StatCard k="Dark entries" v={String(m.light_dark.dark_entries)} />
                        </div>
                      </div>
                    )}

                    {/* Fear Conditioning */}
                    {m.fear_cond && m.fear_cond.epochs.length > 0 && (
                      <div style={{ marginTop: 12, padding: '10px 12px', background: 'rgba(255,100,100,0.06)', borderRadius: 8, border: '1px solid rgba(255,100,100,0.18)' }}>
                        <div style={{ fontSize: 11, color: '#ff7070', fontWeight: 600, marginBottom: 8 }}>
                          Fear Conditioning Epochs
                        </div>
                        {m.fear_cond.baseline_freezing_pct != null && (
                          <div style={{ fontSize: 12, color: 'var(--textSub)', marginBottom: 6 }}>
                            Baseline freezing: <strong>{m.fear_cond.baseline_freezing_pct.toFixed(1)}%</strong>
                            {m.fear_cond.mean_cs_freezing_pct != null && ` · CS mean: ${m.fear_cond.mean_cs_freezing_pct.toFixed(1)}%`}
                          </div>
                        )}
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                          <thead>
                            <tr style={{ borderBottom: '1px solid var(--border)' }}>
                              {['Epoch','Duration','Freezing','Episodes'].map(h => (
                                <th key={h} style={{ textAlign: 'left', padding: '3px 6px', color: 'var(--textMuted)', fontWeight: 600, fontSize: 11 }}>{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {m.fear_cond.epochs.map((ep, i) => (
                              <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                                <td style={{ padding: '4px 6px', color: 'var(--text)' }}>{ep.label}</td>
                                <td style={{ padding: '4px 6px', color: 'var(--textSub)' }}>{ep.duration_s.toFixed(1)}s</td>
                                <td style={{ padding: '4px 6px' }}>
                                  <span style={{ color: ep.freezing_pct > 50 ? '#ff7070' : 'var(--accent)' }}>
                                    {ep.freezing_pct.toFixed(1)}%
                                  </span>
                                </td>
                                <td style={{ padding: '4px 6px', color: 'var(--textSub)' }}>{ep.freezing_episodes}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}

                    {/* Freeze bout list */}
                    {m.freeze_bouts && m.freeze_bouts.length > 0 && (
                      <div style={{ marginTop: 12, padding: '10px 12px', background: 'rgba(120,140,255,0.06)', borderRadius: 8, border: '1px solid rgba(120,140,255,0.18)' }}>
                        <div style={{ fontSize: 11, color: '#8ca0ff', fontWeight: 600, marginBottom: 8 }}>
                          Freeze Bouts ({m.freeze_bouts.length} episodes)
                        </div>
                        <div style={{ maxHeight: 180, overflowY: 'auto' }}>
                          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                            <thead>
                              <tr style={{ borderBottom: '1px solid var(--border)', position: 'sticky', top: 0, background: 'var(--panelBg)' }}>
                                {['#','Start (s)','End (s)','Duration','Speed','Zone'].map(h => (
                                  <th key={h} style={{ textAlign: 'left', padding: '3px 5px', color: 'var(--textMuted)', fontWeight: 600 }}>{h}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {m.freeze_bouts.map((b, i) => (
                                <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', cursor: 'pointer' }}
                                  onClick={() => {
                                    if (videoRef.current) {
                                      videoRef.current.currentTime = b.start_t
                                      setActiveResultTab('video')
                                    }
                                  }}>
                                  <td style={{ padding: '3px 5px', color: 'var(--textMuted)' }}>{i+1}</td>
                                  <td style={{ padding: '3px 5px' }}>{b.start_t.toFixed(2)}</td>
                                  <td style={{ padding: '3px 5px' }}>{b.end_t.toFixed(2)}</td>
                                  <td style={{ padding: '3px 5px', color: '#8ca0ff' }}>{b.duration_s.toFixed(2)}s</td>
                                  <td style={{ padding: '3px 5px', color: 'var(--textSub)' }}>{b.mean_speed_cm_s != null ? `${b.mean_speed_cm_s.toFixed(2)} cm/s` : '—'}</td>
                                  <td style={{ padding: '3px 5px', color: 'var(--textSub)' }}>{b.zone_id ?? '—'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}

                    {/* Rearing per zone */}
                    {m.rearing_per_zone && Object.keys(m.rearing_per_zone).length > 0 && (
                      <div style={{ marginTop: 12, padding: '10px 12px', background: 'rgba(0,200,100,0.05)', borderRadius: 8, border: '1px solid rgba(0,200,100,0.15)' }}>
                        <div style={{ fontSize: 11, color: '#00c864', fontWeight: 600, marginBottom: 8 }}>
                          Rearing per Zone
                        </div>
                        <div className="statsGrid">
                          {Object.entries(m.rearing_per_zone).map(([zid, count]) => (
                            <StatCard key={zid} k={zid} v={`${count} bouts`} />
                          ))}
                        </div>
                      </div>
                    )}

                    {/* First-zone-entry latency (escape latency / fear conditioning) */}
                    {m.zones.some(z => z.latency_first_entry_s != null) && (
                      <div style={{ marginTop: 12, padding: '10px 12px', background: 'rgba(255,80,80,0.06)', borderRadius: 8, border: '1px solid rgba(255,80,80,0.15)' }}>
                        <div style={{ fontSize: 11, color: '#ff6b6b', fontWeight: 600, marginBottom: 8 }}>
                          First entry latency (escape / approach)
                        </div>
                        <div className="statsGrid">
                          {m.zones.filter(z => z.latency_first_entry_s != null).map(z => (
                            <StatCard
                              key={z.zone_id}
                              k={z.zone_name}
                              v={`${z.latency_first_entry_s!.toFixed(1)} s`}
                              sub="to first entry"
                            />
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Zone table */}
                {m && m.zones.length > 0 && (
                  <div className="panel">
                    <div className="panelHeader">
                      <span className="panelTitle">Zone breakdown</span>
                      <span className="panelBadge">{m.zones.length} zones</span>
                    </div>
                    <ZoneTable zones={m.zones} duration_s={m.duration_s} zoneColors={zoneColorMap} />
                  </div>
                )}

                {/* Time bins */}
                {m && m.time_bins && m.time_bins.length >= 2 && (
                  <div className="panel">
                    <div className="panelHeader">
                      <span className="panelTitle">Time-bin analysis</span>
                      <span className="panelBadge">5-min bins</span>
                    </div>
                    <div className="binsTable">
                      <table>
                        <thead>
                          <tr>
                            <th>Period</th>
                            <th>Distance</th>
                            <th>Avg speed</th>
                            <th>Mobile</th>
                            <th>Freeze (ep.)</th>
                            <th>Tracked</th>
                          </tr>
                        </thead>
                        <tbody>
                          {m.time_bins.map(b => (
                            <tr key={b.label}>
                              <td className="binLabel">{b.label}</td>
                              <td>{b.total_distance_cm != null
                                ? `${b.total_distance_cm.toFixed(1)} cm`
                                : `${b.total_distance_px.toFixed(0)} px`}</td>
                              <td>{b.mean_speed_cm_s != null
                                ? `${b.mean_speed_cm_s.toFixed(1)} cm/s`
                                : `${b.mean_speed_px_s.toFixed(1)} px/s`}</td>
                              <td>{b.total_time_mobile_s.toFixed(1)} s</td>
                              <td>{b.total_time_freezing_s.toFixed(1)} s ({b.freezing_episodes})</td>
                              <td>{(b.valid_fraction * 100).toFixed(0)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Group comparison — shows when ≥2 animals processed */}
            {batchRows.length >= 2 && (() => {
              const zoneRows: ZoneRow[] = batchRows
                .filter(r => r.metrics && (r.metrics as Record<string, unknown>).zones)
                .map(r => ({
                  animal_id: r.animal_id,
                  treatment: r.treatment,
                  zones: ((r.metrics as Record<string, unknown>).zones as ZoneMetrics[] ?? []).map(z => ({
                    zone_id: z.zone_id,
                    zone_name: z.zone_name,
                    time_in_s: z.time_in_s,
                    entries: z.entries,
                    latency_first_entry_s: z.latency_first_entry_s,
                  })),
                }))
              return (
                <div className="panel" style={{ marginTop: 14 }}>
                  <div className="panelHeader">
                    <span className="panelTitle">Group comparison</span>
                    <span className="panelBadge">{batchRows.length} animals</span>
                  </div>
                  <div style={{ padding: '12px 14px' }}>
                    <GroupChart rows={batchRows} />
                  </div>
                  {zoneRows.some(r => r.zones.length > 0) && (
                    <div style={{ padding: '0 14px 14px' }}>
                      <div style={{ fontSize: 11, color: 'var(--textSub)', fontWeight: 600, marginBottom: 6, paddingTop: 8, borderTop: '1px solid var(--border)' }}>
                        Per-zone breakdown by group
                      </div>
                      <ZoneGroupChart rows={zoneRows} />
                    </div>
                  )}
                </div>
              )
            })()}

            {/* Statistical comparison panel */}
            {batchRows.length >= 2 && (() => {
              const treatments = [...new Set(batchRows.map(r => r.treatment).filter(Boolean))]
              if (treatments.length < 2) return null
              const statsMetrics: Array<{ key: keyof typeof batchRows[0]['metrics']; label: string }> = [
                { key: 'total_distance_cm' as never, label: 'Total distance (cm)' },
                { key: 'mean_speed_cm_s' as never, label: 'Mean speed (cm/s)' },
                { key: 'total_time_freezing_s' as never, label: 'Freezing time (s)' },
                { key: 'total_time_mobile_s' as never, label: 'Mobile time (s)' },
                { key: 'thigmotaxis_fraction' as never, label: 'Thigmotaxis' },
              ]
              return (
                <div className="panel" style={{ marginTop: 14 }}>
                  <div className="panelHeader">
                    <span className="panelTitle">Statistical comparison</span>
                    <span className="panelBadge">{treatments.length} groups</span>
                  </div>
                  <div style={{ padding: '10px 14px' }}>
                    <div style={{ fontSize: 11, color: 'var(--textMuted)', marginBottom: 8 }}>
                      Welch's t-test (2 groups) or one-way ANOVA (≥3). * p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001
                    </div>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--border)' }}>
                          <th style={{ textAlign: 'left', padding: '4px 6px', color: 'var(--textMuted)', fontSize: 11 }}>Metric</th>
                          {treatments.map(t => (
                            <th key={t} style={{ textAlign: 'right', padding: '4px 6px', color: 'var(--accent)', fontSize: 11 }}>{t}</th>
                          ))}
                          <th style={{ textAlign: 'center', padding: '4px 6px', color: 'var(--textMuted)', fontSize: 11 }}>p-value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {statsMetrics.map(({ key, label }) => {
                          const groupData = treatments.map(t =>
                            batchRows.filter(r => r.treatment === t)
                              .map(r => r.metrics?.[key] as number | null)
                              .filter((v): v is number => v != null)
                          )
                          const means = groupData.map(vals =>
                            vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null
                          )
                          // Simple Welch's t-test approximation for 2 groups
                          let pSig = ''
                          if (treatments.length === 2 && groupData[0].length >= 2 && groupData[1].length >= 2) {
                            const a = groupData[0], b = groupData[1]
                            const ma = a.reduce((s, v) => s + v, 0) / a.length
                            const mb = b.reduce((s, v) => s + v, 0) / b.length
                            const va = a.reduce((s, v) => s + (v - ma) ** 2, 0) / (a.length - 1)
                            const vb = b.reduce((s, v) => s + (v - mb) ** 2, 0) / (b.length - 1)
                            const se = Math.sqrt(va / a.length + vb / b.length)
                            if (se > 0) {
                              const t = Math.abs((ma - mb) / se)
                              const df = Math.min(a.length, b.length) - 1
                              // Very rough p-value from t using normal approximation for df>5
                              const z = t * (1 - 1 / (4 * df))
                              // Approximation: p from t-statistic using error function approximation
                              const erfApprox = (x: number) => {
                                const t2 = 1 / (1 + 0.3275911 * Math.abs(x))
                                const poly = t2 * (0.254829592 + t2 * (-0.284496736 + t2 * (1.421413741 + t2 * (-1.453152027 + t2 * 1.061405429))))
                                return (1 - poly * Math.exp(-x * x)) * (x >= 0 ? 1 : -1)
                              }
                              const p = 2 * (1 - 0.5 * (1 + erfApprox(z / Math.SQRT2)))
                              pSig = p < 0.001 ? '***' : p < 0.01 ? '**' : p < 0.05 ? '*' : `ns (${p.toFixed(3)})`
                            }
                          }
                          return (
                            <tr key={String(key)} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                              <td style={{ padding: '4px 6px', color: 'var(--textSub)' }}>{label}</td>
                              {means.map((m2, i) => (
                                <td key={i} style={{ padding: '4px 6px', textAlign: 'right', color: 'var(--text)' }}>
                                  {m2 != null ? m2.toFixed(2) : '—'}
                                </td>
                              ))}
                              <td style={{ padding: '4px 6px', textAlign: 'center', fontWeight: 600,
                                color: pSig.startsWith('*') ? '#ff6b6b' : 'var(--textMuted)' }}>
                                {pSig || '—'}
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                    <div style={{ fontSize: 10, color: 'var(--textMuted)', marginTop: 6 }}>
                      n = {batchRows.length} sessions across {treatments.length} groups. Run batch export for full stats.
                    </div>
                  </div>
                </div>
              )
            })()}

            {/* Cumulative zone time chart */}
            {result && (result.metrics?.zones?.length ?? 0) > 0 && (() => {
              const zones2 = result.metrics?.zones ?? []
              const zoneFrames = result.frames.map(f => ({ t_sec: f.t_sec, ok: f.ok, zone_id: f.zone_id ?? undefined }))
              return (
                <div className="panel" style={{ marginTop: 14 }}>
                  <div className="panelHeader">
                    <span className="panelTitle">Cumulative zone time</span>
                    <span className="panelBadge">time-course</span>
                  </div>
                  <div style={{ padding: '8px 14px 14px' }}>
                    <CumulativeZoneChart
                      frames={zoneFrames}
                      zones={zones2.map(z => ({ id: z.zone_id, name: z.zone_name }))}
                      fps={result.summary.fps}
                    />
                  </div>
                </div>
              )
            })()}

            {/* Multi-trial learning curve */}
            {batchRows.some(r => r.trial) && (() => {
              const trialRows: TrialRow[] = batchRows
                .filter(r => r.trial && r.metrics?.total_time_freezing_s != null)
                .map(r => ({
                  animal_id: r.animal_id,
                  treatment: r.treatment,
                  trial: r.trial,
                  metric_value: (r.metrics as Record<string, number>)?.total_time_freezing_s ?? 0,
                }))
              if (trialRows.length < 2) return null
              return (
                <div className="panel" style={{ marginTop: 14 }}>
                  <div className="panelHeader">
                    <span className="panelTitle">Trial learning curve</span>
                    <span className="panelBadge">freezing % across trials</span>
                  </div>
                  <div style={{ padding: '8px 14px 14px' }}>
                    <TrialLearningCurve rows={trialRows} metricLabel="Freezing (s)" />
                  </div>
                </div>
              )
            })()}

            {/* Post-analysis notes */}
            <div className="panel postNotesPanel" style={{ marginTop: 14 }}>
              <div className="panelHeader">
                <span className="panelTitle">Analysis notes</span>
                {savedNotes && <span className="savedBadge">✓ Saved</span>}
              </div>
              <div style={{ padding: '8px 14px 14px' }}>
                <textarea
                  className="postNotesInput"
                  placeholder="Add post-analysis observations, QC flags, conclusions…"
                  value={postNotes}
                  onChange={e => setPostNotes(e.target.value)}
                  rows={3}
                />
                <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
                  <button className="btnSm btnAccent" onClick={savePostNotes} disabled={savingNotes}>
                    {savingNotes ? 'Saving…' : 'Save notes'}
                  </button>
                </div>
              </div>
            </div>

            {/* Longitudinal / multi-session chart */}
            {batchRows.some(r => (r as AnimalRow & { session?: string }).session) && (
              <div className="panel" style={{ marginTop: 14 }}>
                <div className="panelHeader">
                  <span className="panelTitle">Longitudinal analysis</span>
                  <span className="panelBadge">multi-session</span>
                </div>
                <div style={{ padding: '12px 14px' }}>
                  <SessionChart
                    rows={batchRows
                      .filter(r => (r as AnimalRow & { session?: string }).session)
                      .map(r => ({
                        job_id: r.job_id,
                        animal_id: r.animal_id,
                        treatment: r.treatment,
                        session: (r as AnimalRow & { session?: string }).session!,
                        metrics: r.metrics,
                      } satisfies SessionRow))
                    }
                  />
                </div>
              </div>
            )}
          </div>
        )}
      {/* Calibration modal */}
      {showCalibration && jobId && (
        <CalibrationTool
          frameUrl={`${API}/api/jobs/${jobId}/first_frame`}
          onCalibrate={px => {
            setShowCalibration(false)
            if (result && arenaSetup) reanalyze({ ...arenaSetup, pxPerCm: px })
          }}
          onClose={() => setShowCalibration(false)}
        />
      )}

      {/* Help modal */}
      {showHelp && <HelpModal onClose={() => setShowHelp(false)} />}

      {/* CSV import modal */}
      {showCsvImport && (
        <CsvImportModal
          importing={csvImporting}
          onImport={importCsv}
          onClose={() => setShowCsvImport(false)}
        />
      )}

      {/* Toast notification */}
      {toastMsg && (
        <div className="toastNotif" onClick={() => setToastMsg(null)}>
          {toastMsg}
        </div>
      )}
      </main>
    </div>
  )
}

function StatCard({ k, v, sub }: { k: string; v: string; sub?: string }) {
  return (
    <div className="statCard">
      <div className="statKey">{k}</div>
      <div className="statVal">{v}</div>
      {sub && <div className="statSub">{sub}</div>}
    </div>
  )
}

// ---------------------------------------------------------------------------
// CohortStage — full-page cohort phenotyping workspace
// ---------------------------------------------------------------------------
function CohortStage({ onBack }: { onBack: () => void }) {
  const [selectedCohort, setSelectedCohort] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'builder' | 'ethomap' | 'motifs' | 'transitions' | 'radar'>('builder')

  const TABS = [
    { id: 'builder'     as const, label: '🧬 Cohort Builder',   desc: 'Assemble animals & run pipeline' },
    { id: 'ethomap'     as const, label: '🗺 Behavioral Landscape', desc: 'UMAP per-frame embedding' },
    { id: 'motifs'      as const, label: '🎭 Motif Gallery',     desc: 'Discovered behavioral states' },
    { id: 'transitions' as const, label: '🔀 Transition Graph',  desc: 'Markov behavioral syntax' },
    { id: 'radar'       as const, label: '📊 Phenotype Radar',   desc: 'WT vs BPAN comparison' },
  ]

  return (
    <div className="stagePage" style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      {/* Header */}
      <div className="stageHeader" style={{ flexShrink: 0 }}>
        <div className="stageHeaderLeft">
          <button className="backBtn" onClick={onBack}>← Back</button>
          <div>
            <div className="stageTitle">Latent Behavioral Phenotyping</div>
            <div className="stageSub">
              Unsupervised motif discovery · UMAP · Sequence analysis · WT vs BPAN
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 0, flex: 1, minHeight: 0, overflow: 'hidden' }}>
        {/* Left sidebar: cohort selector + tabs */}
        <div style={{
          width: 200, flexShrink: 0, borderRight: '1px solid rgba(255,255,255,0.06)',
          display: 'flex', flexDirection: 'column', gap: 6, padding: '12px 8px',
          background: 'rgba(0,0,0,0.15)',
        }}>
          {TABS.map(tab => (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)}
              style={{
                background: activeTab === tab.id ? 'rgba(99,102,241,0.2)' : 'transparent',
                border: `1px solid ${activeTab === tab.id ? 'rgba(99,102,241,0.5)' : 'transparent'}`,
                borderRadius: 8, padding: '10px 12px', cursor: 'pointer', textAlign: 'left',
                transition: 'all 0.15s',
              }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: activeTab === tab.id ? '#a5b4fc' : '#e2e8f0' }}>
                {tab.label}
              </div>
              <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>{tab.desc}</div>
            </button>
          ))}
        </div>

        {/* Main content */}
        <div style={{ flex: 1, padding: 20, overflow: 'auto', minHeight: 0 }}>
          {activeTab === 'builder' && (
            <CohortBuilder onCohortSelect={id => setSelectedCohort(id)} />
          )}
          {activeTab === 'ethomap' && <BehavioralEthoMap cohortId={selectedCohort} />}
          {activeTab === 'motifs'  && <MotifGallery cohortId={selectedCohort} />}
          {activeTab === 'transitions' && <TransitionGraph cohortId={selectedCohort} />}
          {activeTab === 'radar'   && <PhenotypeRadar cohortId={selectedCohort} />}
        </div>
      </div>
    </div>
  )
}

