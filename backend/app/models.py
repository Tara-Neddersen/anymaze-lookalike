from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float
    y: float


class TrackingFrame(BaseModel):
    frame_index: int
    t_sec: float
    centroid: Point | None = None
    ok: bool
    area_px: float | None = None
    # Enriched fields baked in by jobs.py after tracking
    zone_id: str | None = None
    speed_px_s: float | None = None
    speed_cm_s: float | None = None
    heading_deg: float | None = None
    heading_delta_deg: float | None = None
    rearing: bool = False
    quality: str = "ok"
    # Pose-based fields (set when using DLC / SLEAP engines)
    body_length_px: float | None = None
    keypoints: dict[str, Any] | None = None
    # Derived kinematics
    angular_velocity_deg_s: float | None = None


class TrackingSummary(BaseModel):
    fps: float
    frame_count: int
    ok_frames: int
    arena_size_px: tuple[int, int]


class TrackingResult(BaseModel):
    summary: TrackingSummary
    frames: list[TrackingFrame]

    model_config = {"arbitrary_types_allowed": True}

    # Not a Pydantic field — stored as a plain instance attribute after construction.
    # Populated by tracker.py; used by jobs.py for metrics; not serialised.
    _raw_frames: list[dict[str, Any]] = []


class ZoneMetrics(BaseModel):
    zone_id: str
    zone_name: str
    time_in_s: float
    entries: int
    latency_first_entry_s: float | None
    mean_speed_in_zone_cm_s: float | None
    distance_in_zone_cm: float | None


class TimeBin(BaseModel):
    label: str
    start_s: float
    end_s: float
    total_distance_px: float
    total_distance_cm: float | None
    mean_speed_px_s: float
    mean_speed_cm_s: float | None
    total_time_mobile_s: float
    total_time_freezing_s: float
    freezing_episodes: int
    valid_fraction: float


class AnimalMeta(BaseModel):
    animal_id: str = ""
    treatment: str = ""
    trial: str = ""
    notes: str = ""
    n_animals: int = 1
    engine: str = "opencv_mog2_centroid"
    # Lab workflow fields
    session: str = ""          # e.g. "Day 1", "Week 2" for longitudinal studies
    experiment_id: str = ""    # cohort/experiment grouping
    experimenter: str = ""     # who ran the experiment


class ZoneEvent(BaseModel):
    """A single zone entry/exit event."""
    zone_id: str
    zone_name: str
    entry_t: float
    exit_t: float
    duration_s: float


class BehaviorEvent(BaseModel):
    """A contiguous run of a named behavioral state."""
    behavior: str   # "mobile" | "immobile" | "freezing" | "rearing"
    start_t: float
    end_t: float
    duration_s: float


class NORMetrics(BaseModel):
    """Novel Object Recognition discrimination metrics."""
    novel_zone_id: str
    familiar_zone_id: str
    time_novel_s: float
    time_familiar_s: float
    discrimination_index: float   # (T_novel − T_familiar) / (T_novel + T_familiar)
    preference_index: float       # T_novel / (T_novel + T_familiar)
    total_exploration_s: float


class SocialMetrics(BaseModel):
    """Social proximity metrics for multi-animal recordings."""
    total_time_near_s: float
    near_episodes: int
    first_contact_latency_s: float | None
    proximity_threshold_cm: float | None
    proximity_threshold_px: float


class PlacePreference(BaseModel):
    """Left/right and top/bottom spatial preference."""
    time_left_s: float
    time_right_s: float
    time_top_s: float
    time_bottom_s: float
    # (left - right) / total ; positive = prefers left side
    preference_lr: float
    # (top - bottom) / total ; positive = prefers top
    preference_tb: float


class MWMMetrics(BaseModel):
    """Morris Water Maze: quadrant occupancy and escape latency trends."""
    # Fraction of time spent in each quadrant (target, opposite, left, right)
    target_quadrant_pct: float
    opposite_quadrant_pct: float
    left_quadrant_pct: float
    right_quadrant_pct: float
    # Per-trial escape latency (if trials detected from time bins)
    trial_escape_latency_s: list[float] = Field(default_factory=list)
    # Platform zone id (if a zone named 'platform'/'goal'/'target' exists)
    platform_zone_id: str | None = None
    platform_proximity_mean_cm: float | None = None


class YMazeMetrics(BaseModel):
    """Y-maze spontaneous alternation."""
    arm_entries: int
    alternations: int                  # triplets of consecutive different arms
    spontaneous_alternation_pct: float  # alternations / (arm_entries - 2) * 100
    arm_visit_counts: dict[str, int]   # per-arm entry count


class OpenFieldMetrics(BaseModel):
    """Open field center vs. periphery analysis."""
    center_time_s: float
    periphery_time_s: float
    center_time_pct: float
    center_entries: int
    center_distance_cm: float | None
    periphery_distance_cm: float | None


class FreezeBout(BaseModel):
    """A single contiguous freezing episode."""
    start_t: float
    end_t: float
    duration_s: float
    mean_speed_cm_s: float | None
    zone_id: str | None


class LightDarkMetrics(BaseModel):
    """Light/Dark Box: auto-detected from zones named 'light' / 'dark'."""
    light_time_s: float
    dark_time_s: float
    light_time_pct: float
    dark_time_pct: float
    light_entries: int
    dark_entries: int
    latency_to_light_s: float | None
    latency_to_dark_s: float | None
    transitions: int


class FearConditioningEpoch(BaseModel):
    """One CS or US epoch in a fear conditioning session."""
    label: str          # e.g. "CS+", "Tone 1", "Shock", "ITI"
    start_t: float
    end_t: float
    duration_s: float
    freezing_s: float
    freezing_pct: float
    freezing_episodes: int


class FearCondMetrics(BaseModel):
    """Fear conditioning: per-epoch freezing analysis."""
    epochs: list[FearConditioningEpoch]
    baseline_freezing_pct: float | None  # pre-CS period
    mean_cs_freezing_pct: float | None   # mean over all CS epochs


class EPMMetrics(BaseModel):
    """Elevated Plus Maze metrics (detected from zones named 'open' / 'closed')."""
    open_arm_time_s: float
    closed_arm_time_s: float
    open_arm_time_pct: float          # open / (open + closed) * 100
    open_arm_entries: int
    closed_arm_entries: int
    open_arm_entries_pct: float       # open_entries / total_entries * 100


class BehaviorMetrics(BaseModel):
    total_distance_px: float
    total_distance_cm: float | None
    mean_speed_px_s: float
    mean_speed_cm_s: float | None
    max_speed_px_s: float
    max_speed_cm_s: float | None
    total_time_mobile_s: float
    total_time_immobile_s: float
    total_time_freezing_s: float
    freezing_episodes: int
    thigmotaxis_fraction: float | None
    valid_fraction: float
    duration_s: float
    path_efficiency: float | None
    clockwise_rotations: float
    anticlockwise_rotations: float
    # Rearing
    total_time_rearing_s: float = 0.0
    rearing_episodes: int = 0
    # Grooming (pose-derived: spine curvature / head-body angle)
    total_time_grooming_s: float = 0.0
    grooming_episodes: int = 0
    # Pose-derived kinematics (DLC / SLEAP only)
    mean_body_length_px: float | None = None
    mean_body_length_cm: float | None = None
    mean_ear_span_px: float | None = None
    mean_spine_curvature: float | None = None
    mean_head_body_angle_deg: float | None = None
    # Multi-animal
    mean_inter_animal_dist_px: float | None = None
    mean_inter_animal_dist_cm: float | None = None
    # Zone events (entry/exit log)
    zone_events: list[ZoneEvent] = Field(default_factory=list)
    # Behavioral state runs
    behavior_events: list[BehaviorEvent] = Field(default_factory=list)
    # NOR (set when novel/familiar zones are detected)
    nor: NORMetrics | None = None
    # Social proximity (multi-animal)
    social: SocialMetrics | None = None
    # Place preference (left/right top/bottom splits)
    place_preference: PlacePreference | None = None
    # EPM (auto-detected from zone names containing 'open'/'closed')
    epm: EPMMetrics | None = None
    # Morris Water Maze (auto-detected from quadrant zones or arena quadrant split)
    mwm: MWMMetrics | None = None
    # Y-maze (auto-detected from zones named 'arm A/B/C' or similar)
    ymaze: YMazeMetrics | None = None
    # Open field center vs. periphery
    open_field: OpenFieldMetrics | None = None
    # Light/Dark box (auto-detected from zone names)
    light_dark: LightDarkMetrics | None = None
    # Fear conditioning per-epoch analysis
    fear_cond: FearCondMetrics | None = None
    # Freeze bout list (individual episodes)
    freeze_bouts: list[FreezeBout] = Field(default_factory=list)
    # Rearing count per zone
    rearing_per_zone: dict[str, int] = Field(default_factory=dict)
    zones: list[ZoneMetrics]
    time_bins: list[TimeBin]


class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "error"]
    message: str | None = None
    progress: float = Field(ge=0.0, le=1.0, default=0.0)
