from __future__ import annotations

import csv
import io
import math
from typing import Any

from .arena import assign_zones, compute_thigmotaxis, Poly
from .models import (
    BehaviorEvent, BehaviorMetrics, EPMMetrics, FearConditioningEpoch,
    FearCondMetrics, FreezeBout, LightDarkMetrics, MWMMetrics, NORMetrics,
    OpenFieldMetrics, PlacePreference, SocialMetrics, TimeBin,
    YMazeMetrics, ZoneEvent, ZoneMetrics,
)


# ---------------------------------------------------------------------------
# Thresholds (matching AnyMaze defaults)
# ---------------------------------------------------------------------------
IMMOBILITY_THRESHOLD_CM_S = 2.0   # below = immobile (cm/s)
FREEZING_THRESHOLD_CM_S = 0.5     # below = freezing (cm/s)
IMMOBILITY_MIN_DURATION_S = 1.0   # must persist ≥ 1 s to count
FREEZING_MIN_DURATION_S = 1.0

# Pixel-space thresholds used when no calibration is set.
# Approximate assuming ~20 px/cm; these are exported so the frontend
# can draw matching threshold lines on the speed chart.
IMMOBILITY_THRESHOLD_PX_S = 8.0
FREEZING_THRESHOLD_PX_S = 2.0


def _episode_stats(
    frames: list[dict[str, Any]],
    fps: float,
    condition_fn: Any,
    min_duration_s: float,
) -> tuple[float, int]:
    """Count total time and episodes where condition_fn(frame) is True for ≥ min_duration_s."""
    total_time = 0.0
    episodes = 0
    in_episode = False
    ep_start = 0.0

    for f in frames:
        if not f.get("ok") or not f.get("centroid"):
            if in_episode:
                dur = f["t_sec"] - ep_start
                if dur >= min_duration_s:
                    total_time += dur
                    episodes += 1
                in_episode = False
            continue

        cond = condition_fn(f)
        t = float(f["t_sec"])

        if cond and not in_episode:
            in_episode = True
            ep_start = t
        elif not cond and in_episode:
            dur = t - ep_start
            if dur >= min_duration_s:
                total_time += dur
                episodes += 1
            in_episode = False

    if in_episode and frames:
        dur = float(frames[-1]["t_sec"]) - ep_start
        if dur >= min_duration_s:
            total_time += dur
            episodes += 1

    return total_time, episodes


def compute_all_metrics(
    raw_frames: list[dict[str, Any]],
    fps: float,
    px_per_cm: float,
    zones: list[dict[str, Any]],
    arena_poly: Poly | None,
    thigmotaxis_margin_px: float = 50.0,
    freeze_threshold_cm_s: float = 0.0,
) -> BehaviorMetrics:
    """
    Compute AnyMaze-grade metrics from the enriched raw_frames list.

    raw_frames items must have: t_sec, ok, centroid {x,y}, speed_px_s, speed_cm_s.
    Zone assignment is performed here if zones are provided.
    """
    # Use calibrated 5 cm margin when possible (standard open-field thigmotaxis)
    if px_per_cm > 0:
        thigmotaxis_margin_px = 5.0 * px_per_cm
    # Assign zones in-place
    if zones:
        assign_zones(raw_frames, zones)

    # --- Global distance and speed ---
    total_dist_px = 0.0
    total_dist_cm = 0.0
    max_speed_px = 0.0
    max_speed_cm = 0.0
    speed_px_samples: list[float] = []
    speed_cm_samples: list[float] = []

    prev: dict[str, Any] | None = None
    for f in raw_frames:
        if not f.get("ok") or not f.get("centroid"):
            prev = None
            continue
        if prev is not None:
            dx = f["centroid"]["x"] - prev["centroid"]["x"]
            dy = f["centroid"]["y"] - prev["centroid"]["y"]
            dt = f["t_sec"] - prev["t_sec"]
            dist_px = math.hypot(dx, dy)
            total_dist_px += dist_px
            if dt > 0:
                spx = dist_px / dt
                max_speed_px = max(max_speed_px, spx)
                speed_px_samples.append(spx)
                if px_per_cm > 0:
                    scm = spx / px_per_cm
                    total_dist_cm += dist_px / px_per_cm
                    max_speed_cm = max(max_speed_cm, scm)
                    speed_cm_samples.append(scm)
        prev = f

    mean_speed_px = sum(speed_px_samples) / len(speed_px_samples) if speed_px_samples else 0.0
    mean_speed_cm = sum(speed_cm_samples) / len(speed_cm_samples) if speed_cm_samples else None

    # --- Duration ---
    valid = [f for f in raw_frames if f.get("ok") and f.get("centroid")]
    duration_s = (float(raw_frames[-1]["t_sec"]) if raw_frames else 0.0)

    # --- Immobility & freezing (with optional threshold override) ---
    eff_freeze_cm = freeze_threshold_cm_s if freeze_threshold_cm_s > 0 else FREEZING_THRESHOLD_CM_S
    eff_freeze_px = (eff_freeze_cm * px_per_cm) if px_per_cm > 0 else FREEZING_THRESHOLD_PX_S

    def _freeze_check(f: dict[str, Any]) -> bool:
        if f.get("speed_cm_s") is not None:
            return float(f["speed_cm_s"]) < eff_freeze_cm
        if f.get("speed_px_s") is not None:
            return float(f["speed_px_s"]) < eff_freeze_px
        return False

    time_immobile, _ = _episode_stats(
        raw_frames, fps, lambda f: _is_immobile(f, px_per_cm), IMMOBILITY_MIN_DURATION_S
    )
    time_freezing, freezing_eps = _episode_stats(
        raw_frames, fps, _freeze_check, FREEZING_MIN_DURATION_S
    )
    time_mobile = duration_s - time_immobile

    # --- Thigmotaxis ---
    thigmo = None
    if arena_poly and len(arena_poly) >= 3:
        thigmo = compute_thigmotaxis(raw_frames, arena_poly, thigmotaxis_margin_px)

    # --- Zone metrics ---
    zone_metrics_list: list[ZoneMetrics] = []

    for z in zones:
        zid = z["id"]
        zname = z.get("name", zid)
        in_zone_frames = [f for f in raw_frames if f.get("ok") and f.get("zone_id") == zid]
        time_in_s = len(in_zone_frames) / fps if fps > 0 else 0.0

        # Count entries: transitions from outside → inside
        entries = 0
        latency_first: float | None = None
        prev_in = False
        for f in raw_frames:
            if not f.get("ok"):
                prev_in = False
                continue
            in_now = f.get("zone_id") == zid
            if in_now and not prev_in:
                entries += 1
                if latency_first is None:
                    latency_first = float(f["t_sec"])
            prev_in = in_now

        # Speed in zone
        zone_speeds_cm: list[float] = []
        zone_dist_cm = 0.0
        prev_in_frame: dict[str, Any] | None = None
        for f in in_zone_frames:
            if prev_in_frame is not None and f.get("centroid") and prev_in_frame.get("centroid"):
                dx = f["centroid"]["x"] - prev_in_frame["centroid"]["x"]
                dy = f["centroid"]["y"] - prev_in_frame["centroid"]["y"]
                dt = f["t_sec"] - prev_in_frame["t_sec"]
                d_px = math.hypot(dx, dy)
                if dt > 0:
                    spx = d_px / dt
                    if px_per_cm > 0:
                        zone_speeds_cm.append(spx / px_per_cm)
                        zone_dist_cm += d_px / px_per_cm
            prev_in_frame = f

        mean_z_speed = sum(zone_speeds_cm) / len(zone_speeds_cm) if zone_speeds_cm else None

        zone_metrics_list.append(ZoneMetrics(
            zone_id=zid,
            zone_name=zname,
            time_in_s=round(time_in_s, 3),
            entries=entries,
            latency_first_entry_s=round(latency_first, 3) if latency_first is not None else None,
            mean_speed_in_zone_cm_s=round(mean_z_speed, 3) if mean_z_speed is not None else None,
            distance_in_zone_cm=round(zone_dist_cm, 3) if px_per_cm > 0 else None,
        ))

    valid_count = len(valid)
    total_frames = len(raw_frames)

    # --- Rotation (cumulative CW/CCW from heading deltas) ---
    cw_total = 0.0
    ccw_total = 0.0
    for f in raw_frames:
        delta = f.get("heading_delta_deg")
        if delta is None:
            continue
        if delta > 0:
            cw_total += delta
        else:
            ccw_total += abs(delta)
    cw_rotations = round(cw_total / 360.0, 2)
    ccw_rotations = round(ccw_total / 360.0, 2)

    # --- Path efficiency (net displacement / total path length) ---
    path_eff = None
    if total_dist_px > 0:
        first_valid = next((f for f in raw_frames if f.get("ok") and f.get("centroid")), None)
        last_valid = next((f for f in reversed(raw_frames) if f.get("ok") and f.get("centroid")), None)
        if first_valid and last_valid:
            net = math.hypot(
                last_valid["centroid"]["x"] - first_valid["centroid"]["x"],
                last_valid["centroid"]["y"] - first_valid["centroid"]["y"]
            )
            path_eff = round(net / total_dist_px, 4)

    # --- Rearing ---
    REARING_MIN_DURATION_S = 0.2
    time_rearing, rearing_eps = _episode_stats(
        raw_frames, fps, lambda f: bool(f.get("rearing")), REARING_MIN_DURATION_S
    )

    # --- Grooming (pose-derived) ---
    GROOMING_MIN_DURATION_S = 0.5
    time_grooming, grooming_eps = _episode_stats(
        raw_frames, fps, lambda f: bool(f.get("grooming")), GROOMING_MIN_DURATION_S
    )

    # --- Pose-derived kinematics ---
    bl_vals = [f["body_length_px"] for f in raw_frames if f.get("body_length_px") is not None]
    es_vals = [f["ear_span_px"] for f in raw_frames if f.get("ear_span_px") is not None]
    sc_vals = [f["spine_curvature"] for f in raw_frames if f.get("spine_curvature") is not None]
    hb_vals = [f["head_body_angle_deg"] for f in raw_frames if f.get("head_body_angle_deg") is not None]
    mean_body_len_px = round(sum(bl_vals) / len(bl_vals), 2) if bl_vals else None
    mean_body_len_cm = round(mean_body_len_px / px_per_cm, 2) if (mean_body_len_px and px_per_cm > 0) else None
    mean_ear_span_px = round(sum(es_vals) / len(es_vals), 2) if es_vals else None
    mean_spine_curv  = round(sum(sc_vals) / len(sc_vals), 4) if sc_vals else None
    mean_head_body   = round(sum(hb_vals) / len(hb_vals), 2) if hb_vals else None

    # --- Inter-animal distance (multi-animal mode) ---
    ia_dists_px = [f["inter_animal_dist_px"] for f in raw_frames
                   if f.get("inter_animal_dist_px") is not None]
    ia_dists_cm = [f["inter_animal_dist_cm"] for f in raw_frames
                   if f.get("inter_animal_dist_cm") is not None]
    mean_ia_px = round(sum(ia_dists_px) / len(ia_dists_px), 2) if ia_dists_px else None
    mean_ia_cm = round(sum(ia_dists_cm) / len(ia_dists_cm), 2) if ia_dists_cm else None

    # --- Time-binned analysis (5-minute bins, or 3 equal bins for short videos) ---
    time_bins = _compute_time_bins(raw_frames, fps, px_per_cm, duration_s)

    # --- Zone entry/exit event log ---
    zone_name_map = {z["id"]: z.get("name", z["id"]) for z in zones}
    zone_events = _compute_zone_events(raw_frames, zone_name_map)

    # --- Behavioral state runs ---
    behavior_events = _compute_behavior_events(raw_frames, fps, px_per_cm)

    # --- NOR metrics (Novel Object Recognition) ---
    nor = _compute_nor(zone_metrics_list)

    # --- Social proximity (multi-animal) ---
    social = _compute_social_proximity(raw_frames, fps, px_per_cm)

    # --- Place preference (left/right, top/bottom splits) ---
    place_pref = _compute_place_preference(raw_frames, fps, px_per_cm)

    # --- EPM metrics (auto-detected from zone names) ---
    epm = _compute_epm_metrics(zone_metrics_list)

    # --- Light/Dark Box ---
    light_dark = _compute_light_dark_metrics(zone_metrics_list, zone_events)

    # --- Fear conditioning (requires zones named "CS"/"US"/"ITI"/"baseline") ---
    fear_cond = _compute_fear_cond_metrics(raw_frames, fps, px_per_cm, zone_events)

    # --- Freeze bouts list ---
    freeze_bouts = _compute_freeze_bouts(raw_frames, fps, px_per_cm, eff_freeze_cm, eff_freeze_px)

    # --- Rearing per zone ---
    rearing_per_zone = _compute_rearing_per_zone(raw_frames, zones)

    # --- Morris Water Maze ---
    mwm = _compute_mwm_metrics(raw_frames, fps, px_per_cm, zone_metrics_list, duration_s)

    # --- Y-maze spontaneous alternation ---
    ymaze = _compute_ymaze_metrics(zone_metrics_list, zone_events)

    # --- Open field center/periphery ---
    open_field = _compute_open_field_metrics(zone_metrics_list, px_per_cm)

    return BehaviorMetrics(
        total_distance_px=round(total_dist_px, 2),
        total_distance_cm=round(total_dist_cm, 3) if px_per_cm > 0 else None,
        mean_speed_px_s=round(mean_speed_px, 3),
        mean_speed_cm_s=round(mean_speed_cm, 3) if mean_speed_cm is not None else None,
        max_speed_px_s=round(max_speed_px, 3),
        max_speed_cm_s=round(max_speed_cm, 3) if px_per_cm > 0 else None,
        total_time_mobile_s=round(max(0.0, time_mobile), 3),
        total_time_immobile_s=round(time_immobile, 3),
        total_time_freezing_s=round(time_freezing, 3),
        freezing_episodes=freezing_eps,
        thigmotaxis_fraction=round(thigmo, 4) if thigmo is not None else None,
        valid_fraction=round(valid_count / max(1, total_frames), 4),
        duration_s=round(duration_s, 3),
        path_efficiency=path_eff,
        clockwise_rotations=cw_rotations,
        anticlockwise_rotations=ccw_rotations,
        total_time_rearing_s=round(time_rearing, 3),
        rearing_episodes=rearing_eps,
        total_time_grooming_s=round(time_grooming, 3),
        grooming_episodes=grooming_eps,
        mean_body_length_px=mean_body_len_px,
        mean_body_length_cm=mean_body_len_cm,
        mean_ear_span_px=mean_ear_span_px,
        mean_spine_curvature=mean_spine_curv,
        mean_head_body_angle_deg=mean_head_body,
        mean_inter_animal_dist_px=mean_ia_px,
        mean_inter_animal_dist_cm=mean_ia_cm,
        zone_events=zone_events,
        behavior_events=behavior_events,
        nor=nor,
        social=social,
        place_preference=place_pref,
        epm=epm,
        mwm=mwm,
        ymaze=ymaze,
        open_field=open_field,
        light_dark=light_dark,
        fear_cond=fear_cond,
        freeze_bouts=freeze_bouts,
        rearing_per_zone=rearing_per_zone,
        zones=zone_metrics_list,
        time_bins=time_bins,
    )


def _compute_time_bins(
    raw_frames: list[dict[str, Any]],
    fps: float,
    px_per_cm: float,
    duration_s: float,
) -> list[TimeBin]:
    """Split session into time bins and compute core metrics per bin."""
    BIN_SEC = 300.0  # 5 minutes
    MIN_BINS = 2
    MAX_BINS = 12

    if duration_s <= 0:
        return []

    # Choose bin size: 5 min if possible, otherwise split into 3 equal bins
    n_bins = max(MIN_BINS, min(MAX_BINS, int(duration_s // BIN_SEC)))
    bin_dur = duration_s / n_bins

    bins: list[TimeBin] = []
    for b in range(n_bins):
        t0 = b * bin_dur
        t1 = (b + 1) * bin_dur
        bin_frames = [f for f in raw_frames if t0 <= f["t_sec"] < t1]
        label_min = lambda s: f"{int(s // 60)}:{int(s % 60):02d}"
        label = f"{label_min(t0)} – {label_min(t1)}"

        dist_px = 0.0
        dist_cm = 0.0
        speed_px_samples: list[float] = []
        speed_cm_samples: list[float] = []
        prev: dict[str, Any] | None = None
        for f in bin_frames:
            if not f.get("ok") or not f.get("centroid"):
                prev = None
                continue
            if prev is not None:
                dx = f["centroid"]["x"] - prev["centroid"]["x"]
                dy = f["centroid"]["y"] - prev["centroid"]["y"]
                dt = f["t_sec"] - prev["t_sec"]
                d = math.hypot(dx, dy)
                dist_px += d
                if dt > 0:
                    spx = d / dt
                    speed_px_samples.append(spx)
                    if px_per_cm > 0:
                        speed_cm_samples.append(spx / px_per_cm)
                        dist_cm += d / px_per_cm
            prev = f

        mean_spx = sum(speed_px_samples) / len(speed_px_samples) if speed_px_samples else 0.0
        mean_scm = sum(speed_cm_samples) / len(speed_cm_samples) if speed_cm_samples else None

        time_imm, _ = _episode_stats(bin_frames, fps, lambda f: _is_immobile(f, px_per_cm), IMMOBILITY_MIN_DURATION_S)
        time_frz, frz_eps = _episode_stats(bin_frames, fps, lambda f: _is_freezing(f, px_per_cm), FREEZING_MIN_DURATION_S)
        time_mob = bin_dur - time_imm

        valid_b = sum(1 for f in bin_frames if f.get("ok"))
        vfrac = valid_b / max(1, len(bin_frames))

        bins.append(TimeBin(
            label=label,
            start_s=round(t0, 2),
            end_s=round(t1, 2),
            total_distance_px=round(dist_px, 2),
            total_distance_cm=round(dist_cm, 3) if px_per_cm > 0 else None,
            mean_speed_px_s=round(mean_spx, 3),
            mean_speed_cm_s=round(mean_scm, 3) if mean_scm is not None else None,
            total_time_mobile_s=round(max(0.0, time_mob), 3),
            total_time_freezing_s=round(time_frz, 3),
            freezing_episodes=frz_eps,
            valid_fraction=round(vfrac, 4),
        ))

    return bins


# ---------------------------------------------------------------------------
# Zone entry/exit event log
# ---------------------------------------------------------------------------

def _compute_zone_events(
    raw_frames: list[dict[str, Any]],
    zone_name_map: dict[str, str],
) -> list[ZoneEvent]:
    events: list[ZoneEvent] = []
    current_zone: str | None = None
    entry_t: float = 0.0

    for f in raw_frames:
        if not f.get("ok"):
            if current_zone is not None:
                dur = f["t_sec"] - entry_t
                events.append(ZoneEvent(
                    zone_id=current_zone,
                    zone_name=zone_name_map.get(current_zone, current_zone),
                    entry_t=round(entry_t, 3),
                    exit_t=round(f["t_sec"], 3),
                    duration_s=round(dur, 3),
                ))
                current_zone = None
            continue

        zone = f.get("zone_id")
        if zone != current_zone:
            if current_zone is not None:
                dur = f["t_sec"] - entry_t
                events.append(ZoneEvent(
                    zone_id=current_zone,
                    zone_name=zone_name_map.get(current_zone, current_zone),
                    entry_t=round(entry_t, 3),
                    exit_t=round(f["t_sec"], 3),
                    duration_s=round(dur, 3),
                ))
            if zone is not None:
                entry_t = f["t_sec"]
            current_zone = zone

    # Close last open zone
    if current_zone is not None and raw_frames:
        last_t = float(raw_frames[-1]["t_sec"])
        dur = last_t - entry_t
        events.append(ZoneEvent(
            zone_id=current_zone,
            zone_name=zone_name_map.get(current_zone, current_zone),
            entry_t=round(entry_t, 3),
            exit_t=round(last_t, 3),
            duration_s=round(dur, 3),
        ))

    return events


# ---------------------------------------------------------------------------
# Behavioral state runs (for Gantt timeline)
# ---------------------------------------------------------------------------

def _compute_behavior_events(
    raw_frames: list[dict[str, Any]],
    fps: float,
    px_per_cm: float,
) -> list[BehaviorEvent]:
    """
    Segment session into labeled behavioral state runs.
    Priority (highest wins): freezing > rearing > immobile > mobile
    """
    events: list[BehaviorEvent] = []
    if not raw_frames:
        return events

    def _classify(f: dict[str, Any]) -> str:
        if not f.get("ok") or not f.get("centroid"):
            return "untracked"
        if f.get("rearing"):
            return "rearing"
        if _is_freezing(f, px_per_cm):
            return "freezing"
        if _is_immobile(f, px_per_cm):
            return "immobile"
        return "mobile"

    prev_state = _classify(raw_frames[0])
    seg_start = float(raw_frames[0]["t_sec"])

    for f in raw_frames[1:]:
        state = _classify(f)
        if state != prev_state:
            dur = f["t_sec"] - seg_start
            if dur > 0 and prev_state != "untracked":
                events.append(BehaviorEvent(
                    behavior=prev_state,
                    start_t=round(seg_start, 3),
                    end_t=round(float(f["t_sec"]), 3),
                    duration_s=round(dur, 3),
                ))
            prev_state = state
            seg_start = float(f["t_sec"])

    # Close last segment
    if raw_frames:
        dur = float(raw_frames[-1]["t_sec"]) - seg_start
        if dur > 0 and prev_state != "untracked":
            events.append(BehaviorEvent(
                behavior=prev_state,
                start_t=round(seg_start, 3),
                end_t=round(float(raw_frames[-1]["t_sec"]), 3),
                duration_s=round(dur, 3),
            ))

    return events


# ---------------------------------------------------------------------------
# Novel Object Recognition
# ---------------------------------------------------------------------------
_NOR_NOVEL_KEYWORDS    = {"novel", "novel_object", "new", "object1", "obj1"}
_NOR_FAMILIAR_KEYWORDS = {"familiar", "familiar_object", "old", "object2", "obj2"}


def _compute_nor(zone_metrics: list[ZoneMetrics]) -> NORMetrics | None:
    """
    Auto-detect novel/familiar zones by name and compute DI / PI.
    Zone names containing 'novel'/'new' vs 'familiar'/'old' are matched.
    """
    novel_z: ZoneMetrics | None = None
    familiar_z: ZoneMetrics | None = None
    for zm in zone_metrics:
        name_lo = zm.zone_name.lower().replace(" ", "_").replace("-", "_")
        zid_lo  = zm.zone_id.lower()
        if any(k in name_lo or k in zid_lo for k in _NOR_NOVEL_KEYWORDS):
            novel_z = zm
        elif any(k in name_lo or k in zid_lo for k in _NOR_FAMILIAR_KEYWORDS):
            familiar_z = zm

    if novel_z is None or familiar_z is None:
        return None

    tn = novel_z.time_in_s
    tf = familiar_z.time_in_s
    total = tn + tf
    if total <= 0:
        return None

    di = round((tn - tf) / total, 4)
    pi = round(tn / total, 4)

    return NORMetrics(
        novel_zone_id=novel_z.zone_id,
        familiar_zone_id=familiar_z.zone_id,
        time_novel_s=round(tn, 3),
        time_familiar_s=round(tf, 3),
        discrimination_index=di,
        preference_index=pi,
        total_exploration_s=round(total, 3),
    )


# ---------------------------------------------------------------------------
# Social proximity
# ---------------------------------------------------------------------------
_PROXIMITY_CM = 5.0   # < 5 cm = "near" contact
_PROXIMITY_PX = 50.0  # fallback when uncalibrated
_PROXIMITY_MIN_DURATION_S = 0.5


def _compute_social_proximity(
    raw_frames: list[dict[str, Any]],
    fps: float,
    px_per_cm: float,
) -> SocialMetrics | None:
    has_ia = any(f.get("inter_animal_dist_px") is not None for f in raw_frames)
    if not has_ia:
        return None

    thresh_px = _PROXIMITY_CM * px_per_cm if px_per_cm > 0 else _PROXIMITY_PX
    thresh_cm = _PROXIMITY_CM if px_per_cm > 0 else None

    def _is_near(f: dict[str, Any]) -> bool:
        d = f.get("inter_animal_dist_px")
        return d is not None and d < thresh_px

    total_near, episodes = _episode_stats(
        raw_frames, fps, _is_near, _PROXIMITY_MIN_DURATION_S
    )

    # First contact latency
    first_latency: float | None = None
    for f in raw_frames:
        if _is_near(f):
            first_latency = round(float(f["t_sec"]), 3)
            break

    return SocialMetrics(
        total_time_near_s=round(total_near, 3),
        near_episodes=episodes,
        first_contact_latency_s=first_latency,
        proximity_threshold_cm=thresh_cm,
        proximity_threshold_px=round(thresh_px, 1),
    )


def _compute_place_preference(
    raw_frames: list[dict[str, Any]],
    fps: float,
    px_per_cm: float,
) -> PlacePreference | None:
    """
    Compute left/right and top/bottom place preference by splitting the
    bounding box of all valid centroids at its midpoint.
    Returns None when fewer than 20 valid frames are available.
    """
    valid = [
        f for f in raw_frames
        if f.get("ok") and f.get("centroid") and f.get("quality") != "trimmed"
    ]
    if len(valid) < 20:
        return None

    all_x = [f["centroid"]["x"] for f in valid]
    all_y = [f["centroid"]["y"] for f in valid]
    mid_x = (min(all_x) + max(all_x)) / 2
    mid_y = (min(all_y) + max(all_y)) / 2

    dt = 1.0 / fps if fps > 0 else 1.0
    left = right = top = bottom = 0.0

    for f in valid:
        x = f["centroid"]["x"]
        y = f["centroid"]["y"]
        if x < mid_x:
            left += dt
        else:
            right += dt
        if y < mid_y:
            top += dt
        else:
            bottom += dt

    total_lr = left + right
    total_tb = top + bottom

    pref_lr = round((left - right) / total_lr, 4) if total_lr > 0 else 0.0
    pref_tb = round((top - bottom) / total_tb, 4) if total_tb > 0 else 0.0

    return PlacePreference(
        time_left_s=round(left, 3),
        time_right_s=round(right, 3),
        time_top_s=round(top, 3),
        time_bottom_s=round(bottom, 3),
        preference_lr=pref_lr,
        preference_tb=pref_tb,
    )


# ---------------------------------------------------------------------------
# Light/Dark Box helper
# ---------------------------------------------------------------------------
def _compute_light_dark_metrics(
    zone_metrics: list[ZoneMetrics],
    zone_events: list[ZoneEvent],
) -> LightDarkMetrics | None:
    light_kws = ("light", "bright", "illuminated")
    dark_kws  = ("dark", "black", "unilluminated", "shelter")

    light_zones = [z for z in zone_metrics
                   if any(kw in z.zone_name.lower() for kw in light_kws)]
    dark_zones  = [z for z in zone_metrics
                   if any(kw in z.zone_name.lower() for kw in dark_kws)
                   and not any(kw in z.zone_name.lower() for kw in light_kws)]

    if not light_zones and not dark_zones:
        return None

    light_ids = {z.zone_id for z in light_zones}
    dark_ids  = {z.zone_id for z in dark_zones}

    l_time   = sum(z.time_in_s for z in light_zones)
    d_time   = sum(z.time_in_s for z in dark_zones)
    total    = l_time + d_time
    l_entries = sum(z.entries for z in light_zones)
    d_entries = sum(z.entries for z in dark_zones)

    # Sort events by entry time
    sorted_evs = sorted(zone_events, key=lambda e: e.entry_t)
    latency_light: float | None = next(
        (e.entry_t for e in sorted_evs if e.zone_id in light_ids), None)
    latency_dark: float | None = next(
        (e.entry_t for e in sorted_evs if e.zone_id in dark_ids), None)

    # Transitions = alternations between light/dark compartments
    transitions = 0
    prev_compartment: str | None = None
    for ev in sorted_evs:
        if ev.zone_id in light_ids:
            cur = "light"
        elif ev.zone_id in dark_ids:
            cur = "dark"
        else:
            continue
        if prev_compartment is not None and cur != prev_compartment:
            transitions += 1
        prev_compartment = cur

    return LightDarkMetrics(
        light_time_s=round(l_time, 3),
        dark_time_s=round(d_time, 3),
        light_time_pct=round(l_time / total * 100, 2) if total > 0 else 0.0,
        dark_time_pct=round(d_time / total * 100, 2) if total > 0 else 0.0,
        light_entries=l_entries,
        dark_entries=d_entries,
        latency_to_light_s=round(latency_light, 3) if latency_light is not None else None,
        latency_to_dark_s=round(latency_dark, 3) if latency_dark is not None else None,
        transitions=transitions,
    )


# ---------------------------------------------------------------------------
# Fear conditioning helper
# ---------------------------------------------------------------------------
def _compute_fear_cond_metrics(
    raw_frames: list[dict[str, Any]],
    fps: float,
    px_per_cm: float,
    zone_events: list[ZoneEvent],
) -> FearCondMetrics | None:
    """
    Detect fear conditioning epochs from zone names containing:
    'cs', 'us', 'tone', 'shock', 'iti', 'baseline', 'pre', 'recall'.
    Returns None if none found.
    """
    epoch_kws = ("cs", "us", "tone", "shock", "iti", "baseline",
                 "pre-cs", "post-cs", "recall", "extinction")

    cs_events = [e for e in zone_events
                 if any(kw in e.zone_name.lower() for kw in epoch_kws)]
    if not cs_events:
        return None

    dt = 1.0 / fps if fps > 0 else 1.0

    def _freezing_in_window(t_start: float, t_end: float) -> tuple[float, int]:
        """Return (freeze_s, n_episodes) in a time window."""
        window = [f for f in raw_frames
                  if f.get("ok") and t_start <= float(f.get("t_sec", 0)) <= t_end]
        if not window:
            return 0.0, 0
        frz_s, frz_eps = _episode_stats(
            window, fps,
            lambda f: (
                float(f["speed_cm_s"]) < FREEZING_THRESHOLD_CM_S
                if (px_per_cm > 0 and f.get("speed_cm_s") is not None)
                else (float(f["speed_px_s"]) < FREEZING_THRESHOLD_PX_S
                      if f.get("speed_px_s") is not None else False)
            ),
            FREEZING_MIN_DURATION_S,
        )
        return frz_s, frz_eps

    epochs_out: list[FearConditioningEpoch] = []
    baseline_pct: float | None = None
    cs_pcts: list[float] = []

    for ev in sorted(cs_events, key=lambda e: e.entry_t):
        frz_s, frz_eps = _freezing_in_window(ev.entry_t, ev.exit_t)
        dur = ev.duration_s if ev.duration_s > 0 else 0.001
        pct = round(frz_s / dur * 100, 2)

        label_lower = ev.zone_name.lower()
        epoch = FearConditioningEpoch(
            label=ev.zone_name,
            start_t=round(ev.entry_t, 3),
            end_t=round(ev.exit_t, 3),
            duration_s=round(dur, 3),
            freezing_s=round(frz_s, 3),
            freezing_pct=pct,
            freezing_episodes=frz_eps,
        )
        epochs_out.append(epoch)

        if any(kw in label_lower for kw in ("baseline", "pre", "iti")):
            baseline_pct = pct
        elif any(kw in label_lower for kw in ("cs", "tone", "recall")):
            cs_pcts.append(pct)

    mean_cs = round(sum(cs_pcts) / len(cs_pcts), 2) if cs_pcts else None

    return FearCondMetrics(
        epochs=epochs_out,
        baseline_freezing_pct=baseline_pct,
        mean_cs_freezing_pct=mean_cs,
    )


# ---------------------------------------------------------------------------
# Freeze bout list helper
# ---------------------------------------------------------------------------
def _compute_freeze_bouts(
    raw_frames: list[dict[str, Any]],
    fps: float,
    px_per_cm: float,
    eff_freeze_cm: float,
    eff_freeze_px: float,
) -> list[FreezeBout]:
    """Return a list of individual freezing bouts with metadata."""
    bouts: list[FreezeBout] = []
    in_bout = False
    bout_start: float = 0.0
    bout_frames: list[dict[str, Any]] = []

    def _is_frz(f: dict[str, Any]) -> bool:
        if f.get("speed_cm_s") is not None:
            return float(f["speed_cm_s"]) < eff_freeze_cm
        if f.get("speed_px_s") is not None:
            return float(f["speed_px_s"]) < eff_freeze_px
        return False

    for f in raw_frames:
        if not f.get("ok") or not f.get("centroid"):
            if in_bout and bout_frames:
                dur = float(bout_frames[-1]["t_sec"]) - bout_start
                if dur >= FREEZING_MIN_DURATION_S:
                    spd = [float(b["speed_cm_s"]) for b in bout_frames
                           if b.get("speed_cm_s") is not None]
                    mean_spd = round(sum(spd) / len(spd), 3) if spd else None
                    bouts.append(FreezeBout(
                        start_t=round(bout_start, 3),
                        end_t=round(float(bout_frames[-1]["t_sec"]), 3),
                        duration_s=round(dur, 3),
                        mean_speed_cm_s=mean_spd,
                        zone_id=bout_frames[0].get("zone_id"),
                    ))
                in_bout = False
                bout_frames = []
            continue

        frz = _is_frz(f)
        t = float(f["t_sec"])

        if frz and not in_bout:
            in_bout = True
            bout_start = t
            bout_frames = [f]
        elif frz and in_bout:
            bout_frames.append(f)
        elif not frz and in_bout:
            dur = t - bout_start
            if dur >= FREEZING_MIN_DURATION_S:
                spd = [float(b["speed_cm_s"]) for b in bout_frames
                       if b.get("speed_cm_s") is not None]
                mean_spd = round(sum(spd) / len(spd), 3) if spd else None
                bouts.append(FreezeBout(
                    start_t=round(bout_start, 3),
                    end_t=round(t, 3),
                    duration_s=round(dur, 3),
                    mean_speed_cm_s=mean_spd,
                    zone_id=bout_frames[0].get("zone_id"),
                ))
            in_bout = False
            bout_frames = []

    # Close last bout
    if in_bout and bout_frames:
        dur = float(bout_frames[-1]["t_sec"]) - bout_start
        if dur >= FREEZING_MIN_DURATION_S:
            spd = [float(b["speed_cm_s"]) for b in bout_frames
                   if b.get("speed_cm_s") is not None]
            mean_spd = round(sum(spd) / len(spd), 3) if spd else None
            bouts.append(FreezeBout(
                start_t=round(bout_start, 3),
                end_t=round(float(bout_frames[-1]["t_sec"]), 3),
                duration_s=round(dur, 3),
                mean_speed_cm_s=mean_spd,
                zone_id=bout_frames[0].get("zone_id"),
            ))

    return bouts


# ---------------------------------------------------------------------------
# Rearing per zone helper
# ---------------------------------------------------------------------------
def _compute_rearing_per_zone(
    raw_frames: list[dict[str, Any]],
    zones: list[dict[str, Any]],
) -> dict[str, int]:
    """Count rearing frames per zone."""
    counts: dict[str, int] = {z["id"]: 0 for z in zones}
    for f in raw_frames:
        if f.get("rearing") and f.get("zone_id") and f["zone_id"] in counts:
            counts[f["zone_id"]] += 1
    # Convert frame counts to bouts (approximate: divide by min 3 frames per bout)
    return {k: max(1, v // 3) for k, v in counts.items() if v > 0}


def _is_immobile(f: dict[str, Any], px_per_cm: float) -> bool:
    if px_per_cm > 0 and f.get("speed_cm_s") is not None:
        return float(f["speed_cm_s"]) < IMMOBILITY_THRESHOLD_CM_S
    if f.get("speed_px_s") is not None:
        return float(f["speed_px_s"]) < IMMOBILITY_THRESHOLD_PX_S
    return False


def _is_freezing(f: dict[str, Any], px_per_cm: float) -> bool:
    if px_per_cm > 0 and f.get("speed_cm_s") is not None:
        return float(f["speed_cm_s"]) < FREEZING_THRESHOLD_CM_S
    if f.get("speed_px_s") is not None:
        return float(f["speed_px_s"]) < FREEZING_THRESHOLD_PX_S
    return False


# ---------------------------------------------------------------------------
# CSV generation
# ---------------------------------------------------------------------------

_CANONICAL_KP_NAMES = ["nose", "left_ear", "right_ear", "neck",
                       "mid_spine", "hips", "tail_base"]


def to_per_frame_csv(raw_frames: list[dict[str, Any]], px_per_cm: float) -> str:
    # Detect whether any frame has canonical keypoints
    has_pose = any(f.get("canonical_kps") for f in raw_frames)

    buf = io.StringIO()
    w = csv.writer(buf)

    base_header = ["frame", "t_sec", "x_px", "y_px", "x_cm", "y_cm",
                   "zone_id", "speed_px_s", "speed_cm_s",
                   "heading_deg", "body_length_px", "rearing", "grooming",
                   "ear_span_px", "spine_curvature", "head_body_angle_deg", "quality"]
    kp_header: list[str] = []
    if has_pose:
        for kp in _CANONICAL_KP_NAMES:
            kp_header += [f"{kp}_x", f"{kp}_y", f"{kp}_likelihood"]

    w.writerow(base_header + kp_header)

    for f in raw_frames:
        c = f.get("centroid") or {}
        cx = c.get("x") if isinstance(c, dict) else None
        cy = c.get("y") if isinstance(c, dict) else None
        x_px = round(cx, 2) if cx is not None else ""
        y_px = round(cy, 2) if cy is not None else ""
        x_cm = round(cx / px_per_cm, 3) if (cx is not None and px_per_cm > 0) else ""
        y_cm = round(cy / px_per_cm, 3) if (cy is not None and px_per_cm > 0) else ""
        spx  = round(f["speed_px_s"], 3) if f.get("speed_px_s") is not None else ""
        scm  = round(f["speed_cm_s"], 3) if f.get("speed_cm_s") is not None else ""
        hdg  = round(f["heading_deg"], 2) if f.get("heading_deg") is not None else ""
        bl   = round(f["body_length_px"], 2) if f.get("body_length_px") is not None else ""
        es   = round(f["ear_span_px"], 2) if f.get("ear_span_px") is not None else ""
        sc   = round(f["spine_curvature"], 4) if f.get("spine_curvature") is not None else ""
        hb   = round(f["head_body_angle_deg"], 2) if f.get("head_body_angle_deg") is not None else ""

        row: list[Any] = [
            f.get("frame_index", ""),
            round(f.get("t_sec", 0), 4),
            x_px, y_px, x_cm, y_cm,
            f.get("zone_id") or "",
            spx, scm, hdg, bl,
            int(bool(f.get("rearing"))),
            int(bool(f.get("grooming"))),
            es, sc, hb,
            f.get("quality", ""),
        ]

        if has_pose:
            ckps = f.get("canonical_kps") or {}
            for kp in _CANONICAL_KP_NAMES:
                kd = ckps.get(kp)
                if kd:
                    row += [round(kd.get("x", ""), 2), round(kd.get("y", ""), 2),
                            round(kd.get("likelihood", 1.0), 3)]
                else:
                    row += ["", "", ""]

        w.writerow(row)
    return buf.getvalue()


def to_summary_csv(metrics: BehaviorMetrics, zone_names: dict[str, str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)

    w.writerow(["metric", "value", "unit"])
    w.writerow(["duration", metrics.duration_s, "s"])
    w.writerow(["tracked_fraction", metrics.valid_fraction, ""])
    w.writerow(["total_distance_px", metrics.total_distance_px, "px"])
    w.writerow(["total_distance_cm", metrics.total_distance_cm if metrics.total_distance_cm is not None else "", "cm"])
    w.writerow(["mean_speed_px_s", metrics.mean_speed_px_s, "px/s"])
    w.writerow(["mean_speed_cm_s", metrics.mean_speed_cm_s if metrics.mean_speed_cm_s is not None else "", "cm/s"])
    w.writerow(["max_speed_px_s", metrics.max_speed_px_s, "px/s"])
    w.writerow(["max_speed_cm_s", metrics.max_speed_cm_s if metrics.max_speed_cm_s is not None else "", "cm/s"])
    w.writerow(["time_mobile_s", metrics.total_time_mobile_s, "s"])
    w.writerow(["time_immobile_s", metrics.total_time_immobile_s, "s"])
    w.writerow(["time_freezing_s", metrics.total_time_freezing_s, "s"])
    w.writerow(["freezing_episodes", metrics.freezing_episodes, ""])
    w.writerow(["thigmotaxis_fraction", metrics.thigmotaxis_fraction if metrics.thigmotaxis_fraction is not None else "", ""])

    if metrics.zones:
        w.writerow([])
        w.writerow(["zone_id", "zone_name", "time_in_s", "entries",
                    "latency_first_entry_s", "mean_speed_cm_s", "distance_cm"])
        for z in metrics.zones:
            w.writerow([
                z.zone_id,
                z.zone_name,
                z.time_in_s,
                z.entries,
                z.latency_first_entry_s if z.latency_first_entry_s is not None else "",
                z.mean_speed_in_zone_cm_s if z.mean_speed_in_zone_cm_s is not None else "",
                z.distance_in_zone_cm if z.distance_in_zone_cm is not None else "",
            ])

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Morris Water Maze helper
# ---------------------------------------------------------------------------
def _compute_mwm_metrics(
    raw_frames: list[dict[str, Any]],
    fps: float,
    px_per_cm: float,
    zone_metrics: list[ZoneMetrics],
    duration_s: float,
) -> MWMMetrics | None:
    """
    Detect MWM from zone names containing 'quadrant', 'target', 'platform', 'goal'.
    If 4 quadrant zones are present, computes occupancy %.
    If a platform/goal zone exists, tracks proximity.
    Returns None if no MWM-related zones are found.
    """
    quad_kws = ("quadrant", "q1", "q2", "q3", "q4", "nw", "ne", "sw", "se",
                "north", "south", "east", "west")
    plat_kws = ("platform", "goal", "target", "escape")

    quadrant_zones = [z for z in zone_metrics
                      if any(kw in z.zone_name.lower() for kw in quad_kws)]
    platform_zones = [z for z in zone_metrics
                      if any(kw in z.zone_name.lower() for kw in plat_kws)]

    # Need at least one relevant zone to proceed
    if not quadrant_zones and not platform_zones:
        return None

    total_zone_time = sum(z.time_in_s for z in quadrant_zones)
    if total_zone_time <= 0:
        total_zone_time = max(duration_s, 0.001)

    def q_pct(rank: int) -> float:
        """Return occupancy % for the nth most-visited quadrant (0-indexed)."""
        sorted_q = sorted(quadrant_zones, key=lambda z: z.time_in_s, reverse=True)
        if len(sorted_q) > rank:
            return round(sorted_q[rank].time_in_s / total_zone_time * 100, 2)
        return 0.0

    target_pct = q_pct(0)
    opposite_pct = q_pct(2) if len(quadrant_zones) >= 3 else q_pct(1)
    left_pct = q_pct(1) if len(quadrant_zones) >= 2 else 0.0
    right_pct = q_pct(3) if len(quadrant_zones) >= 4 else 0.0

    # Per-trial escape latency (from time bins: latency decreases across trials)
    trial_latencies: list[float] = []
    if platform_zones:
        plat = platform_zones[0]
        if plat.latency_first_entry_s is not None:
            trial_latencies = [plat.latency_first_entry_s]

    # Platform proximity (mean distance to platform centroid)
    plat_zone_id: str | None = platform_zones[0].zone_id if platform_zones else None
    plat_prox: float | None = None
    if plat_zone_id and px_per_cm > 0:
        prox_vals = [f.get("dist_to_platform_px") for f in raw_frames
                     if f.get("dist_to_platform_px") is not None]
        if prox_vals:
            plat_prox = round(sum(prox_vals) / len(prox_vals) / px_per_cm, 2)

    return MWMMetrics(
        target_quadrant_pct=target_pct,
        opposite_quadrant_pct=opposite_pct,
        left_quadrant_pct=left_pct,
        right_quadrant_pct=right_pct,
        trial_escape_latency_s=trial_latencies,
        platform_zone_id=plat_zone_id,
        platform_proximity_mean_cm=plat_prox,
    )


# ---------------------------------------------------------------------------
# Y-maze helper
# ---------------------------------------------------------------------------
def _compute_ymaze_metrics(
    zone_metrics: list[ZoneMetrics],
    zone_events: list[ZoneEvent],
) -> YMazeMetrics | None:
    """
    Detect Y-maze zones: any 3 zones named arm / A / B / C / left / right / centre.
    Computes spontaneous alternation % from the sequence of arm visits.
    """
    arm_kws = ("arm", "left", "right", "centre", "center", "novel", " a", " b", " c")
    arm_zones = [z for z in zone_metrics
                 if any(kw in z.zone_name.lower() for kw in arm_kws)]

    if len(arm_zones) < 3:
        return None

    # Sort events by entry time to get the visit sequence
    arm_ids = {z.zone_id for z in arm_zones}
    seq = [ev.zone_id for ev in sorted(zone_events, key=lambda e: e.entry_t)
           if ev.zone_id in arm_ids]

    if len(seq) < 3:
        return None

    # Count alternations (triplets of all-different arms)
    alternations = 0
    for i in range(len(seq) - 2):
        if seq[i] != seq[i+1] and seq[i+1] != seq[i+2] and seq[i] != seq[i+2]:
            alternations += 1

    possible = max(1, len(seq) - 2)
    alt_pct = round(alternations / possible * 100, 2)

    visit_counts = {z.zone_id: z.entries for z in arm_zones}

    return YMazeMetrics(
        arm_entries=len(seq),
        alternations=alternations,
        spontaneous_alternation_pct=alt_pct,
        arm_visit_counts=visit_counts,
    )


# ---------------------------------------------------------------------------
# Open field center/periphery helper
# ---------------------------------------------------------------------------
def _compute_open_field_metrics(
    zone_metrics: list[ZoneMetrics],
    px_per_cm: float,
) -> OpenFieldMetrics | None:
    """
    Detect open-field center/periphery zones by name.
    Center: name contains 'center'/'centre'/'inner'.
    Periphery: name contains 'peripher'/'outer'/'wall'/'thigmo'.
    Returns None if neither is found.
    """
    center_kws = ("center", "centre", "inner", "central")
    periph_kws = ("peripher", "outer", "wall", "thigmo", "border")

    center_zones = [z for z in zone_metrics
                    if any(kw in z.zone_name.lower() for kw in center_kws)]
    periph_zones  = [z for z in zone_metrics
                     if any(kw in z.zone_name.lower() for kw in periph_kws)]

    if not center_zones and not periph_zones:
        return None

    c_time = sum(z.time_in_s for z in center_zones)
    p_time = sum(z.time_in_s for z in periph_zones)
    total = c_time + p_time
    c_entries = sum(z.entries for z in center_zones)

    c_dist_px = sum(z.distance_in_zone_cm or 0 for z in center_zones)
    p_dist_px = sum(z.distance_in_zone_cm or 0 for z in periph_zones)

    return OpenFieldMetrics(
        center_time_s=round(c_time, 3),
        periphery_time_s=round(p_time, 3),
        center_time_pct=round(c_time / total * 100, 2) if total > 0 else 0.0,
        center_entries=c_entries,
        center_distance_cm=round(c_dist_px, 2) if c_dist_px > 0 else None,
        periphery_distance_cm=round(p_dist_px, 2) if p_dist_px > 0 else None,
    )


# ---------------------------------------------------------------------------
# EPM helper
# ---------------------------------------------------------------------------
def _compute_epm_metrics(zone_metrics: list[ZoneMetrics]) -> EPMMetrics | None:
    """
    Auto-detect Elevated Plus Maze zones by name.
    Open arm: zone name contains 'open'.
    Closed arm: zone name contains 'closed' or 'close' (but not 'open').
    Returns None if neither type is found.
    """
    open_zones = [z for z in zone_metrics if "open" in z.zone_name.lower()]
    closed_zones = [z for z in zone_metrics
                    if "closed" in z.zone_name.lower() or
                    ("close" in z.zone_name.lower() and "open" not in z.zone_name.lower())]

    if not open_zones and not closed_zones:
        return None

    open_time = sum(z.time_in_s for z in open_zones)
    closed_time = sum(z.time_in_s for z in closed_zones)
    open_entries = sum(z.entries for z in open_zones)
    closed_entries = sum(z.entries for z in closed_zones)
    total_arm_time = open_time + closed_time
    total_entries = open_entries + closed_entries

    return EPMMetrics(
        open_arm_time_s=round(open_time, 3),
        closed_arm_time_s=round(closed_time, 3),
        open_arm_time_pct=round(open_time / total_arm_time * 100, 2) if total_arm_time > 0 else 0.0,
        open_arm_entries=open_entries,
        closed_arm_entries=closed_entries,
        open_arm_entries_pct=round(open_entries / total_entries * 100, 2) if total_entries > 0 else 0.0,
    )
