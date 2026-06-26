from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_ROOT = Path(__file__).resolve().parent
STREAMING_JSON = DEMO_ROOT / "Assets" / "StreamingAssets" / "poultry_twin_demo_timeline.json"
FALLBACK_JSON = WORKSPACE_ROOT / "May_version" / "mvp_biomarker_state_twin" / "outputs" / "unity_json" / "poultry_twin_demo_timeline.json"

FEATURE_TABLE = WORKSPACE_ROOT / "data" / "processed" / "data_preparation_outputs_0612" / "features" / "processed_multimodal_window_table.csv"
ZONE_FEATURE_TABLE = WORKSPACE_ROOT / "data" / "processed" / "data_preparation_outputs_0612" / "features" / "semantic_zone_video_features.csv"
ZONE_CONFIG_PATH = WORKSPACE_ROOT / "data" / "processed" / "data_preparation_outputs_0612" / "zones" / "semantic_zone_config_room_1.json"
TRACK_ROOT = WORKSPACE_ROOT / "data" / "processed" / "video_behavior" / "Room 1"

ROOM_ID = "room_1"
ROOM_LABEL = "Room 1"
ROOM_BIRD_COUNT = 30
DATE_START = "2025-08-16"
DATE_END_EXCLUSIVE = "2025-08-20"
TIMELINE_END_DATE = "2025-08-19"

BEHAVIOR_ORDER = [
    "active",
    "feeding",
    "drinking",
    "idle",
    "preening",
    "perching",
    "wing_flapping",
]

BIRD_ACTION_SLICE_SECONDS = 10
BIRD_ACTION_PRIORITY_ORDER = [
    "wing_flapping",
    "drinking",
    "feeding",
    "perching",
    "preening",
    "active",
    "idle",
]
BIRD_ACTION_MIN_EVIDENCE = {
    "wing_flapping": 2.0,
    "drinking": 0.5,
    "feeding": 0.5,
    "perching": 0.75,
    "preening": 0.75,
    "active": 0.5,
    "idle": 0.0,
}
BIRD_ACTION_MIN_SECONDS = {
    "wing_flapping": 2,
    "drinking": 1,
    "feeding": 1,
    "perching": 1,
    "preening": 1,
    "active": 1,
    "idle": 0,
}

UNITY_ZONE_DEFINITIONS = [
    {
        "zone_id": "drinker",
        "display_name": "Drinker",
        "row": 0,
        "col": 0,
        "polygon": [
            {"x": 0.19, "y": 0.14},
            {"x": 0.30, "y": 0.14},
            {"x": 0.30, "y": 0.47},
            {"x": 0.19, "y": 0.47},
        ],
    },
    {
        "zone_id": "feeder",
        "display_name": "Feeder",
        "row": 0,
        "col": 1,
        "polygon": [
            {"x": 0.31, "y": 0.29},
            {"x": 0.45, "y": 0.29},
            {"x": 0.45, "y": 0.60},
            {"x": 0.31, "y": 0.60},
        ],
    },
    {
        "zone_id": "resting_floor",
        "display_name": "Resting / Floor",
        "row": 1,
        "col": 1,
        "polygon": [
            {"x": 0.77, "y": 0.08},
            {"x": 0.89, "y": 0.08},
            {"x": 0.89, "y": 0.88},
            {"x": 0.77, "y": 0.88},
        ],
    },
    {
        "zone_id": "open_movement",
        "display_name": "Open Movement",
        "row": 1,
        "col": 0,
        "polygon": [
            {"x": 0.18, "y": 0.56},
            {"x": 0.75, "y": 0.56},
            {"x": 0.75, "y": 0.90},
            {"x": 0.18, "y": 0.90},
        ],
    },
]

RAW_TO_UNITY_ZONES = [
    {
        "raw_zone_id": "drinking_zone",
        "unity_zone_id": "drinker",
        "source_display_name": "Drinker",
        "is_proxy_zone": False,
    },
    {
        "raw_zone_id": "feeding_zone",
        "unity_zone_id": "feeder",
        "source_display_name": "Feeder",
        "is_proxy_zone": False,
    },
    {
        "raw_zone_id": "resting_zone",
        "unity_zone_id": "resting_floor",
        "source_display_name": "Resting / Floor",
        "is_proxy_zone": False,
    },
    {
        "raw_zone_id": "open_movement_zone",
        "unity_zone_id": "open_movement",
        "source_display_name": "Open Movement",
        "is_proxy_zone": False,
    },
]

ZONE_PRIORITY = ["drinking_zone", "feeding_zone", "resting_zone", "open_movement_zone"]
ROOM_WORLD_WIDTH = 10.8
ROOM_WORLD_DEPTH = 16.4
TRACK_SECOND_FILE_NAME = "track_second_behavior_table.csv"


@dataclass(frozen=True)
class WindowRow:
    frame_index: int
    window_id: str
    room_id: str
    session_id: str
    video_path: str
    video_file: str
    video_key: str
    start_dt: datetime
    end_dt: datetime
    local_date: str
    video_start_offset_sec: float
    activity_mean: float
    normalized_activity: float
    mobility_index: float
    spatial_freedom_index: float
    occupancy_imbalance_index: float
    semantic_transition_proxy: float
    drinking_activity_fraction: float
    feeding_activity_fraction: float
    open_movement_activity_fraction: float
    resting_activity_fraction: float
    state_label: str
    welfare_risk_score: float
    event_phase: str
    row: dict[str, str]


@dataclass(frozen=True)
class ZoneMetric:
    video_start_offset_sec: float
    activity_mean: float
    activity_norm: float


@dataclass
class ZoneConfig:
    image_width: int
    image_height: int
    polygons: dict[str, list[list[tuple[float, float]]]]


@dataclass(frozen=True)
class BehaviorObservation:
    second: int
    track_id: str
    behavior: str
    behavior_evidence: dict[str, float]
    confidence: float
    source_zone_id: str
    display_zone_id: str
    x_norm: float
    y_norm: float
    world_x: float
    world_z: float


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default

    text = value.strip()
    if not text:
        return default

    try:
        return float(text)
    except ValueError:
        return default


def parse_int(value: str | None, default: int = 0) -> int:
    if value is None:
        return default

    text = value.strip()
    if not text:
        return default

    try:
        return int(float(text))
    except ValueError:
        return default


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_video_key(session_name: str, video_name: str) -> str:
    return f"{session_name.strip().lower()}|{video_name.strip().lower()}"


def video_key_from_feature_path(video_path: str) -> str:
    path = Path(video_path)
    return normalize_video_key(path.parent.name, path.name)


def video_key_from_track_path(csv_path: Path) -> str:
    session_name = csv_path.parent.parent.name
    video_name = f"{csv_path.parent.name}.MP4"
    return normalize_video_key(session_name, video_name)


def raw_zone_to_unity_zone(raw_zone_id: str) -> str:
    for definition in RAW_TO_UNITY_ZONES:
        if definition["raw_zone_id"] == raw_zone_id:
            return str(definition["unity_zone_id"])
    return "open_movement"


def get_bird_action_second_range(window: WindowRow) -> tuple[int, int]:
    window_start = float(window.video_start_offset_sec)
    duration_seconds = max(1.0, (window.end_dt - window.start_dt).total_seconds())
    if duration_seconds <= BIRD_ACTION_SLICE_SECONDS:
        action_start = window_start
        action_end = window_start + duration_seconds
    else:
        action_start = window_start + ((duration_seconds - BIRD_ACTION_SLICE_SECONDS) * 0.5)
        action_end = action_start + BIRD_ACTION_SLICE_SECONDS

    start_second = int(math.floor(action_start + 1e-9))
    end_second = int(math.ceil(action_end - 1e-9))
    if end_second <= start_second:
        end_second = start_second + 1
    return start_second, end_second


def has_min_action_evidence(behavior_evidence: Counter[str], behavior_seconds: Counter[str], behavior: str) -> bool:
    min_evidence = BIRD_ACTION_MIN_EVIDENCE.get(behavior, 0.0)
    min_seconds = BIRD_ACTION_MIN_SECONDS.get(behavior, 0)
    return behavior_evidence.get(behavior, 0.0) >= min_evidence and behavior_seconds.get(behavior, 0) >= min_seconds


def select_bird_action(
    behavior_evidence: Counter[str],
    behavior_seconds: Counter[str],
    source_zone_id: str,
) -> str:
    for behavior in BIRD_ACTION_PRIORITY_ORDER:
        if has_min_action_evidence(behavior_evidence, behavior_seconds, behavior):
            return behavior

    return "idle"


def load_zone_config() -> ZoneConfig:
    payload = json.loads(ZONE_CONFIG_PATH.read_text(encoding="utf-8"))
    polygons: dict[str, list[list[tuple[float, float]]]] = {}
    for zone in payload.get("zones", []):
        zone_id = zone.get("zone_id")
        if not zone_id:
            continue
        zone_polygons: list[list[tuple[float, float]]] = []
        if zone.get("polygon"):
            normalized_polygon: list[tuple[float, float]] = []
            for point in zone["polygon"]:
                if len(point) < 2:
                    continue
                normalized_polygon.append((float(point[0]), float(point[1])))
            if normalized_polygon:
                zone_polygons.append(normalized_polygon)
        for polygon in zone.get("polygons", []):
            normalized_polygon = []
            for point in polygon:
                if len(point) < 2:
                    continue
                normalized_polygon.append((float(point[0]), float(point[1])))
            if normalized_polygon:
                zone_polygons.append(normalized_polygon)
        if zone_polygons:
            polygons[zone_id] = zone_polygons

    return ZoneConfig(
        image_width=int(payload.get("image_width", 1920)),
        image_height=int(payload.get("image_height", 1080)),
        polygons=polygons,
    )


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < ((xj - xi) * (y - yi) / ((yj - yi) or 1e-9)) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def classify_semantic_zone(center_x: float, center_y: float, zone_config: ZoneConfig) -> str:
    for zone_id in ZONE_PRIORITY:
        polygons = zone_config.polygons.get(zone_id)
        if polygons is None:
            continue
        for polygon in polygons:
            if point_in_polygon(center_x, center_y, polygon):
                return zone_id

    return next(iter(zone_config.polygons.keys()), "")


def classify_unity_zone(x_norm: float, y_norm: float) -> str:
    for zone in UNITY_ZONE_DEFINITIONS:
        polygon = [(point["x"], point["y"]) for point in zone["polygon"]]
        if point_in_polygon(x_norm, y_norm, polygon):
            return zone["zone_id"]

    return "open_movement"


def normalized_to_world(x_norm: float, y_norm: float) -> tuple[float, float]:
    world_x = (clamp01(x_norm) * ROOM_WORLD_WIDTH) - (ROOM_WORLD_WIDTH * 0.5)
    world_z = ((1.0 - clamp01(y_norm)) * ROOM_WORLD_DEPTH) - (ROOM_WORLD_DEPTH * 0.5)
    return (round(world_x, 4), round(world_z, 4))


def compute_risk_score(
    normalized_activity: float,
    spatial_freedom_index: float,
    occupancy_imbalance_index: float,
    drinking_fraction: float,
    feeding_fraction: float,
) -> float:
    low_activity = clamp01(1.0 - normalized_activity)
    clustering = clamp01(1.0 - spatial_freedom_index)
    crowding = clamp01(max(occupancy_imbalance_index, clustering))
    resource_use = clamp01(drinking_fraction + feeding_fraction)
    resource_penalty = clamp01((0.55 - resource_use) / 0.55)
    return round(
        clamp01((0.38 * low_activity) + (0.32 * crowding) + (0.20 * clustering) + (0.10 * resource_penalty)),
        6,
    )


def derive_state_label(
    normalized_activity: float,
    spatial_freedom_index: float,
    occupancy_imbalance_index: float,
    drinking_fraction: float,
    feeding_fraction: float,
    open_movement_fraction: float,
    resting_fraction: float,
) -> str:
    clustering = clamp01(1.0 - spatial_freedom_index)
    if normalized_activity <= 0.16:
        return "localized_low_activity"
    if occupancy_imbalance_index >= 0.60 or clustering >= 0.60:
        return "crowding_proxy"
    if feeding_fraction >= max(drinking_fraction, open_movement_fraction, resting_fraction) and feeding_fraction >= 0.20:
        return "feeder_focused_activity"
    if drinking_fraction >= max(feeding_fraction, open_movement_fraction, resting_fraction) and drinking_fraction >= 0.14:
        return "drinker_focused_activity"
    if resting_fraction >= max(open_movement_fraction, feeding_fraction, drinking_fraction) and resting_fraction >= 0.38:
        return "resting_zone_dominant"
    if open_movement_fraction >= max(resting_fraction, feeding_fraction, drinking_fraction) and open_movement_fraction >= 0.30:
        return "open_movement_dominant"
    return "balanced_room_activity"


def derive_event_phase(risk_score: float) -> str:
    if risk_score >= 0.66:
        return "during"
    if risk_score >= 0.48:
        return "recovery"
    return "background"


def load_zone_feature_lookup() -> dict[str, dict[str, ZoneMetric]]:
    lookup: dict[str, dict[str, ZoneMetric]] = defaultdict(dict)
    with ZONE_FEATURE_TABLE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            room_id = (raw.get("room_id") or "").strip()
            start_time = (raw.get("start_time") or "").strip()
            if room_id != ROOM_ID or not (DATE_START <= start_time[:10] < DATE_END_EXCLUSIVE):
                continue

            window_id = (raw.get("window_id") or "").strip()
            zone_id = (raw.get("zone_id") or "").strip()
            if not window_id or not zone_id:
                continue

            lookup[window_id][zone_id] = ZoneMetric(
                video_start_offset_sec=parse_float(raw.get("video_start_offset_sec")),
                activity_mean=parse_float(raw.get("activity_mean")),
                activity_norm=parse_float(raw.get("activity_norm")),
            )

    return lookup


def load_window_rows(zone_feature_lookup: dict[str, dict[str, ZoneMetric]]) -> list[WindowRow]:
    rows: list[WindowRow] = []
    with FEATURE_TABLE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            room_id = (raw.get("room_id") or "").strip()
            start_time = (raw.get("start_time") or "").strip()
            if room_id != ROOM_ID or not (DATE_START <= start_time[:10] < DATE_END_EXCLUSIVE):
                continue

            window_id = (raw.get("window_id") or f"frame_{len(rows):05d}").strip()
            end_time = (raw.get("end_time") or "").strip()
            local_date = (raw.get("local_date") or start_time[:10]).strip()
            video_path = (raw.get("video_path") or "").strip()

            zone_metrics = zone_feature_lookup.get(window_id, {})
            video_offset_sec = 0.0
            if zone_metrics:
                video_offset_sec = next(iter(zone_metrics.values())).video_start_offset_sec

            normalized_activity = parse_float(raw.get("normalized_activity"))
            spatial_freedom_index = parse_float(raw.get("spatial_freedom_index"))
            occupancy_imbalance_index = parse_float(raw.get("occupancy_imbalance_index"))
            drinking_fraction = parse_float(raw.get("drinking_activity_fraction"))
            feeding_fraction = parse_float(raw.get("feeding_activity_fraction"))
            open_movement_fraction = parse_float(raw.get("open_movement_activity_fraction"))
            resting_fraction = parse_float(raw.get("resting_activity_fraction"))
            risk_score = compute_risk_score(
                normalized_activity=normalized_activity,
                spatial_freedom_index=spatial_freedom_index,
                occupancy_imbalance_index=occupancy_imbalance_index,
                drinking_fraction=drinking_fraction,
                feeding_fraction=feeding_fraction,
            )

            rows.append(
                WindowRow(
                    frame_index=0,
                    window_id=window_id,
                    room_id=room_id,
                    session_id=(raw.get("session_id") or "").strip(),
                    video_path=video_path,
                    video_file=Path(video_path).name,
                    video_key=video_key_from_feature_path(video_path),
                    start_dt=datetime.fromisoformat(start_time),
                    end_dt=datetime.fromisoformat(end_time),
                    local_date=local_date,
                    video_start_offset_sec=video_offset_sec,
                    activity_mean=parse_float(raw.get("activity_mean")),
                    normalized_activity=normalized_activity,
                    mobility_index=parse_float(raw.get("mobility_index")),
                    spatial_freedom_index=spatial_freedom_index,
                    occupancy_imbalance_index=occupancy_imbalance_index,
                    semantic_transition_proxy=parse_float(raw.get("semantic_transition_proxy")),
                    drinking_activity_fraction=drinking_fraction,
                    feeding_activity_fraction=feeding_fraction,
                    open_movement_activity_fraction=open_movement_fraction,
                    resting_activity_fraction=resting_fraction,
                    state_label=derive_state_label(
                        normalized_activity=normalized_activity,
                        spatial_freedom_index=spatial_freedom_index,
                        occupancy_imbalance_index=occupancy_imbalance_index,
                        drinking_fraction=drinking_fraction,
                        feeding_fraction=feeding_fraction,
                        open_movement_fraction=open_movement_fraction,
                        resting_fraction=resting_fraction,
                    ),
                    welfare_risk_score=risk_score,
                    event_phase=derive_event_phase(risk_score),
                    row=raw,
                )
            )

    rows.sort(key=lambda item: (item.start_dt, item.window_id))
    normalized_rows: list[WindowRow] = []
    for frame_index, row in enumerate(rows):
        normalized_rows.append(
            WindowRow(
                frame_index=frame_index,
                window_id=row.window_id,
                room_id=row.room_id,
                session_id=row.session_id,
                video_path=row.video_path,
                video_file=row.video_file,
                video_key=row.video_key,
                start_dt=row.start_dt,
                end_dt=row.end_dt,
                local_date=row.local_date,
                video_start_offset_sec=row.video_start_offset_sec,
                activity_mean=row.activity_mean,
                normalized_activity=row.normalized_activity,
                mobility_index=row.mobility_index,
                spatial_freedom_index=row.spatial_freedom_index,
                occupancy_imbalance_index=row.occupancy_imbalance_index,
                semantic_transition_proxy=row.semantic_transition_proxy,
                drinking_activity_fraction=row.drinking_activity_fraction,
                feeding_activity_fraction=row.feeding_activity_fraction,
                open_movement_activity_fraction=row.open_movement_activity_fraction,
                resting_activity_fraction=row.resting_activity_fraction,
                state_label=row.state_label,
                welfare_risk_score=row.welfare_risk_score,
                event_phase=row.event_phase,
                row=row.row,
            )
        )

    return normalized_rows


def build_track_file_lookup() -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for csv_path in TRACK_ROOT.rglob(TRACK_SECOND_FILE_NAME):
        lookup[video_key_from_track_path(csv_path)] = csv_path
    return lookup


def load_video_behavior_records(
    csv_path: Path,
    zone_config: ZoneConfig,
) -> dict[int, list[BehaviorObservation]]:
    records_by_second: dict[int, list[BehaviorObservation]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            behavior = (raw.get("dominant_behavior") or "").strip().lower()
            track_id = (raw.get("track_id") or "").strip()
            if not behavior or not track_id:
                continue

            second = parse_int(raw.get("second_offset"))
            center_x = parse_float(raw.get("center_x"))
            center_y = parse_float(raw.get("center_y"))
            x_norm = clamp01(
                parse_float(raw.get("x_norm"), center_x / max(zone_config.image_width, 1))
            )
            y_norm = clamp01(
                parse_float(raw.get("y_norm"), center_y / max(zone_config.image_height, 1))
            )
            source_zone_id = (raw.get("zone_id") or "").strip()
            display_zone_id = raw_zone_to_unity_zone(source_zone_id)
            fallback_world_x, fallback_world_z = normalized_to_world(x_norm, y_norm)
            world_x = parse_float(raw.get("world_x"), fallback_world_x)
            world_z = parse_float(raw.get("world_z"), fallback_world_z)
            behavior_evidence: dict[str, float] = {}
            for behavior_name in BEHAVIOR_ORDER:
                evidence = parse_float(raw.get(f"{behavior_name}_share"))
                if evidence > 0.0:
                    behavior_evidence[behavior_name] = evidence
            if not behavior_evidence and behavior:
                behavior_evidence[behavior] = 1.0

            records_by_second[second].append(
                BehaviorObservation(
                    second=second,
                    track_id=track_id,
                    behavior=behavior,
                    behavior_evidence=behavior_evidence,
                    confidence=parse_float(raw.get("mean_confidence")),
                    source_zone_id=source_zone_id,
                    display_zone_id=display_zone_id,
                    x_norm=round(x_norm, 6),
                    y_norm=round(y_norm, 6),
                    world_x=round(world_x, 4),
                    world_z=round(world_z, 4),
                )
            )

    return records_by_second


def empty_behavior_summary(observed_seconds: int) -> dict[str, Any]:
    return {
        "available": False,
        "total_records": 0,
        "unique_tracks": 0,
        "observed_seconds": max(1, observed_seconds),
        "mean_detected_birds": 0.0,
        "mean_confidence": 0.0,
        "dominant_behaviour": "unavailable",
        "behavior_mix": [],
    }


def build_behavior_mix(counter: Counter[str], total: int) -> list[dict[str, Any]]:
    labels = list(BEHAVIOR_ORDER)
    extras = sorted(label for label in counter.keys() if label not in labels)
    labels.extend(extras)

    mix: list[dict[str, Any]] = []
    for label in labels:
        count = counter.get(label, 0)
        if total <= 0 and count <= 0:
            continue

        mix.append(
            {
                "behavior_id": label,
                "label": label.replace("_", " ").title(),
                "value": round((count / total) if total > 0 else 0.0, 6),
                "count": int(count),
            }
        )

    return mix


def build_behavior_summary(
    counter: Counter[str],
    total_records: int,
    unique_tracks: int,
    mean_detected_birds: float,
    mean_confidence: float,
    observed_seconds: int,
) -> dict[str, Any]:
    dominant = "unavailable"
    if total_records > 0 and counter:
        dominant = counter.most_common(1)[0][0]

    return {
        "available": total_records > 0,
        "total_records": int(total_records),
        "unique_tracks": int(unique_tracks),
        "observed_seconds": int(max(1, observed_seconds)),
        "mean_detected_birds": round(mean_detected_birds, 3),
        "mean_confidence": round(mean_confidence, 6),
        "dominant_behaviour": dominant,
        "behavior_mix": build_behavior_mix(counter, total_records),
    }


def update_track_snapshot(
    track_snapshots: dict[str, dict[str, Any]],
    observation: BehaviorObservation,
) -> None:
    snapshot = track_snapshots.setdefault(
        observation.track_id,
        {
            "track_id": observation.track_id,
            "count": 0,
            "confidence_sum": 0.0,
            "x_norm_sum": 0.0,
            "y_norm_sum": 0.0,
            "world_x_sum": 0.0,
            "world_z_sum": 0.0,
            "behavior_counts": Counter(),
            "behavior_evidence": Counter(),
            "behavior_seconds": Counter(),
            "display_zone_counts": Counter(),
            "source_zone_counts": Counter(),
            "latest_second": -1,
            "latest_x_norm": 0.0,
            "latest_y_norm": 0.0,
            "latest_world_x": 0.0,
            "latest_world_z": 0.0,
            "latest_behavior": "",
            "latest_display_zone": "",
            "latest_source_zone": "",
        },
    )
    snapshot["count"] += 1
    snapshot["confidence_sum"] += observation.confidence
    snapshot["x_norm_sum"] += observation.x_norm
    snapshot["y_norm_sum"] += observation.y_norm
    snapshot["world_x_sum"] += observation.world_x
    snapshot["world_z_sum"] += observation.world_z
    snapshot["behavior_counts"][observation.behavior] += 1
    for behavior_name, evidence in observation.behavior_evidence.items():
        snapshot["behavior_evidence"][behavior_name] += evidence
        if evidence > 0.0:
            snapshot["behavior_seconds"][behavior_name] += 1
    snapshot["display_zone_counts"][observation.display_zone_id] += 1
    if observation.source_zone_id:
        snapshot["source_zone_counts"][observation.source_zone_id] += 1
    if observation.second >= snapshot["latest_second"]:
        snapshot["latest_second"] = observation.second
        snapshot["latest_x_norm"] = observation.x_norm
        snapshot["latest_y_norm"] = observation.y_norm
        snapshot["latest_world_x"] = observation.world_x
        snapshot["latest_world_z"] = observation.world_z
        snapshot["latest_behavior"] = observation.behavior
        snapshot["latest_display_zone"] = observation.display_zone_id
        snapshot["latest_source_zone"] = observation.source_zone_id


def build_window_behavior_payload(
    records_by_second: dict[int, list[BehaviorObservation]],
    window: WindowRow,
) -> dict[str, Any]:
    window_start_second = int(math.floor(window.video_start_offset_sec))
    duration_seconds = max(1.0, (window.end_dt - window.start_dt).total_seconds())
    window_end_second = int(math.ceil(window.video_start_offset_sec + duration_seconds))
    observed_seconds = max(1, window_end_second - window_start_second)
    action_start_second, action_end_second = get_bird_action_second_range(window)

    whole_counts: Counter[str] = Counter()
    whole_tracks_union: set[str] = set()
    whole_confidence_sum = 0.0
    zone_counts: dict[str, Counter[str]] = defaultdict(Counter)
    zone_tracks_union: dict[str, set[str]] = defaultdict(set)
    zone_confidence_sum: dict[str, float] = defaultdict(float)
    total_whole_track_count = 0.0
    total_zone_track_count: dict[str, float] = defaultdict(float)
    track_snapshots: dict[str, dict[str, Any]] = {}
    bird_action_snapshots: dict[str, dict[str, Any]] = {}

    for second in range(window_start_second, window_end_second):
        observations = records_by_second.get(second)
        if not observations:
            continue

        second_tracks: set[str] = set()
        second_zone_tracks: dict[str, set[str]] = defaultdict(set)

        for observation in observations:
            whole_counts[observation.behavior] += 1
            whole_tracks_union.add(observation.track_id)
            whole_confidence_sum += observation.confidence
            second_tracks.add(observation.track_id)

            if observation.source_zone_id:
                zone_counts[observation.source_zone_id][observation.behavior] += 1
                zone_tracks_union[observation.source_zone_id].add(observation.track_id)
                zone_confidence_sum[observation.source_zone_id] += observation.confidence
                second_zone_tracks[observation.source_zone_id].add(observation.track_id)

            update_track_snapshot(track_snapshots, observation)
            if action_start_second <= second < action_end_second:
                update_track_snapshot(bird_action_snapshots, observation)

        total_whole_track_count += len(second_tracks)
        for zone_id, tracks in second_zone_tracks.items():
            total_zone_track_count[zone_id] += len(tracks)

    whole_records = sum(whole_counts.values())
    mean_whole_birds = total_whole_track_count / observed_seconds
    whole_summary = build_behavior_summary(
        counter=whole_counts,
        total_records=whole_records,
        unique_tracks=len(whole_tracks_union),
        mean_detected_birds=mean_whole_birds,
        mean_confidence=(whole_confidence_sum / whole_records) if whole_records > 0 else 0.0,
        observed_seconds=observed_seconds,
    )

    zones: dict[str, Any] = {}
    for raw_zone_id in [str(definition["raw_zone_id"]) for definition in RAW_TO_UNITY_ZONES]:
        counter = zone_counts.get(raw_zone_id, Counter())
        total_records = sum(counter.values())
        mean_zone_birds = total_zone_track_count.get(raw_zone_id, 0.0) / observed_seconds
        occupancy_share = (mean_zone_birds / mean_whole_birds) if mean_whole_birds > 0 else 0.0
        zones[raw_zone_id] = {
            "occupancy_share": round(occupancy_share, 6),
            "mean_detected_birds": round(mean_zone_birds, 3),
            "behavior_summary": build_behavior_summary(
                counter=counter,
                total_records=total_records,
                unique_tracks=len(zone_tracks_union.get(raw_zone_id, set())),
                mean_detected_birds=mean_zone_birds,
                mean_confidence=(zone_confidence_sum.get(raw_zone_id, 0.0) / total_records) if total_records > 0 else 0.0,
                observed_seconds=observed_seconds,
            ),
        }

    bird_snapshot_source = bird_action_snapshots if bird_action_snapshots else track_snapshots
    selected_tracks = sorted(
        bird_snapshot_source.values(),
        key=lambda item: (-item["count"], -item["confidence_sum"], parse_int(item["track_id"], 10_000_000)),
    )[:ROOM_BIRD_COUNT]
    selected_tracks.sort(key=lambda item: parse_int(item["track_id"], 10_000_000))

    birds: list[dict[str, Any]] = []
    for index, snapshot in enumerate(selected_tracks):
        count = max(1, int(snapshot["count"]))
        dominant_display_zone = snapshot["latest_display_zone"] or snapshot["display_zone_counts"].most_common(1)[0][0]
        source_zone_id = snapshot["latest_source_zone"]
        if not source_zone_id and snapshot["source_zone_counts"]:
            source_zone_id = snapshot["source_zone_counts"].most_common(1)[0][0]
        dominant_behavior = select_bird_action(
            snapshot["behavior_evidence"],
            snapshot["behavior_seconds"],
            source_zone_id,
        )

        birds.append(
            {
                "bird_id": f"bird_{index + 1:02d}",
                "track_id": snapshot["track_id"],
                "zone_id": dominant_display_zone,
                "source_zone_id": source_zone_id,
                "behavior": dominant_behavior,
                "confidence": round(snapshot["confidence_sum"] / count, 6),
                "x_norm": round(snapshot["latest_x_norm"], 6),
                "y_norm": round(snapshot["latest_y_norm"], 6),
                "world_x": round(snapshot["latest_world_x"], 4),
                "world_z": round(snapshot["latest_world_z"], 4),
                "observation_count": count,
            }
        )

    return {
        "whole_summary": whole_summary,
        "zones": zones,
        "birds": birds,
    }


def state_id_map(rows: list[WindowRow]) -> dict[str, int]:
    unique_labels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.state_label in seen:
            continue
        seen.add(row.state_label)
        unique_labels.append(row.state_label)
    return {label: index for index, label in enumerate(unique_labels)}


def risk_level_from_score(score: float) -> str:
    if score >= 0.66:
        return "high"
    if score >= 0.33:
        return "medium"
    return "low"


def build_zone_frames(
    window: WindowRow,
    zone_metrics_lookup: dict[str, dict[str, ZoneMetric]],
    behavior_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    zone_fractions = {
        "drinking_zone": window.drinking_activity_fraction,
        "feeding_zone": window.feeding_activity_fraction,
        "open_movement_zone": window.open_movement_activity_fraction,
        "resting_zone": window.resting_activity_fraction,
    }

    raw_zone_metrics = zone_metrics_lookup.get(window.window_id, {})
    frames: list[dict[str, Any]] = []
    for zone_definition in RAW_TO_UNITY_ZONES:
        raw_zone_id = zone_definition["raw_zone_id"]
        unity_zone_id = zone_definition["unity_zone_id"]
        source_display_name = zone_definition["source_display_name"]
        is_proxy = zone_definition["is_proxy_zone"]

        raw_metric = raw_zone_metrics.get(raw_zone_id) if raw_zone_id is not None else None
        zone_activity = raw_metric.activity_mean if raw_metric is not None else 0.0
        zone_activity_norm = raw_metric.activity_norm if raw_metric is not None else 0.0
        occupancy_share = zone_fractions.get(raw_zone_id, 0.0)
        mean_detected_birds = round(occupancy_share * ROOM_BIRD_COUNT, 3)
        behavior_summary = empty_behavior_summary(int(max(1.0, (window.end_dt - window.start_dt).total_seconds())))

        if behavior_payload is not None and raw_zone_id is not None:
            zone_payload = behavior_payload["zones"].get(raw_zone_id, {})
            occupancy_share = zone_payload.get("occupancy_share", occupancy_share)
            mean_detected_birds = zone_payload.get("mean_detected_birds", mean_detected_birds)
            behavior_summary = zone_payload.get("behavior_summary", behavior_summary)

        frames.append(
            {
                "zone_id": unity_zone_id,
                "display_name": unity_zone_id.replace("_", " ").title(),
                "source_zone_id": raw_zone_id,
                "source_display_name": source_display_name,
                "is_proxy_zone": is_proxy,
                "activity": round(zone_activity, 8),
                "activity_norm": round(zone_activity_norm, 8),
                "overlay_intensity": round(zone_activity_norm, 8),
                "occupancy_share": round(occupancy_share, 6),
                "bird_count_mean": round(mean_detected_birds, 3),
                "behavior_summary": behavior_summary,
            }
        )

    return frames


def build_frame_payload(
    row: WindowRow,
    zone_metrics_lookup: dict[str, dict[str, ZoneMetric]],
    behavior_payload: dict[str, Any] | None,
    states: dict[str, int],
) -> dict[str, Any]:
    observed_seconds = int(max(1.0, (row.end_dt - row.start_dt).total_seconds()))
    whole_behavior = behavior_payload["whole_summary"] if behavior_payload is not None else empty_behavior_summary(observed_seconds)
    bird_payload = behavior_payload["birds"] if behavior_payload is not None else []

    return {
        "frame_index": row.frame_index,
        "window_id": row.window_id,
        "room_id": row.room_id,
        "room_label": ROOM_LABEL,
        "bird_count": ROOM_BIRD_COUNT,
        "start_time": row.start_dt.isoformat(),
        "end_time": row.end_dt.isoformat(),
        "local_date": row.local_date,
        "metrics": {
            "mobility_index": round(row.mobility_index, 8),
            "spatial_freedom_index": round(row.spatial_freedom_index, 8),
            "occupancy_imbalance_index": round(row.occupancy_imbalance_index, 8),
            "activity_mean": round(row.activity_mean, 8),
            "normalized_activity": round(row.normalized_activity, 8),
        },
        "state": {
            "state_id": states[row.state_label],
            "state_label": row.state_label,
            "state_probability": 1.0,
        },
        "welfare": {
            "risk_score": round(row.welfare_risk_score, 8),
            "risk_level": risk_level_from_score(row.welfare_risk_score),
            "sustained_risk_flag": row.welfare_risk_score >= 0.66,
        },
        "zones": build_zone_frames(
            window=row,
            zone_metrics_lookup=zone_metrics_lookup,
            behavior_payload=behavior_payload,
        ),
        "event": {
            "event_id": f"{row.local_date}_{row.window_id}",
            "event_phase": row.event_phase,
            "event_type": "abnormality_window" if row.welfare_risk_score >= 0.48 else "background_window",
        },
        "behavior_summary": whole_behavior,
        "dashboard_context": {
            "data_zone_schema": "4-zone room twin using 0612 semantic-zone features and second-level flock behavior playback. Risk is a provisional prototype score derived from activity, crowding, and feeder/drinker use.",
            "timeline_date_label": row.local_date,
            "source_video": row.video_file,
        },
        "birds": bird_payload,
    }


def build_timeline_payload(
    rows: list[WindowRow],
    zone_config: ZoneConfig,
    zone_metrics_lookup: dict[str, dict[str, ZoneMetric]],
) -> dict[str, Any]:
    states = state_id_map(rows)
    behavior_lookup = build_track_file_lookup()
    rows_by_video: dict[str, list[WindowRow]] = defaultdict(list)
    for row in rows:
        rows_by_video[row.video_key].append(row)

    frame_lookup: dict[str, dict[str, Any]] = {}
    for video_key, video_rows in rows_by_video.items():
        behavior_path = behavior_lookup.get(video_key)
        records_by_second: dict[int, list[BehaviorObservation]] = {}
        if behavior_path is not None:
            records_by_second = load_video_behavior_records(behavior_path, zone_config)

        for row in video_rows:
            behavior_payload = None
            if records_by_second:
                behavior_payload = build_window_behavior_payload(records_by_second, row)

            frame_lookup[row.window_id] = build_frame_payload(
                row=row,
                zone_metrics_lookup=zone_metrics_lookup,
                behavior_payload=behavior_payload,
                states=states,
            )

    timeline = [frame_lookup[row.window_id] for row in rows]
    dates = sorted({row.local_date for row in rows})
    return {
        "metadata": {
            "schema_version": "poultry_twin_demo_v3",
            "created_at": datetime.now().astimezone().isoformat(),
            "source_feature_table": str(FEATURE_TABLE),
            "source_zone_feature_table": str(ZONE_FEATURE_TABLE),
            "source_zone_config": str(ZONE_CONFIG_PATH),
            "source_behavior_root": str(TRACK_ROOT),
            "notes": "Room 1 Aug 16-19 digital-twin prototype driven by 0612 four-zone 30-s feature windows plus center 10-s bird action playback from second-level behavior evidence. Welfare risk remains a provisional visualization score, not a validated diagnosis.",
            "room_id": ROOM_ID,
            "room_label": ROOM_LABEL,
            "bird_count": ROOM_BIRD_COUNT,
            "bird_action_slice_seconds": BIRD_ACTION_SLICE_SECONDS,
            "bird_action_slice_position": "center",
            "bird_action_priority_order": BIRD_ACTION_PRIORITY_ORDER,
            "bird_action_min_evidence": BIRD_ACTION_MIN_EVIDENCE,
            "bird_action_min_seconds": BIRD_ACTION_MIN_SECONDS,
            "bird_action_resource_zone_fallback": False,
            "timeline_start_date": DATE_START,
            "timeline_end_date": TIMELINE_END_DATE,
            "timeline_dates": dates,
        },
        "rooms": [
            {
                "room_id": ROOM_ID,
                "display_name": ROOM_LABEL,
                "zones": UNITY_ZONE_DEFINITIONS,
            }
        ],
        "timeline": timeline,
    }


def write_payload(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    STREAMING_JSON.parent.mkdir(parents=True, exist_ok=True)
    FALLBACK_JSON.parent.mkdir(parents=True, exist_ok=True)
    STREAMING_JSON.write_text(text, encoding="utf-8")
    FALLBACK_JSON.write_text(text, encoding="utf-8")


def main() -> int:
    zone_metrics_lookup = load_zone_feature_lookup()
    rows = load_window_rows(zone_metrics_lookup)
    if not rows:
        raise RuntimeError("No matching Aug 16-19 Room 1 windows found.")

    zone_config = load_zone_config()
    payload = build_timeline_payload(rows=rows, zone_config=zone_config, zone_metrics_lookup=zone_metrics_lookup)
    write_payload(payload)

    dates = sorted({row.local_date for row in rows})
    print(f"Wrote {len(payload['timeline'])} frames to {STREAMING_JSON}")
    print(f"Updated fallback copy at {FALLBACK_JSON}")
    print("Dates:", ", ".join(dates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
