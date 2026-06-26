from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from .utils import ensure_directories, resolve_path


@dataclass
class PreparationConfig:
    config_path: Path
    config_dir: Path
    project_root: Path
    raw_root: Path
    metadata_root: Path
    output_root: Path
    cache_root: Path
    metadata_output_dir: Path
    features_output_dir: Path
    zones_output_dir: Path
    reports_output_dir: Path
    logs_output_dir: Path
    handoff_output_dir: Path
    video_feature_cache_dir: Path
    audio_feature_cache_dir: Path
    failed_output_path: Path
    log_path: Path
    video_globs: list[str]
    env_glob: str
    semantic_zone_ref_dir: Path
    caretaker_event_globs: list[str]
    ffprobe_path: str
    ffmpeg_path: str
    exiftool_path: str
    rooms_include: str | list[str]
    rooms_exclude: list[str]
    window_seconds: int
    stride_seconds: int
    video_features_enabled: bool
    video_method: str
    resize_width: int
    max_windows: int | None
    max_windows_per_room: int | None
    max_windows_per_media: int | None
    max_media: int | None
    video_force_recompute: bool
    audio_features_enabled: bool
    audio_sample_rate: int
    audio_frame_seconds: float
    audio_analysis_frame_seconds: float
    audio_entropy_bands: int
    audio_regime_score_smoothing_windows: int
    audio_cache_per_media: bool
    audio_force_recompute: bool
    semantic_zones_enabled: bool
    semantic_auto_detect: bool
    semantic_allow_manual_fallback: bool
    write_csv: bool
    write_parquet: bool
    copy_manifest_for_mvp: bool
    write_partitioned_outputs: bool
    timestamp_output_root: bool
    run_timestamp: str | None
    paths_in_outputs: str


def load_config(config_path: str | Path, max_windows: int | None = None, max_media: int | None = None) -> PreparationConfig:
    resolved_config_path = Path(config_path).resolve()
    with resolved_config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    config_dir = resolved_config_path.parent
    paths = raw_config.get("paths", {})
    inputs = raw_config.get("inputs", {})
    tools = raw_config.get("tools", {})
    rooms = raw_config.get("rooms", {})
    windowing = raw_config.get("windowing", {})
    video_features = raw_config.get("video_features", {})
    audio_features = raw_config.get("audio_features", {})
    semantic_zones = raw_config.get("semantic_zones", {})
    output = raw_config.get("output", {})

    project_root = resolve_path(config_dir, paths.get("project_root", ".."))
    raw_root = resolve_path(project_root, paths.get("raw_root", "data/raw"))
    metadata_root = resolve_path(project_root, paths.get("metadata_root", "data/metadata"))
    output_root = resolve_path(project_root, paths.get("output_root", "poultry_data_preparation/outputs"))
    timestamp_output_root = bool(output.get("timestamp_output_root", False))
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp_output_root else None
    if run_timestamp:
        output_root = output_root.with_name(f"{output_root.name}_{run_timestamp}")
    cache_root = resolve_path(project_root, paths.get("cache_root", str(output_root / "cache")))

    config = PreparationConfig(
        config_path=resolved_config_path,
        config_dir=config_dir,
        project_root=project_root,
        raw_root=raw_root,
        metadata_root=metadata_root,
        output_root=output_root,
        cache_root=cache_root,
        metadata_output_dir=output_root / "metadata",
        features_output_dir=output_root / "features",
        zones_output_dir=output_root / "zones",
        reports_output_dir=output_root / "reports",
        logs_output_dir=output_root / "logs",
        handoff_output_dir=output_root / "handoff_for_mvp",
        video_feature_cache_dir=cache_root / "video_features",
        audio_feature_cache_dir=cache_root / "audio_features",
        failed_output_path=output_root / "reports" / "failed_windows.csv",
        log_path=output_root / "logs" / "preparation.log",
        video_globs=list(inputs.get("video_globs", ["video/Room */**/*.MP4"])),
        env_glob=str(inputs.get("env_glob", "env/Combined Room *.xlsx")),
        semantic_zone_ref_dir=resolve_path(metadata_root, inputs.get("semantic_zone_ref_dir", "semantic_zone_refs")),
        caretaker_event_globs=_ensure_list(inputs.get("caretaker_event_glob", [])),
        ffprobe_path=str(tools.get("ffprobe_path", "ffprobe")),
        ffmpeg_path=str(tools.get("ffmpeg_path", "ffmpeg")),
        exiftool_path=str(tools.get("exiftool_path", "exiftool")),
        rooms_include=rooms.get("include", "auto"),
        rooms_exclude=[str(item) for item in rooms.get("exclude", [])],
        window_seconds=int(windowing.get("window_seconds", 30)),
        stride_seconds=int(windowing.get("stride_seconds", 10)),
        video_features_enabled=bool(video_features.get("enabled", True)),
        video_method=str(video_features.get("method", "frame_difference")),
        resize_width=int(video_features.get("resize_width", 640)),
        max_windows=_optional_int(max_windows if max_windows is not None else video_features.get("max_windows")),
        max_windows_per_room=_optional_int(video_features.get("max_windows_per_room")),
        max_windows_per_media=_optional_int(video_features.get("max_windows_per_media")),
        max_media=_optional_int(max_media if max_media is not None else video_features.get("max_media")),
        video_force_recompute=bool(video_features.get("force_recompute", False)),
        audio_features_enabled=bool(audio_features.get("enabled", True)),
        audio_sample_rate=int(audio_features.get("sample_rate", 22050)),
        audio_frame_seconds=float(audio_features.get("frame_seconds", 1.0)),
        audio_analysis_frame_seconds=float(audio_features.get("analysis_frame_seconds", 0.1)),
        audio_entropy_bands=int(audio_features.get("entropy_bands", 10)),
        audio_regime_score_smoothing_windows=int(audio_features.get("regime_score_smoothing_windows", 12)),
        audio_cache_per_media=bool(audio_features.get("cache_per_media_audio", True)),
        audio_force_recompute=bool(audio_features.get("force_recompute", False)),
        semantic_zones_enabled=bool(semantic_zones.get("enabled", True)),
        semantic_auto_detect=bool(semantic_zones.get("auto_detect_from_annotated_image", True)),
        semantic_allow_manual_fallback=bool(semantic_zones.get("allow_manual_fallback", True)),
        write_csv=bool(output.get("write_csv", True)),
        write_parquet=bool(output.get("write_parquet", False)),
        copy_manifest_for_mvp=bool(output.get("copy_manifest_for_mvp", True)),
        write_partitioned_outputs=bool(output.get("write_partitioned_outputs", True)),
        timestamp_output_root=timestamp_output_root,
        run_timestamp=run_timestamp,
        paths_in_outputs=str(output.get("paths_in_outputs", "relative")).lower(),
    )
    return config


def ensure_output_dirs(config: PreparationConfig) -> None:
    ensure_directories(
        [
            config.output_root,
            config.cache_root,
            config.metadata_output_dir,
            config.features_output_dir,
            config.zones_output_dir,
            config.reports_output_dir,
            config.logs_output_dir,
            config.handoff_output_dir,
            config.video_feature_cache_dir,
            config.audio_feature_cache_dir,
        ]
    )


def _ensure_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    return int(value)
