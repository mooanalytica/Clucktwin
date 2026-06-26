from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .audio_feature_extractor import AUDIO_COLUMNS
from .config import PreparationConfig
from .utils import combine_warnings, prepare_time_columns, read_csv_if_exists, write_table


REQUIRED_PROCESSED_COLUMNS = [
    "window_id",
    "media_id",
    "room_id",
    "session_id",
    "start_time",
    "end_time",
    "local_date",
    "video_path",
    "activity_mean",
    "normalized_activity",
    "mobility_index",
    "spatial_freedom_index",
    "occupancy_imbalance_index",
    "drinking_activity_fraction",
    "feeding_activity_fraction",
    "open_movement_activity_fraction",
    "resting_activity_fraction",
    "general_activity_fraction",
    "semantic_transition_proxy",
    "audio_available",
    "audio_rms",
    "audio_short_time_energy",
    "audio_zero_crossing_rate",
    "audio_spectral_centroid",
    "audio_spectral_bandwidth",
    "audio_spectral_rolloff",
    "audio_spectral_flatness",
    "audio_energy_entropy",
    "audio_band_energy_ratio_2_5khz",
    "audio_band_energy_ratio_3_6khz",
    "audio_band_energy_ratio_300_1500hz",
    "audio_spectral_flux",
    "audio_frame_spectral_flatness_mean",
    "audio_dominant_frequency",
    "audio_call_like_event_rate",
    "audio_call_like_occupancy",
    "audio_chirp_like_event_rate",
    "audio_chirp_like_occupancy",
    "audio_chirp_like_mean_duration_sec",
    "audio_cluck_like_event_rate",
    "audio_cluck_like_occupancy",
    "audio_cluck_like_mean_duration_sec",
    "audio_bird_event_rate",
    "audio_bird_event_occupancy",
    "audio_bird_event_mean_duration_sec",
    "audio_dense_bird_event_rate",
    "audio_dense_bird_event_occupancy",
    "audio_dense_bird_event_mean_duration_sec",
    "audio_sparse_bird_event_rate",
    "audio_sparse_bird_event_occupancy",
    "audio_sparse_bird_event_mean_duration_sec",
    "audio_nonbird_event_rate",
    "audio_nonbird_event_occupancy",
    "audio_nonbird_event_mean_duration_sec",
    "audio_disturbance_event_rate",
    "audio_disturbance_event_occupancy",
    "audio_disturbance_event_mean_duration_sec",
    "audio_impact_like_event_rate",
    "audio_impact_like_occupancy",
    "audio_lowfreq_uncertain_event_rate",
    "audio_lowfreq_uncertain_occupancy",
    "audio_lowfreq_uncertain_mean_duration_sec",
    "audio_active_event_occupancy",
    "audio_bird_minus_nonbird_occupancy",
    "audio_bird_fraction_within_active",
    "audio_event_regime_hint",
    "audio_dense_regime_axis",
    "audio_sparse_regime_axis",
    "audio_regime_dense_score",
    "audio_regime_background_score",
    "audio_regime_sparse_score",
    "audio_regime_dense_score_smoothed_2min",
    "audio_regime_background_score_smoothed_2min",
    "audio_regime_sparse_score_smoothed_2min",
    "audio_regime_hard_label",
    "audio_regime_margin",
    "temp_context",
    "rh_context",
    "temp_daily_mean",
    "rh_daily_mean",
    "env_quality_flag",
    "quality_status",
    "warnings",
]


@dataclass(frozen=True)
class MergeResult:
    processed_df: pd.DataFrame
    output_path: Path
    report_path: Path
    handoff_readme_path: Path
    failed_output_path: Path


def merge_processed_outputs(
    config: PreparationConfig,
    media_df: pd.DataFrame,
    window_df: pd.DataFrame,
    biomarker_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    env_df: pd.DataFrame,
    failed_frames: list[pd.DataFrame] | None = None,
    zone_feature_df: pd.DataFrame | None = None,
) -> MergeResult:
    window_working_df = prepare_time_columns(window_df)
    processed_df = window_working_df.merge(
        biomarker_df.drop_duplicates(subset=["window_id"], keep="first"),
        on=["window_id", "media_id", "room_id", "session_id", "start_time", "end_time"],
        how="left",
        suffixes=("", "_video"),
    )
    if not audio_df.empty:
        processed_df = processed_df.merge(
            audio_df.drop_duplicates(subset=["window_id"], keep="first"),
            on=["window_id", "media_id", "room_id", "session_id", "start_time", "end_time"],
            how="left",
            suffixes=("", "_audio"),
        )

    processed_df["local_date"] = processed_df["start_time_dt"].dt.date
    processed_df["env_period"] = np.where(processed_df["start_time_dt"].dt.hour < 12, "AM", "PM")
    if not env_df.empty:
        env_working_df = env_df.copy()
        env_working_df["local_date"] = pd.to_datetime(env_working_df["date"], errors="coerce").dt.date
        processed_df = processed_df.merge(
            env_working_df[
                [
                    "room_id",
                    "local_date",
                    "temp_am_mean",
                    "temp_pm",
                    "temp_daily_mean",
                    "rh_am",
                    "rh_pm",
                    "rh_daily_mean",
                    "env_quality_flag",
                ]
            ],
            on=["room_id", "local_date"],
            how="left",
        )
        processed_df["temp_context"] = np.where(processed_df["env_period"] == "AM", processed_df["temp_am_mean"], processed_df["temp_pm"])
        processed_df["temp_context"] = processed_df["temp_context"].fillna(processed_df["temp_daily_mean"])
        processed_df["rh_context"] = np.where(processed_df["env_period"] == "AM", processed_df["rh_am"], processed_df["rh_pm"])
        processed_df["rh_context"] = processed_df["rh_context"].fillna(processed_df["rh_daily_mean"])
    else:
        processed_df["temp_context"] = np.nan
        processed_df["rh_context"] = np.nan
        processed_df["temp_daily_mean"] = np.nan
        processed_df["rh_daily_mean"] = np.nan
        processed_df["env_quality_flag"] = np.nan

    processed_df["audio_available"] = processed_df.get("audio_available", False)
    processed_df["audio_available"] = processed_df["audio_available"].fillna(False).astype(bool)
    processed_df["quality_status"] = processed_df.apply(_merged_quality_status, axis=1)
    processed_df["warnings"] = processed_df.apply(_merged_warnings, axis=1)

    for column in REQUIRED_PROCESSED_COLUMNS:
        if column not in processed_df.columns:
            processed_df[column] = np.nan

    processed_df = processed_df.sort_values(["room_id", "session_id", "start_time_dt", "window_id"], kind="stable").reset_index(drop=True)
    processed_df = processed_df.drop(columns=["start_time_dt", "end_time_dt", "env_period"], errors="ignore")
    processed_df = processed_df.drop_duplicates(subset=["window_id"], keep="first")

    output_path = config.features_output_dir / "processed_multimodal_window_table.csv"
    report_path = config.reports_output_dir / "processed_multimodal_handoff_report.md"
    handoff_readme_path = config.handoff_output_dir / "README_HANDOFF.md"
    write_table(processed_df[REQUIRED_PROCESSED_COLUMNS], output_path, write_csv=config.write_csv, write_parquet=config.write_parquet)

    failed_df = _combine_failed_frames(failed_frames)
    failed_df.to_csv(config.failed_output_path, index=False)

    _write_handoff_files(config, media_df, window_df, biomarker_df, audio_df, env_df, output_path, handoff_readme_path)
    if config.write_partitioned_outputs:
        _write_partitioned_outputs(
            config=config,
            media_df=media_df,
            window_df=window_df,
            biomarker_df=biomarker_df,
            audio_df=audio_df,
            env_df=env_df,
            processed_df=processed_df[REQUIRED_PROCESSED_COLUMNS],
            failed_df=failed_df,
            zone_feature_df=zone_feature_df,
        )
    report_path.write_text(_build_merge_report(processed_df[REQUIRED_PROCESSED_COLUMNS]), encoding="utf-8")
    return MergeResult(
        processed_df=processed_df[REQUIRED_PROCESSED_COLUMNS],
        output_path=output_path,
        report_path=report_path,
        handoff_readme_path=handoff_readme_path,
        failed_output_path=config.failed_output_path,
    )


def _merged_quality_status(row: pd.Series) -> str:
    if pd.isna(row.get("activity_mean")):
        return "failed"
    if str(row.get("quality_status_video", "")).strip().lower() == "failed":
        return "failed"
    if str(row.get("audio_quality_flag", "")).strip().lower() not in {"", "ok", "manifest_has_audio_false"}:
        return "warning"
    if str(row.get("env_quality_flag", "")).strip().lower() not in {"", "ok"}:
        return "warning"
    if str(row.get("quality_status", "")).strip().lower() in {"warning", "failed"}:
        return str(row.get("quality_status"))
    if str(row.get("quality_status_video", "")).strip().lower() == "warning":
        return "warning"
    return "ok"


def _merged_warnings(row: pd.Series) -> str:
    return combine_warnings(
        row.get("warnings", ""),
        row.get("warnings_video", ""),
        row.get("warnings_audio", ""),
        row.get("audio_quality_flag", "") if str(row.get("audio_quality_flag", "")).lower() not in {"", "ok"} else "",
        row.get("env_quality_flag", "") if str(row.get("env_quality_flag", "")).lower() not in {"", "ok"} else "",
    )


def _combine_failed_frames(failed_frames: list[pd.DataFrame] | None) -> pd.DataFrame:
    valid_frames = [frame for frame in (failed_frames or []) if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame(columns=["stage", "window_id", "media_id", "room_id", "issue"])
    combined = pd.concat(valid_frames, ignore_index=True)
    return combined.drop_duplicates(subset=["stage", "window_id", "issue"], keep="first")


def _write_handoff_files(
    config: PreparationConfig,
    media_df: pd.DataFrame,
    window_df: pd.DataFrame,
    biomarker_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    env_df: pd.DataFrame,
    processed_output_path: Path,
    handoff_readme_path: Path,
) -> None:
    if not config.copy_manifest_for_mvp:
        return
    config.handoff_output_dir.mkdir(parents=True, exist_ok=True)
    file_map = {
        "media_manifest.csv": config.metadata_output_dir / "media_manifest.csv",
        "video_window_index.csv": config.metadata_output_dir / "video_window_index.csv",
        "semantic_zone_video_features.csv": config.features_output_dir / "semantic_zone_video_features.csv",
        "semantic_biomarker_window_table.csv": config.features_output_dir / "semantic_biomarker_window_table.csv",
        "audio_window_features.csv": config.features_output_dir / "audio_window_features.csv",
        "env_daily.csv": config.features_output_dir / "env_daily.csv",
        "processed_multimodal_window_table.csv": processed_output_path,
        "semantic_zone_configs.json": config.zones_output_dir / "semantic_zone_configs.json",
    }
    for target_name, source_path in file_map.items():
        if source_path.exists():
            shutil.copy2(source_path, config.handoff_output_dir / target_name)

    handoff_readme_path.write_text(
        "\n".join(
            [
                "# Handoff For MVP",
                "",
                "This handoff folder contains processed metadata and feature tables for downstream MVPs.",
                "",
                "## Files",
                "",
                "- `media_manifest.csv`: raw-video media-level manifest with technical metadata and timestamps",
                "- `video_window_index.csv`: fixed video windows and primary handoff for window-based processing",
                "- `semantic_zone_video_features.csv`: one row per room/media/window/semantic zone activity feature",
                "- `semantic_biomarker_window_table.csv`: window-level semantic activity distribution features with explicit drinking / feeding / open movement / resting fractions",
                "- `audio_window_features.csv`: embedded MP4 audio features aggregated to the same windows",
                "- `env_daily.csv`: room-level daily environment context",
                "- `processed_multimodal_window_table.csv`: merged handoff table for downstream analytics",
                "- `semantic_zone_configs.json`: semantic zone geometry definitions per room",
                "",
                "## Notes",
                "",
                "- Videos are not copied into this folder.",
                "- Path columns may be absolute or relative depending on `paths_in_outputs` in the config.",
                "- Downstream MVPs should read `video_window_index.csv` and `processed_multimodal_window_table.csv` as their main window-level handoff tables.",
                "- `general_activity_fraction` is kept as a compatibility aggregate equal to `open_movement_activity_fraction + resting_activity_fraction`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_partitioned_outputs(
    config: PreparationConfig,
    media_df: pd.DataFrame,
    window_df: pd.DataFrame,
    biomarker_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    env_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    failed_df: pd.DataFrame,
    zone_feature_df: pd.DataFrame | None,
) -> None:
    if media_df.empty:
        return

    partition_df = _build_media_partition_df(media_df)
    if partition_df.empty:
        return

    for (room_folder, session_folder), partition_media_df in partition_df.groupby(
        ["room_folder", "session_folder"],
        sort=False,
        dropna=False,
    ):
        media_ids = set(partition_media_df["media_id"].astype(str))
        session_dir = config.output_root / str(room_folder) / str(session_folder)
        session_dir.mkdir(parents=True, exist_ok=True)
        window_subset = window_df[window_df["media_id"].astype(str).isin(media_ids)].copy() if not window_df.empty else pd.DataFrame()
        window_ids = set(window_subset["window_id"].astype(str)) if not window_subset.empty else set()
        room_ids = set(partition_media_df["room_id"].astype(str))

        _write_partition_table(partition_media_df, session_dir / "media_manifest.csv")
        _write_partition_table(window_subset, session_dir / "video_window_index.csv")
        _write_partition_table(_filter_by_window_ids(processed_df, window_ids), session_dir / "processed_multimodal_window_table.csv")
        _write_partition_table(_filter_by_window_ids(biomarker_df, window_ids), session_dir / "semantic_biomarker_window_table.csv")
        _write_partition_table(_filter_by_window_ids(audio_df, window_ids).reindex(columns=[column for column in AUDIO_COLUMNS if column in audio_df.columns]), session_dir / "audio_window_features.csv")
        if zone_feature_df is not None and not zone_feature_df.empty:
            _write_partition_table(_filter_by_window_ids(zone_feature_df, window_ids), session_dir / "semantic_zone_video_features.csv")
        _write_partition_table(_filter_by_room_ids(env_df, room_ids), session_dir / "env_daily.csv")
        _write_partition_table(_filter_failed_windows(failed_df, window_ids), session_dir / "failed_windows.csv")
        _write_partition_readme(session_dir, str(room_folder), str(session_folder), len(partition_media_df), len(window_subset))


def _build_media_partition_df(media_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in media_df.iterrows():
        room_folder, session_folder = _room_session_folders_from_media(row)
        record = row.to_dict()
        record["room_folder"] = room_folder
        record["session_folder"] = session_folder
        rows.append(record)
    return pd.DataFrame(rows)


def _room_session_folders_from_media(row: pd.Series) -> tuple[str, str]:
    relative_path = Path(str(row.get("relative_path", "")))
    parts = list(relative_path.parts)
    lower_parts = [part.lower() for part in parts]
    if "video" in lower_parts:
        video_index = lower_parts.index("video")
        if len(parts) > video_index + 1:
            room_folder = parts[video_index + 1]
        else:
            room_folder = _display_room_folder(row)
        if len(parts) > video_index + 3:
            session_folder = parts[video_index + 2]
        elif len(parts) > video_index + 2:
            session_folder = parts[video_index + 1]
        else:
            session_folder = str(row.get("session_id", "unknown_session"))
    else:
        room_folder = _display_room_folder(row)
        session_folder = str(row.get("session_id", "unknown_session"))
    return _safe_folder_name(room_folder), _safe_folder_name(session_folder)


def _display_room_folder(row: pd.Series) -> str:
    display_name = str(row.get("room_display_name", "")).strip()
    if display_name:
        return display_name
    room_id = str(row.get("room_id", "unknown_room"))
    if room_id.startswith("room_"):
        suffix = room_id.replace("room_", "", 1)
        if suffix.isdigit():
            return f"Room {int(suffix)}"
    return room_id


def _safe_folder_name(value: object) -> str:
    text = str(value or "").strip() or "unknown"
    for character in '<>:"/\\|?*':
        text = text.replace(character, "_")
    return text.rstrip(". ") or "unknown"


def _filter_by_window_ids(dataframe: pd.DataFrame, window_ids: set[str]) -> pd.DataFrame:
    if dataframe.empty or "window_id" not in dataframe.columns or not window_ids:
        return pd.DataFrame(columns=dataframe.columns)
    return dataframe[dataframe["window_id"].astype(str).isin(window_ids)].copy()


def _filter_by_room_ids(dataframe: pd.DataFrame, room_ids: set[str]) -> pd.DataFrame:
    if dataframe.empty or "room_id" not in dataframe.columns or not room_ids:
        return pd.DataFrame(columns=dataframe.columns)
    return dataframe[dataframe["room_id"].astype(str).isin(room_ids)].copy()


def _filter_failed_windows(failed_df: pd.DataFrame, window_ids: set[str]) -> pd.DataFrame:
    if failed_df.empty or "window_id" not in failed_df.columns or not window_ids:
        return pd.DataFrame(columns=failed_df.columns if not failed_df.empty else ["stage", "window_id", "media_id", "room_id", "issue"])
    return failed_df[failed_df["window_id"].astype(str).isin(window_ids)].copy()


def _write_partition_table(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def _write_partition_readme(
    session_dir: Path,
    room_folder: str,
    session_folder: str,
    media_count: int,
    window_count: int,
) -> None:
    (session_dir / "README_SESSION.md").write_text(
        "\n".join(
            [
                f"# {room_folder} / {session_folder}",
                "",
                "This folder contains CSV subsets for one raw-video session folder.",
                "",
                f"- Media files: {media_count}",
                f"- Windows: {window_count}",
                "",
                "The top-level output folder still contains full-run summary tables.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _build_merge_report(processed_df: pd.DataFrame) -> str:
    total_windows = len(processed_df)
    with_video = int(processed_df["activity_mean"].notna().sum()) if total_windows else 0
    with_audio = int(processed_df["audio_available"].fillna(False).astype(bool).sum()) if total_windows else 0
    with_env = int((processed_df["temp_context"].notna() | processed_df["rh_context"].notna()).sum()) if total_windows else 0
    lines = [
        "# Processed Multimodal Handoff Report",
        "",
        f"- Total windows: {total_windows}",
        f"- Windows per room: {processed_df['room_id'].value_counts().to_dict() if total_windows else {}}",
        f"- Time span per room: {processed_df.groupby('room_id')['start_time'].agg(['min', 'max']).to_dict('index') if total_windows else {}}",
        f"- Percent with video features: {(100.0 * with_video / total_windows):.2f}% " if total_windows else "- Percent with video features: n/a",
        f"- Percent with audio features: {(100.0 * with_audio / total_windows):.2f}% " if total_windows else "- Percent with audio features: n/a",
        f"- Percent with environment match: {(100.0 * with_env / total_windows):.2f}% " if total_windows else "- Percent with environment match: n/a",
        "",
        "## Missingness Summary",
        "",
    ]
    for column, count in processed_df.isna().sum().to_dict().items():
        lines.append(f"- `{column}`: {int(count)}")
    lines.extend(
        [
            "",
            "## Recommended Next Commands",
            "",
            "- Downstream semantic-zone analytics can read `outputs/features/processed_multimodal_window_table.csv`.",
            "- Window-based pipelines can also read `outputs/metadata/video_window_index.csv` and `outputs/features/semantic_biomarker_window_table.csv`.",
        ]
    )
    return "\n".join(lines) + "\n"
