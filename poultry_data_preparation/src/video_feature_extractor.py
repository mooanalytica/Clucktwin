from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

from .config import PreparationConfig
from .semantic_zone_loader import build_zone_masks, get_zone_polygons, scale_polygon
from .utils import combine_warnings, distribution_from_values, minmax_scale, prepare_time_columns, write_table

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # pragma: no cover - depends on local environment
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


LOGGER = logging.getLogger(__name__)

VIDEO_CACHE_SCHEMA_VERSION = 2
FRAME_DIFF_SAMPLE_PERIOD_SEC = 2.0

EXPLICIT_ZONE_IDS = [
    "drinking_zone",
    "feeding_zone",
    "open_movement_zone",
    "resting_zone",
]


ZONE_FEATURE_COLUMNS = [
    "window_id",
    "media_id",
    "room_id",
    "session_id",
    "start_time",
    "end_time",
    "video_path",
    "video_start_offset_sec",
    "duration_seconds",
    "zone_config_id",
    "zone_id",
    "semantic_type",
    "zone_area_pixels",
    "zone_area_fraction",
    "activity_mean",
    "activity_std",
    "activity_sum",
    "activity_norm",
    "frame_count_used",
    "quality_status",
    "warnings",
]


BIOMARKER_COLUMNS = [
    "window_id",
    "media_id",
    "room_id",
    "session_id",
    "start_time",
    "end_time",
    "activity_mean",
    "normalized_activity",
    "mobility_index",
    "spatial_freedom_index",
    "occupancy_imbalance_index",
    "semantic_transition_proxy",
    "drinking_activity_fraction",
    "feeding_activity_fraction",
    "open_movement_activity_fraction",
    "resting_activity_fraction",
    "general_activity_fraction",
    "feeding_plus_drinking_activity_fraction",
    "drinking_to_feeding_activity_ratio",
    "quality_status",
    "warnings",
]


@dataclass(frozen=True)
class VideoFeatureResult:
    zone_feature_df: pd.DataFrame
    biomarker_df: pd.DataFrame
    zone_feature_output_path: Path
    biomarker_output_path: Path
    report_path: Path
    failed_df: pd.DataFrame


def extract_video_features(
    config: PreparationConfig,
    media_df: pd.DataFrame,
    window_df: pd.DataFrame,
    zone_configs: list[dict],
    dry_run: bool = False,
) -> VideoFeatureResult:
    zone_feature_output_path = config.features_output_dir / "semantic_zone_video_features.csv"
    biomarker_output_path = config.features_output_dir / "semantic_biomarker_window_table.csv"
    report_path = config.reports_output_dir / "video_feature_report.md"

    if dry_run or not config.video_features_enabled:
        zone_feature_df = pd.DataFrame(columns=ZONE_FEATURE_COLUMNS)
        biomarker_df = pd.DataFrame(columns=BIOMARKER_COLUMNS)
        report_path.write_text(_build_dry_run_report(window_df, zone_configs, config.video_features_enabled), encoding="utf-8")
        return VideoFeatureResult(zone_feature_df, biomarker_df, zone_feature_output_path, biomarker_output_path, report_path, pd.DataFrame())

    if cv2 is None:
        raise RuntimeError("OpenCV is required for video semantic-zone feature extraction.")

    zone_config_map = {item["room_id"]: item for item in zone_configs}
    all_rows: list[dict] = []
    failed_rows: list[dict] = []
    media_lookup = media_df.set_index("media_id", drop=False).to_dict(orient="index")

    grouped = list(window_df.groupby("media_id", sort=False, dropna=False))
    total_groups = len(grouped)
    processed_windows = 0
    for group_index, (media_id, media_windows) in enumerate(grouped, start=1):
        media_windows = media_windows.sort_values("video_start_offset_sec", kind="stable").reset_index(drop=True)
        room_id = str(media_windows.iloc[0]["room_id"])
        zone_config = zone_config_map.get(room_id)
        LOGGER.info("Video features media %s/%s: %s (%s windows)", group_index, total_groups, media_id, len(media_windows))
        if zone_config is None:
            for _, row in media_windows.iterrows():
                failed_rows.append(_failed_window_row(row, "missing_semantic_zone_config"))
            continue

        media_metadata = media_lookup.get(str(media_id))
        if media_metadata is None:
            for _, row in media_windows.iterrows():
                failed_rows.append(_failed_window_row(row, "missing_media_manifest_row"))
            continue

        absolute_path = Path(str(media_metadata["absolute_path"]))
        cache_data = _load_or_compute_media_cache(config, zone_config, absolute_path, media_windows, force_recompute=config.video_force_recompute)
        media_rows, media_failures = _aggregate_media_windows(media_windows, zone_config, cache_data)
        all_rows.extend(media_rows)
        failed_rows.extend(media_failures)
        processed_windows += len(media_windows)

        partial_df = pd.DataFrame(all_rows, columns=ZONE_FEATURE_COLUMNS)
        if not partial_df.empty:
            partial_df = _finalize_zone_feature_df(partial_df)
        write_table(partial_df, zone_feature_output_path, write_csv=config.write_csv, write_parquet=False)
        LOGGER.info("Processed %s/%s video windows", processed_windows, len(window_df))

    zone_feature_df = pd.DataFrame(all_rows, columns=ZONE_FEATURE_COLUMNS)
    if not zone_feature_df.empty:
        zone_feature_df = _finalize_zone_feature_df(zone_feature_df)

    biomarker_df = build_semantic_biomarker_table(zone_feature_df)
    write_table(zone_feature_df, zone_feature_output_path, write_csv=config.write_csv, write_parquet=config.write_parquet)
    write_table(biomarker_df, biomarker_output_path, write_csv=config.write_csv, write_parquet=config.write_parquet)

    failed_df = pd.DataFrame(failed_rows)
    _plot_room_level_outputs(zone_feature_df, biomarker_df, config)
    report_path.write_text(_build_video_feature_report(zone_feature_df, biomarker_df, failed_df), encoding="utf-8")
    return VideoFeatureResult(zone_feature_df, biomarker_df, zone_feature_output_path, biomarker_output_path, report_path, failed_df)


def build_semantic_biomarker_table(zone_feature_df: pd.DataFrame) -> pd.DataFrame:
    if zone_feature_df.empty:
        return pd.DataFrame(columns=BIOMARKER_COLUMNS)

    working_df = prepare_time_columns(zone_feature_df)
    rows: list[dict] = []
    grouped = working_df.groupby(["window_id", "media_id", "room_id", "session_id", "start_time", "end_time"], sort=False, dropna=False)
    for group_key, group_df in grouped:
        zone_lookup = {
            str(row["zone_id"]): row
            for _, row in group_df.iterrows()
        }

        zone_values = pd.Series(
            {
                zone_id: pd.to_numeric(zone_lookup.get(zone_id, {}).get("activity_mean"), errors="coerce")
                for zone_id in EXPLICIT_ZONE_IDS
            },
            dtype=float,
        ).fillna(0.0)
        distribution = distribution_from_values(zone_values)
        activity_mean = float(zone_values.mean())
        rows.append(
            {
                "window_id": str(group_key[0]),
                "media_id": str(group_key[1]),
                "room_id": str(group_key[2]),
                "session_id": str(group_key[3]),
                "start_time": str(group_key[4]),
                "end_time": str(group_key[5]),
                "activity_mean": activity_mean,
                "spatial_freedom_index": _spatial_freedom(distribution.to_numpy(dtype=float)),
                "occupancy_imbalance_index": _imbalance(distribution.to_numpy(dtype=float)),
                "drinking_activity_fraction": float(distribution["drinking_zone"]),
                "feeding_activity_fraction": float(distribution["feeding_zone"]),
                "open_movement_activity_fraction": float(distribution["open_movement_zone"]),
                "resting_activity_fraction": float(distribution["resting_zone"]),
                "general_activity_fraction": float(distribution["open_movement_zone"] + distribution["resting_zone"]),
                "feeding_plus_drinking_activity_fraction": float(distribution["drinking_zone"] + distribution["feeding_zone"]),
                "drinking_to_feeding_activity_ratio": float(distribution["drinking_zone"] / distribution["feeding_zone"]) if distribution["feeding_zone"] > 0 else np.nan,
                "quality_status": "failed" if (group_df["quality_status"] == "failed").any() else ("warning" if (group_df["quality_status"] == "warning").any() else "ok"),
                "warnings": combine_warnings(group_df["warnings"].tolist()),
            }
        )

    biomarker_df = pd.DataFrame(rows, columns=BIOMARKER_COLUMNS)
    biomarker_df = prepare_time_columns(biomarker_df)
    biomarker_df = biomarker_df.sort_values(["room_id", "session_id", "start_time_dt", "window_id"], kind="stable").reset_index(drop=True)

    transition_values: list[float] = []
    for _, room_group in biomarker_df.groupby("room_id", sort=False, dropna=False):
        previous = None
        for _, row in room_group.iterrows():
            current = np.array(
                [
                    row["drinking_activity_fraction"],
                    row["feeding_activity_fraction"],
                    row["open_movement_activity_fraction"],
                    row["resting_activity_fraction"],
                ],
                dtype=float,
            )
            transition_values.append(0.0 if previous is None else float(0.5 * np.abs(current - previous).sum()))
            previous = current
    biomarker_df["semantic_transition_proxy"] = transition_values
    biomarker_df["normalized_activity"] = biomarker_df.groupby("room_id", sort=False)["activity_mean"].transform(minmax_scale)
    biomarker_df["mobility_index"] = (
        0.7 * biomarker_df["normalized_activity"].fillna(0.0)
        + 0.3 * biomarker_df["semantic_transition_proxy"].fillna(0.0)
    ).clip(0.0, 1.0)
    biomarker_df = biomarker_df.drop(columns=["start_time_dt", "end_time_dt"], errors="ignore")
    return biomarker_df[BIOMARKER_COLUMNS]


def _load_or_compute_media_cache(
    config: PreparationConfig,
    zone_config: dict,
    absolute_path: Path,
    media_windows: pd.DataFrame,
    force_recompute: bool,
) -> dict[str, object]:
    media_id = str(media_windows.iloc[0]["media_id"])
    cache_path = config.video_feature_cache_dir / f"{media_id}.joblib"
    expected_metadata = _build_video_cache_metadata(config, zone_config, absolute_path)
    if cache_path.exists() and not force_recompute:
        cached = joblib.load(cache_path)
        if _cache_metadata_matches(cached, expected_metadata):
            LOGGER.info("Reusing cached video features for %s", media_id)
            return cached
        LOGGER.info("Discarding stale video feature cache for %s", media_id)
    LOGGER.info("Computing video features for %s", media_id)
    cache_data = _compute_media_interval_metrics(config, zone_config, absolute_path, media_windows)
    cache_data["cache_metadata"] = expected_metadata
    joblib.dump(cache_data, cache_path)
    return cache_data


def _build_video_cache_metadata(config: PreparationConfig, zone_config: dict, absolute_path: Path) -> dict[str, object]:
    resolved_path = absolute_path.resolve()
    file_stat = resolved_path.stat()
    return {
        "schema_version": VIDEO_CACHE_SCHEMA_VERSION,
        "video_method": str(config.video_method),
        "resize_width": int(config.resize_width),
        "sample_period_sec": float(FRAME_DIFF_SAMPLE_PERIOD_SEC),
        "source_video_path": str(resolved_path),
        "source_video_size": int(file_stat.st_size),
        "source_video_mtime_ns": int(file_stat.st_mtime_ns),
        "zone_signature": _zone_signature(zone_config),
    }


def _zone_signature(zone_config: dict) -> str:
    serialized = {
        "room_id": str(zone_config.get("room_id", "")),
        "zone_config_id": str(zone_config.get("zone_config_id", "")),
        "image_width": int(zone_config.get("image_width", 0) or 0),
        "image_height": int(zone_config.get("image_height", 0) or 0),
        "zones": sorted(
            [
                {
                    "zone_id": str(zone.get("zone_id", "")),
                    "semantic_type": str(zone.get("semantic_type", "")),
                    "polygon": zone.get("polygon"),
                    "polygons": zone.get("polygons"),
                    "definition": zone.get("definition"),
                }
                for zone in zone_config.get("zones", [])
            ],
            key=lambda item: item["zone_id"],
        ),
    }
    return json.dumps(serialized, sort_keys=True, separators=(",", ":"))


def _cache_metadata_matches(cache_data: dict[str, object], expected_metadata: dict[str, object]) -> bool:
    cached_metadata = cache_data.get("cache_metadata")
    if not isinstance(cached_metadata, dict):
        return False
    return all(cached_metadata.get(key) == value for key, value in expected_metadata.items())


def _compute_media_interval_metrics(
    config: PreparationConfig,
    zone_config: dict,
    absolute_path: Path,
    media_windows: pd.DataFrame,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(absolute_path))
    if not capture or not capture.isOpened():
        if capture:
            capture.release()
        return {"interval_center_sec": np.array([], dtype=float), "zone_metrics": {}, "warnings": ["opencv_open_failed"]}

    sample_period = FRAME_DIFF_SAMPLE_PERIOD_SEC
    max_offset = float(
        (
            pd.to_numeric(media_windows["video_start_offset_sec"], errors="coerce").fillna(0.0)
            + pd.to_numeric(media_windows["duration_seconds"], errors="coerce").fillna(0.0)
        ).max()
    )
    sample_times = np.arange(0.0, max_offset + sample_period * 0.5, sample_period, dtype=float)
    sampled_frames: list[np.ndarray] = []
    sampled_times: list[float] = []
    resized_masks: dict[str, np.ndarray] | None = None
    resize_height: int | None = None
    warnings: list[str] = []
    zone_ids = _ordered_zone_ids(zone_config)

    for sample_time in sample_times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(sample_time) * 1000.0)
        success, frame = capture.read()
        if not success or frame is None:
            warnings.append(f"sample_read_failed_at_{sample_time:.2f}s")
            continue
        resized = _resize_frame(frame, config.resize_width)
        grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if resize_height is None:
            resize_height = int(grayscale.shape[0])
            resized_masks = _build_resized_masks(zone_config, config.resize_width, resize_height)
        sampled_frames.append(grayscale)
        sampled_times.append(float(sample_time))
    capture.release()

    if len(sampled_frames) < 2 or resized_masks is None:
        warnings.append("insufficient_sampled_frames")
        return {"interval_center_sec": np.array([], dtype=float), "zone_metrics": {}, "warnings": warnings}

    zone_metrics = {
        zone_id: {"activity_mean": [], "activity_std": [], "activity_sum": []}
        for zone_id in zone_ids
    }
    interval_centers: list[float] = []
    for previous_time, previous_frame, current_time, current_frame in zip(
        sampled_times[:-1], sampled_frames[:-1], sampled_times[1:], sampled_frames[1:]
    ):
        diff = np.abs(current_frame - previous_frame)
        interval_centers.append(float((previous_time + current_time) / 2.0))
        for zone_id in zone_ids:
            mask = resized_masks[zone_id]
            zone_values = diff[mask]
            zone_metrics[zone_id]["activity_mean"].append(float(zone_values.mean()) if zone_values.size else np.nan)
            zone_metrics[zone_id]["activity_std"].append(float(zone_values.std()) if zone_values.size else np.nan)
            zone_metrics[zone_id]["activity_sum"].append(float(zone_values.sum()) if zone_values.size else np.nan)

    return {
        "interval_center_sec": np.asarray(interval_centers, dtype=float),
        "zone_metrics": {
            zone_id: {metric_name: np.asarray(values, dtype=float) for metric_name, values in metric_dict.items()}
            for zone_id, metric_dict in zone_metrics.items()
        },
        "warnings": warnings,
    }


def _build_resized_masks(zone_config: dict, resize_width: int, resize_height: int) -> dict[str, np.ndarray]:
    source_width = int(zone_config["image_width"])
    source_height = int(zone_config["image_height"])
    resized_config = dict(zone_config)
    resized_config["image_width"] = int(resize_width)
    resized_config["image_height"] = int(resize_height)
    resized_zones: list[dict] = []
    for zone in zone_config["zones"]:
        resized_zone = dict(zone)
        polygons = get_zone_polygons(zone)
        if polygons:
            scaled_polygons = [
                scale_polygon(polygon, source_width, source_height, resize_width, resize_height).round().astype(int).tolist()
                for polygon in polygons
            ]
            resized_zone.pop("polygon", None)
            resized_zone.pop("polygons", None)
            if len(scaled_polygons) == 1:
                resized_zone["polygon"] = scaled_polygons[0]
            else:
                resized_zone["polygons"] = scaled_polygons
        resized_zones.append(resized_zone)
    resized_config["zones"] = resized_zones
    return build_zone_masks(resized_config, resize_width, resize_height)


def _aggregate_media_windows(media_windows: pd.DataFrame, zone_config: dict, cache_data: dict[str, object]) -> tuple[list[dict], list[dict]]:
    interval_center_sec = np.asarray(cache_data.get("interval_center_sec", np.array([], dtype=float)), dtype=float)
    zone_metrics = cache_data.get("zone_metrics", {})
    warnings = cache_data.get("warnings", [])
    source_masks = build_zone_masks(zone_config, int(zone_config["image_width"]), int(zone_config["image_height"]))
    total_pixels = int(zone_config["image_width"]) * int(zone_config["image_height"])
    zone_ids = _ordered_zone_ids(zone_config)
    zone_area_lookup = {
        zone_id: {
            "zone_area_pixels": int(source_masks[zone_id].sum()),
            "zone_area_fraction": float(source_masks[zone_id].sum() / total_pixels),
        }
        for zone_id in zone_ids
    }
    semantic_lookup = {
        str(zone["zone_id"]): str(zone.get("semantic_type", str(zone["zone_id"]).replace("_zone", "")))
        for zone in zone_config["zones"]
    }

    rows: list[dict] = []
    failed_rows: list[dict] = []
    for _, row in media_windows.iterrows():
        start_offset = float(pd.to_numeric(row["video_start_offset_sec"], errors="coerce"))
        duration_seconds = float(pd.to_numeric(row["duration_seconds"], errors="coerce"))
        end_offset = start_offset + duration_seconds
        overlap_mask = (interval_center_sec >= start_offset) & (interval_center_sec < end_offset)
        frame_count_used = int(overlap_mask.sum())
        status = "ok"
        warning_text = combine_warnings(row.get("warnings", ""), warnings)
        if frame_count_used == 0:
            status = "failed"
            warning_text = combine_warnings(warning_text, "no_interval_overlap")
            failed_rows.append(_failed_window_row(row, "no_interval_overlap"))

        for zone_id in zone_ids:
            metric_dict = zone_metrics.get(zone_id, {})
            activity_mean = float(np.nanmean(metric_dict.get("activity_mean", [np.nan])[overlap_mask])) if frame_count_used > 0 else np.nan
            activity_std = float(np.nanmean(metric_dict.get("activity_std", [np.nan])[overlap_mask])) if frame_count_used > 0 else np.nan
            activity_sum = float(np.nansum(metric_dict.get("activity_sum", [np.nan])[overlap_mask])) if frame_count_used > 0 else np.nan
            rows.append(
                {
                    "window_id": str(row["window_id"]),
                    "media_id": str(row["media_id"]),
                    "room_id": str(row["room_id"]),
                    "session_id": str(row["session_id"]),
                    "start_time": str(row["start_time"]),
                    "end_time": str(row["end_time"]),
                    "video_path": str(row["video_path"]),
                    "video_start_offset_sec": start_offset,
                    "duration_seconds": duration_seconds,
                    "zone_config_id": str(zone_config["zone_config_id"]),
                    "zone_id": zone_id,
                    "semantic_type": semantic_lookup.get(zone_id, zone_id),
                    "zone_area_pixels": zone_area_lookup[zone_id]["zone_area_pixels"],
                    "zone_area_fraction": zone_area_lookup[zone_id]["zone_area_fraction"],
                    "activity_mean": activity_mean,
                    "activity_std": activity_std,
                    "activity_sum": activity_sum,
                    "activity_norm": np.nan,
                    "frame_count_used": frame_count_used,
                    "quality_status": status if str(row.get("quality_status", "ok")) == "ok" else str(row.get("quality_status")),
                    "warnings": warning_text,
                }
            )
    return rows, failed_rows


def _finalize_zone_feature_df(zone_feature_df: pd.DataFrame) -> pd.DataFrame:
    finalized = prepare_time_columns(zone_feature_df)
    finalized = finalized.sort_values(["room_id", "session_id", "start_time_dt", "window_id", "zone_id"], kind="stable").reset_index(drop=True)
    total_by_window = finalized.groupby("window_id", sort=False)["activity_sum"].transform("sum")
    count_by_window = finalized.groupby("window_id", sort=False)["zone_id"].transform("count").replace(0, np.nan)
    finalized["activity_norm"] = np.where(
        pd.to_numeric(total_by_window, errors="coerce").fillna(0.0) > 0,
        pd.to_numeric(finalized["activity_sum"], errors="coerce").fillna(0.0) / total_by_window,
        1.0 / count_by_window.fillna(1.0),
    )
    finalized["activity_norm"] = pd.to_numeric(finalized["activity_norm"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    finalized = finalized.drop(columns=["start_time_dt", "end_time_dt"], errors="ignore")
    return finalized[ZONE_FEATURE_COLUMNS]


def _plot_room_level_outputs(zone_feature_df: pd.DataFrame, biomarker_df: pd.DataFrame, config: PreparationConfig) -> None:
    if zone_feature_df.empty or biomarker_df.empty:
        return
    zone_plot_df = prepare_time_columns(zone_feature_df)
    biomarker_plot_df = prepare_time_columns(biomarker_df)

    for room_id, room_zone_df in zone_plot_df.groupby("room_id", sort=False):
        figure, axis = plt.subplots(figsize=(12, 5))
        labeled_zones: set[str] = set()
        for zone_id, group_df in room_zone_df.groupby("zone_id", sort=False):
            for _, session_df in group_df.groupby("session_id", sort=False):
                ordered = session_df.sort_values("start_time_dt", kind="stable")
                label = zone_id if zone_id not in labeled_zones else None
                axis.plot(ordered["start_time_dt"], ordered["activity_mean"], label=label, linewidth=1.2)
            labeled_zones.add(zone_id)
        axis.set_title(f"Semantic Zone Activity Timeseries: {room_id}")
        axis.set_ylabel("Activity Mean")
        axis.set_xlabel("Window Start Time")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(config.reports_output_dir / f"semantic_zone_activity_timeseries_{room_id}.png", dpi=180)
        plt.close(figure)

    for room_id, room_biomarker_df in biomarker_plot_df.groupby("room_id", sort=False):
        ordered = room_biomarker_df.sort_values("start_time_dt", kind="stable")
        figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        metrics = [
            ("mobility_index", "Mobility Index", "#28536b"),
            ("spatial_freedom_index", "Spatial Freedom Index", "#5f8f3e"),
            ("occupancy_imbalance_index", "Occupancy Imbalance Index", "#c97c1f"),
            ("feeding_plus_drinking_activity_fraction", "Functional Area Activity Fraction", "#c94f3d"),
        ]
        for axis, (column, title, color) in zip(axes, metrics):
            for _, session_df in ordered.groupby("session_id", sort=False):
                session_ordered = session_df.sort_values("start_time_dt", kind="stable")
                axis.plot(session_ordered["start_time_dt"], session_ordered[column], color=color, linewidth=1.3)
            axis.set_ylabel(title)
            axis.grid(alpha=0.25)
        axes[0].set_title(f"Semantic Biomarker Timeseries: {room_id}")
        axes[-1].set_xlabel("Window Start Time")
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(config.reports_output_dir / f"semantic_biomarker_timeseries_{room_id}.png", dpi=180)
        plt.close(figure)


def _spatial_freedom(probabilities: np.ndarray) -> float:
    if probabilities.size <= 1:
        return 1.0
    mask = probabilities > 0
    entropy = -np.sum(probabilities[mask] * np.log(probabilities[mask]))
    return float(entropy / np.log(probabilities.size))


def _imbalance(probabilities: np.ndarray) -> float:
    uniform = np.repeat(1.0 / probabilities.size, probabilities.size)
    return float(0.5 * np.abs(probabilities - uniform).sum())


def _resize_frame(frame: np.ndarray, resize_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = float(resize_width) / float(width)
    resize_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)


def _failed_window_row(row: pd.Series, issue: str) -> dict:
    return {
        "stage": "video_features",
        "window_id": str(row.get("window_id", "")),
        "media_id": str(row.get("media_id", "")),
        "room_id": str(row.get("room_id", "")),
        "issue": issue,
    }


def _ordered_zone_ids(zone_config: dict) -> list[str]:
    available_ids = [str(zone.get("zone_id", "")) for zone in zone_config.get("zones", [])]
    ordered = [zone_id for zone_id in EXPLICIT_ZONE_IDS if zone_id in available_ids]
    leftovers = [zone_id for zone_id in available_ids if zone_id not in ordered]
    return ordered + leftovers


def _build_dry_run_report(window_df: pd.DataFrame, zone_configs: list[dict], enabled: bool) -> str:
    lines = [
        "# Video Feature Report",
        "",
        f"- Video feature extraction enabled: `{enabled}`",
        "- Dry run mode skipped heavy video feature extraction.",
        f"- Planned windows: {len(window_df)}",
        f"- Rooms with valid zone configs: {len(zone_configs)}",
    ]
    return "\n".join(lines) + "\n"


def _build_video_feature_report(zone_feature_df: pd.DataFrame, biomarker_df: pd.DataFrame, failed_df: pd.DataFrame) -> str:
    coverage_summary = "No biomarker rows available."
    if not biomarker_df.empty:
        coverage_df = prepare_time_columns(biomarker_df)
        coverage_summary = (
            coverage_df.groupby(["room_id", "session_id"], as_index=False)
            .agg(
                start_time=("start_time_dt", "min"),
                end_time=("end_time_dt", "max"),
                window_count=("window_id", "count"),
            )
            .sort_values(["room_id", "start_time"], kind="stable")
            .to_string(index=False)
        )
    lines = [
        "# Video Feature Report",
        "",
        "- Semantic-zone activity is activity-based and not true occupancy.",
        f"- Zone feature rows: {len(zone_feature_df)}",
        f"- Biomarker windows: {len(biomarker_df)}",
        f"- Failed windows: {len(failed_df)}",
        "",
        "## Per-Zone Activity Summary",
        "",
        "```text",
        zone_feature_df.groupby("zone_id", sort=False)["activity_mean"].agg(["count", "mean", "std", "max"]).to_string() if not zone_feature_df.empty else "No zone feature rows available.",
        "```",
        "",
        "## Biomarker Summary",
        "",
        "```text",
        biomarker_df[
            [
                "activity_mean",
                "mobility_index",
                "spatial_freedom_index",
                "occupancy_imbalance_index",
                "feeding_plus_drinking_activity_fraction",
            ]
        ].describe().to_string() if not biomarker_df.empty else "No biomarker rows available.",
        "```",
        "",
        "## Session Coverage",
        "",
        "```text",
        coverage_summary,
        "```",
    ]
    return "\n".join(lines) + "\n"
