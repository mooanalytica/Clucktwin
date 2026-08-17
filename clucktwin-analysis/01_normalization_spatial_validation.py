"""Validate session normalization and the canonical spatial assignment operator."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
SOURCE = WORKSPACE / "experiments" / "room1_normalized_zone_quality_hmm_20260727"
VERIFIED = WORKSPACE / "experiments" / "room1_growth_caretaker_dynamics_verified_fall_20260727"
NORMALIZED = SOURCE / "processed" / "normalized_multimodal_30s.parquet"
GEOMETRY_PATH = SOURCE / "audit" / "session_geometry_normalization.csv"
SESSION_METRICS_PATH = SOURCE / "tables" / "normalized_session_metrics.csv"
MEDIA_INVENTORY_PATH = VERIFIED / "audit" / "media_inventory.csv"
REFERENCE_ROOT = WORKSPACE / "data" / "metadata" / "semantic_zone_refs" / "by_session" / "Room 1"

TABLES = ROOT / "analysis" / "tables"
FIGURES = ROOT / "analysis" / "figures"
AUDIT = ROOT / "analysis" / "audit"

ZONES = ["drinking", "feeding", "resting", "open_movement"]
ZONE_COLORS = {
    "drinking": "#2C7FB8",
    "feeding": "#D99A00",
    "resting": "#4C956C",
    "open_movement": "#C05A73",
    "outside": "#666666",
}
PRIMARY_EPSILON = 0.005
PERTURB_PIXELS = [5, 10]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config(path: Path, session_id: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    configs = payload if isinstance(payload, list) else [payload]
    for config in configs:
        if str(config.get("session_id", "")) == session_id:
            return config
    raise RuntimeError(f"No zone config for {session_id} in {path}")


def zone_polygons(config: dict) -> dict[str, list[np.ndarray]]:
    output = {zone: [] for zone in ZONES}
    for item in config.get("zones", []):
        semantic = str(item.get("semantic_type") or item.get("zone_id", "")).replace("_zone", "")
        if semantic not in output:
            continue
        polygons = item.get("polygons")
        if polygons is None:
            polygons = [item.get("polygon", [])]
        for polygon in polygons:
            if polygon and len(polygon) >= 3:
                output[semantic].append(np.rint(np.asarray(polygon, dtype=float)).astype(np.int32))
    return output


def assigned_mask(config: dict, perturbation: int = 0) -> np.ndarray:
    width = int(config["image_width"])
    height = int(config["image_height"])
    polygons = zone_polygons(config)
    assigned = np.zeros((height, width), dtype=np.uint8)
    if perturbation:
        radius = abs(int(perturbation))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    for code, zone in enumerate(ZONES, start=1):
        mask = np.zeros((height, width), dtype=np.uint8)
        if polygons[zone]:
            points = []
            for polygon in polygons[zone]:
                clipped = polygon.copy()
                clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
                clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
                points.append(clipped.reshape((-1, 1, 2)))
            cv2.fillPoly(mask, points, color=1)
        if perturbation < 0:
            mask = cv2.erode(mask, kernel)
        elif perturbation > 0:
            mask = cv2.dilate(mask, kernel)
        assigned[(mask > 0) & (assigned == 0)] = code
    return assigned


def resolve_reference(session_folder: str) -> Path:
    direct = REFERENCE_ROOT / session_folder / "reference.png"
    if direct.exists():
        return direct
    inner = re.sub(r"^Room 1\s*\(|\)$", "", session_folder).strip()
    candidate = REFERENCE_ROOT / inner / "reference.png"
    if candidate.exists():
        return candidate
    key = re.sub(r"[^a-z0-9]", "", inner.lower())
    for path in REFERENCE_ROOT.glob("*/reference.png"):
        folder_key = re.sub(r"[^a-z0-9]", "", path.parent.name.lower())
        if key == folder_key or key in folder_key or folder_key in key:
            return path
    raise FileNotFoundError(f"Reference image not found for {session_folder}")


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & np.isfinite(values) & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def build_hour_balanced_daily(data: pd.DataFrame, metric_quality: dict[str, str]) -> pd.DataFrame:
    frame = data.copy()
    frame["start_dt"] = pd.to_datetime(frame["start_dt"])
    frame = frame[frame["start_dt"].dt.hour.between(7, 16)].copy()
    frame["date"] = frame["start_dt"].dt.floor("D")
    frame["hour"] = frame["start_dt"].dt.hour
    rows = []
    group_columns = ["session_id", "date", "hour"]
    for keys, group in frame.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        for metric, quality in metric_quality.items():
            effective = float(group[quality].fillna(0).clip(lower=0).sum())
            row[metric] = weighted_mean(group[metric], group[quality]) if effective >= 2.0 else np.nan
        rows.append(row)
    hourly = pd.DataFrame(rows)
    daily = hourly.groupby(["session_id", "date"], as_index=False)[list(metric_quality)].mean()
    daily["days_since_start"] = (daily["date"] - pd.Timestamp("2025-07-03")).dt.days
    return daily


def trend_rows(daily: pd.DataFrame, metrics: list[str], analysis: str) -> list[dict]:
    rows = []
    for metric in metrics:
        subset = daily[["days_since_start", metric]].dropna()
        rho, p_value = spearmanr(subset["days_since_start"], subset[metric]) if len(subset) >= 5 else (np.nan, np.nan)
        rows.append(
            {
                "analysis": analysis,
                "metric": metric,
                "days": len(subset),
                "spearman_rho": rho,
                "p_value": p_value,
                "first_14_day_mean": subset.sort_values("days_since_start")[metric].head(14).mean(),
                "last_14_day_mean": subset.sort_values("days_since_start")[metric].tail(14).mean(),
            }
        )
    return rows


def similar_camera_scale_sensitivity(daily: pd.DataFrame, geometry: pd.DataFrame) -> tuple[pd.DataFrame, list[str], float]:
    scales = geometry.drop_duplicates("session_id").set_index("session_id")["camera_linear_scale"].astype(float)
    median_scale = float(scales.median())
    stable_sessions = scales[(scales / median_scale).between(0.98, 1.02)].index.tolist()
    subset = daily[daily["session_id"].isin(stable_sessions)]
    rows = []
    rows.extend(trend_rows(subset, [f"zone_{zone}_fraction" for zone in ZONES], "similar_camera_raw_zone_fraction"))
    rows.extend(trend_rows(subset, [f"zone_{zone}_log_selectivity" for zone in ZONES], "similar_camera_area_adjusted_selectivity"))
    rows.extend(trend_rows(subset, ["flock_spread_rms", "activity_mean"], "similar_camera_raw_camera_dependent"))
    rows.extend(trend_rows(subset, ["flock_spread_camera_normalized", "video_activity_camera_normalized"], "similar_camera_adjusted"))
    return pd.DataFrame(rows), stable_sessions, median_scale


def robust_standardize(values: pd.Series) -> pd.Series:
    median = values.median()
    scale = (values.quantile(0.75) - values.quantile(0.25)) / 1.349
    if not np.isfinite(scale) or scale < 1e-9:
        scale = values.std(ddof=1)
    return (values - median) / max(float(scale), 1e-9)


def boundary_analysis(daily: pd.DataFrame, metric_pairs: list[tuple[str, str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_order = (
        daily.groupby("session_id", as_index=False)["date"].agg(["min", "max"]).reset_index().sort_values("min")
    )
    detail_rows = []
    summary_rows = []
    for endpoint, raw_metric, adjusted_metric in metric_pairs:
        metric_detail = []
        for metric, representation in [(raw_metric, "raw"), (adjusted_metric, "adjusted")]:
            standardized = daily[["session_id", "date", metric]].dropna().copy()
            standardized["value_z"] = robust_standardize(standardized[metric])
            groups = {key: group.sort_values("date") for key, group in standardized.groupby("session_id")}
            within = []
            for group in groups.values():
                within.extend(np.abs(np.diff(group["value_z"].to_numpy())).tolist())
            within_median = float(np.median(within)) if within else np.nan
            for index in range(1, len(session_order)):
                previous = session_order.iloc[index - 1]["session_id"]
                current = session_order.iloc[index]["session_id"]
                if previous not in groups or current not in groups:
                    continue
                left = groups[previous].iloc[-1]
                right = groups[current].iloc[0]
                metric_detail.append(
                    {
                        "endpoint": endpoint,
                        "representation": representation,
                        "metric": metric,
                        "previous_session": previous,
                        "current_session": current,
                        "previous_date": left["date"],
                        "current_date": right["date"],
                        "absolute_standardized_jump": abs(float(right["value_z"] - left["value_z"])),
                        "within_session_adjacent_day_median_jump": within_median,
                    }
                )
        detail = pd.DataFrame(metric_detail)
        detail_rows.extend(metric_detail)
        pivot = detail.pivot_table(
            index=["previous_session", "current_session"],
            columns="representation",
            values="absolute_standardized_jump",
        ).dropna()
        if len(pivot):
            try:
                statistic, p_value = wilcoxon(pivot["adjusted"], pivot["raw"], zero_method="wilcox")
            except ValueError:
                statistic, p_value = np.nan, 1.0
            summary_rows.append(
                {
                    "endpoint": endpoint,
                    "boundaries": len(pivot),
                    "raw_median_abs_standardized_jump": pivot["raw"].median(),
                    "adjusted_median_abs_standardized_jump": pivot["adjusted"].median(),
                    "adjusted_minus_raw_median_jump": (pivot["adjusted"] - pivot["raw"]).median(),
                    "paired_wilcoxon_statistic": statistic,
                    "paired_wilcoxon_p": p_value,
                }
            )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def pseudocount_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    base = data[["session_id", "start_dt", "quality_zone", *[f"zone_{zone}_conditional_fraction" for zone in ZONES], *[f"{zone}_semantic_area_share" for zone in ZONES]]].copy()
    rows = []
    primary_daily = {}
    for epsilon in [0.001, 0.0025, 0.005, 0.01, 0.02]:
        metric_quality = {}
        for zone in ZONES:
            metric = f"{zone}_selectivity_eps_{epsilon:g}"
            base[metric] = np.log(base[f"zone_{zone}_conditional_fraction"] + epsilon) - np.log(base[f"{zone}_semantic_area_share"] + epsilon)
            metric_quality[metric] = "quality_zone"
        daily = build_hour_balanced_daily(base, metric_quality)
        for zone in ZONES:
            metric = f"{zone}_selectivity_eps_{epsilon:g}"
            subset = daily[["date", "days_since_start", metric]].dropna()
            rho, p_value = spearmanr(subset["days_since_start"], subset[metric])
            if epsilon == PRIMARY_EPSILON:
                primary_daily[zone] = subset.set_index("date")[metric]
                agreement = 1.0
            else:
                joined = subset.set_index("date")[[metric]].join(primary_daily.get(zone, pd.Series(dtype=float).rename("primary")), how="inner")
                agreement = joined.corr(method="spearman").iloc[0, 1] if joined.shape[1] == 2 and len(joined) >= 3 else np.nan
            rows.append(
                {
                    "zone": zone,
                    "pseudocount": epsilon,
                    "days": len(subset),
                    "spearman_rho_vs_date": rho,
                    "p_value": p_value,
                    "daily_rank_agreement_with_eps_0_005": agreement,
                }
            )
    output = pd.DataFrame(rows)
    for zone in ZONES:
        primary = output[(output["zone"] == zone) & (output["pseudocount"] == PRIMARY_EPSILON)]
        if primary.empty:
            continue
        primary_metric = f"{zone}_selectivity_eps_{PRIMARY_EPSILON:g}"
        primary_frame = base[["session_id", "start_dt", "quality_zone", primary_metric]].copy()
        primary_daily[zone] = build_hour_balanced_daily(primary_frame, {primary_metric: "quality_zone"}).set_index("date")[primary_metric]
    for index, row in output.iterrows():
        if row["pseudocount"] == PRIMARY_EPSILON:
            output.loc[index, "daily_rank_agreement_with_eps_0_005"] = 1.0
            continue
        zone = row["zone"]
        metric = f"{zone}_selectivity_eps_{row['pseudocount']:g}"
        daily = build_hour_balanced_daily(base[["session_id", "start_dt", "quality_zone", metric]], {metric: "quality_zone"}).set_index("date")[metric]
        joined = pd.concat([daily.rename("candidate"), primary_daily[zone].rename("primary")], axis=1).dropna()
        output.loc[index, "daily_rank_agreement_with_eps_0_005"] = joined.corr(method="spearman").iloc[0, 1]
    return output


def sample_track_points(media: pd.Series, config: dict) -> pd.DataFrame:
    path = Path(str(media["track_path"]))
    source_width = float(media["behavior_source_width"])
    source_height = float(media["behavior_source_height"])
    target_width = int(config["image_width"])
    target_height = int(config["image_height"])
    chunks = []
    columns = ["timestamp_sec", "timestamp_bin_sec", "track_id", "x1", "y1", "x2", "y2"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250_000):
        seconds = pd.to_numeric(chunk["timestamp_bin_sec"], errors="coerce")
        keep = seconds.notna() & np.isclose(np.mod(seconds, 10.0), 0.0, atol=1e-6)
        sample = chunk.loc[keep].drop_duplicates(["timestamp_bin_sec", "track_id"])
        if len(sample):
            chunks.append(sample)
    if not chunks:
        return pd.DataFrame(columns=[*columns, "center_x", "center_y", "bottom_x", "bottom_y"])
    frame = pd.concat(chunks, ignore_index=True)
    if len(frame) > 50_000:
        frame = frame.sample(50_000, random_state=20260816)
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    frame["center_x"] = ((frame["x1"] + frame["x2"]) * 0.5 * scale_x).clip(0, target_width - 1)
    frame["center_y"] = ((frame["y1"] + frame["y2"]) * 0.5 * scale_y).clip(0, target_height - 1)
    frame["bottom_x"] = frame["center_x"]
    frame["bottom_y"] = (frame["y2"] * scale_y).clip(0, target_height - 1)
    for column in ["x1", "x2"]:
        frame[f"scaled_{column}"] = (frame[column] * scale_x).clip(0, target_width - 1)
    for column in ["y1", "y2"]:
        frame[f"scaled_{column}"] = (frame[column] * scale_y).clip(0, target_height - 1)
    return frame


def classify(mask: np.ndarray, x: pd.Series, y: pd.Series) -> np.ndarray:
    xi = np.rint(x.to_numpy()).astype(int).clip(0, mask.shape[1] - 1)
    yi = np.rint(y.to_numpy()).astype(int).clip(0, mask.shape[0] - 1)
    return mask[yi, xi]


def resolve_raw_video(media: pd.Series) -> Path | None:
    declared = Path(str(media.get("raw_video_path", "")))
    if declared.is_file():
        return declared
    local = WORKSPACE / "data" / "raw" / "video" / "Room 1" / str(media["session_folder"]) / str(media["video_file"])
    return local if local.is_file() else None


def read_matching_frame(video_path: Path, timestamp_sec: float, width: int, height: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        return None
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def find_assignment_example(media: pd.DataFrame, geometry_map: pd.DataFrame) -> dict | None:
    candidates = media.copy()
    candidates["start_time"] = pd.to_datetime(candidates["start_time"])
    candidates = candidates[
        (candidates["start_time"] >= pd.Timestamp("2025-08-16"))
        & candidates["start_time"].dt.hour.between(8, 15)
    ].sort_values("start_time")
    zone_names = ["outside", *ZONES]
    for row in candidates.itertuples(index=False):
        if row.session_id not in geometry_map.index:
            continue
        media_row = pd.Series(row._asdict())
        video_path = resolve_raw_video(media_row)
        if video_path is None:
            continue
        geometry_row = geometry_map.loc[row.session_id]
        config = load_config(Path(str(geometry_row["zone_configs_path"])), row.session_id)
        points = sample_track_points(media_row, config)
        if points.empty:
            continue
        mask = assigned_mask(config, 0)
        center = classify(mask, points["center_x"], points["center_y"])
        bottom = classify(mask, points["bottom_x"], points["bottom_y"])
        differing = points.loc[center != bottom].copy()
        if differing.empty:
            continue
        differing["center_code"] = center[center != bottom]
        differing["bottom_code"] = bottom[center != bottom]
        differing["box_area"] = (
            (differing["scaled_x2"] - differing["scaled_x1"])
            * (differing["scaled_y2"] - differing["scaled_y1"])
        )
        duration = float(media_row.get("duration_seconds", np.inf))
        differing = differing[
            differing["timestamp_sec"].between(60.0, duration - 60.0)
            & differing["box_area"].between(1_000, 40_000)
        ].sort_values("box_area", ascending=False)
        for _, point in differing.drop_duplicates("timestamp_bin_sec").head(20).iterrows():
            frame = read_matching_frame(
                video_path,
                float(point["timestamp_sec"]),
                int(config["image_width"]),
                int(config["image_height"]),
            )
            if frame is None or cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() < 20:
                continue
            return {
                "session_id": row.session_id,
                "video_file": row.video_file,
                "video_path": str(video_path),
                "media_key": str(media_row.get("media_key", "")),
                "media_signature": str(media_row.get("media_signature", "")),
                "timestamp_sec": float(point["timestamp_sec"]),
                "timestamp_bin_sec": float(point["timestamp_bin_sec"]),
                "track_id": int(point["track_id"]),
                "point": point.to_dict(),
                "center_code": int(point["center_code"]),
                "bottom_code": int(point["bottom_code"]),
                "center_zone": zone_names[int(point["center_code"])],
                "bottom_zone": zone_names[int(point["bottom_code"])],
                "config": config,
                "frame": frame,
            }
    return None


def assignment_and_polygon_sensitivity(geometry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    media = pd.read_csv(MEDIA_INVENTORY_PATH, low_memory=False)
    media = media[(media["primary_media_eligible"].astype(str).str.lower() == "true") & (media["track_exists"].astype(str).str.lower() == "true")]
    media["start_time"] = pd.to_datetime(media["start_time"])
    selected = media.sort_values("start_time").groupby("session_id", as_index=False).first()
    geometry_map = geometry.set_index("session_id")
    assignment_rows = []
    perturbation_rows = []
    for row in selected.itertuples(index=False):
        if row.session_id not in geometry_map.index:
            continue
        geometry_row = geometry_map.loc[row.session_id]
        config_path = Path(str(geometry_row["zone_configs_path"]))
        config = load_config(config_path, row.session_id)
        points = sample_track_points(pd.Series(row._asdict()), config)
        if points.empty:
            continue
        base = assigned_mask(config, 0)
        center = classify(base, points["center_x"], points["center_y"])
        bottom = classify(base, points["bottom_x"], points["bottom_y"])
        agreement = center == bottom
        assignment_rows.append(
            {
                "session_id": row.session_id,
                "video_file": row.video_file,
                "sampled_detections": len(points),
                "center_bottom_agreement": agreement.mean(),
                "center_outside_fraction": (center == 0).mean(),
                "bottom_center_outside_fraction": (bottom == 0).mean(),
            }
        )
        for code, zone in enumerate(["outside", *ZONES]):
            assignment_rows[-1][f"center_{zone}_fraction"] = (center == code).mean()
            assignment_rows[-1][f"bottom_center_{zone}_fraction"] = (bottom == code).mean()
        for perturbation in [-10, -5, 5, 10]:
            altered = assigned_mask(config, perturbation)
            altered_code = classify(altered, points["bottom_x"], points["bottom_y"])
            perturbation_rows.append(
                {
                    "session_id": row.session_id,
                    "video_file": row.video_file,
                    "sampled_detections": len(points),
                    "perturbation_pixels": perturbation,
                    "assignment_agreement_with_original": (altered_code == bottom).mean(),
                    "outside_fraction_original": (bottom == 0).mean(),
                    "outside_fraction_perturbed": (altered_code == 0).mean(),
                }
            )
    example = find_assignment_example(media, geometry_map)
    return pd.DataFrame(assignment_rows), pd.DataFrame(perturbation_rows), example


def build_reference_manifest(geometry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in geometry.itertuples(index=False):
        reference = resolve_reference(row.session_folder)
        config_path = Path(row.zone_configs_path)
        session_config = load_config(config_path, row.session_id)
        rows.append(
            {
                "session_id": row.session_id,
                "session_folder": row.session_folder,
                "session_start": row.session_start,
                "reference_image": str(reference),
                "reference_sha256": sha256(reference),
                "zone_config": str(config_path),
                "zone_config_object_sha256": canonical_json_sha256(session_config),
                "zone_config_container_sha256": sha256(config_path),
                "geometry_valid": row.geometry_valid,
                "semantic_area_fraction": row.semantic_area_fraction,
                "camera_linear_scale": row.camera_linear_scale,
                **{f"{zone}_semantic_area_share": getattr(row, f"{zone}_semantic_area_share") for zone in ZONES},
            }
        )
    return pd.DataFrame(rows)


def draw_overlay(axis, image: np.ndarray, config: dict, title: str) -> None:
    axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    polygons = zone_polygons(config)
    for zone in ZONES:
        for polygon in polygons[zone]:
            closed = np.vstack([polygon, polygon[0]])
            axis.plot(closed[:, 0], closed[:, 1], color=ZONE_COLORS[zone], linewidth=2.0, label=zone.replace("_", " "))
    axis.set_title(title, fontsize=8)
    axis.axis("off")


def make_session_figure(geometry: pd.DataFrame, example: dict | None) -> None:
    ordered = geometry.sort_values("semantic_area_fraction")
    picks = [ordered.iloc[0], ordered.iloc[len(ordered) // 2], ordered.iloc[-1]]
    labels = ["Minimum coverage", "Median coverage", "Maximum coverage"]
    fig = plt.figure(figsize=(14, 7.6))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.1], hspace=0.14, wspace=0.08)
    legend_handles = None
    for column, (row, label) in enumerate(zip(picks, labels)):
        reference = resolve_reference(row["session_folder"])
        image = cv2.imread(str(reference))
        config = load_config(Path(row["zone_configs_path"]), row["session_id"])
        top = fig.add_subplot(grid[0, column])
        top.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        top.set_title(f"{label}\n{row['session_id']}\ncoverage={row['semantic_area_fraction']:.3f}, scale={row['camera_linear_scale']:.3f}", fontsize=8)
        top.axis("off")
        bottom = fig.add_subplot(grid[1, column])
        draw_overlay(bottom, image, config, "Session polygons")
        if legend_handles is None:
            legend_handles, legend_labels = bottom.get_legend_handles_labels()

    assignment_axis = fig.add_subplot(grid[0, 3])
    if example is not None:
        image = example["frame"]
        point = example["point"]
        assignment_axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        polygons = zone_polygons(example["config"])
        for code in sorted({example["center_code"], example["bottom_code"]}):
            if code == 0:
                continue
            zone = ZONES[code - 1]
            for polygon in polygons[zone]:
                closed = np.vstack([polygon, polygon[0]])
                assignment_axis.plot(
                    closed[:, 0],
                    closed[:, 1],
                    color=ZONE_COLORS[zone],
                    linewidth=2.0,
                    label=f"{zone.replace('_', ' ')} boundary",
                )
        x1, x2 = point["scaled_x1"], point["scaled_x2"]
        y1, y2 = point["scaled_y1"], point["scaled_y2"]
        assignment_axis.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="white", linewidth=2))
        assignment_axis.scatter([point["center_x"]], [point["center_y"]], s=45, color="#D95F02", label="box center")
        assignment_axis.scatter([point["bottom_x"]], [point["bottom_y"]], s=55, marker="x", color="#1B9E77", linewidth=2.2, label="bottom-center")
        pad = max(x2 - x1, y2 - y1) * 2.5
        assignment_axis.set_xlim(max(0, x1 - pad), min(image.shape[1], x2 + pad))
        assignment_axis.set_ylim(min(image.shape[0], y2 + pad), max(0, y1 - pad))
        assignment_axis.set_title(
            "Matched detection frame\n"
            f"center: {example['center_zone'].replace('_', ' ')}; "
            f"bottom: {example['bottom_zone'].replace('_', ' ')}",
            fontsize=8,
        )
        handles, labels = assignment_axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        assignment_axis.legend(unique.values(), unique.keys(), fontsize=6.2, loc="upper right")
    assignment_axis.axis("off")

    audit_axis = fig.add_subplot(grid[1, 3])
    audit = geometry.sort_values("session_start").reset_index(drop=True)
    audit_axis.plot(audit.index, audit["semantic_area_fraction"], color="#222222", marker="o", markersize=2.5, linewidth=1.1, label="semantic/image")
    for zone in ZONES:
        audit_axis.plot(audit.index, audit[f"{zone}_semantic_area_share"], color=ZONE_COLORS[zone], linewidth=0.9, label=zone.replace("_", " "))
    audit_axis.set_xlabel("Chronological session index")
    audit_axis.set_ylabel("Area share")
    audit_axis.set_title("All-session geometry audit", fontsize=9)
    audit_axis.grid(alpha=0.2)
    audit_axis.legend(fontsize=6.5, ncol=2)
    if legend_handles:
        unique = dict(zip(legend_labels, legend_handles))
        fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.36, -0.01))
    fig.suptitle("Session-specific geometry and canonical bottom-center assignment", fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.02, right=0.99)
    fig.savefig(FIGURES / "fig_3_session_normalization_validation.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "fig_3_session_normalization_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def make_sensitivity_figure(boundary_summary: pd.DataFrame, pseudocount: pd.DataFrame, assignment: pd.DataFrame, perturbation: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    positions = np.arange(len(boundary_summary))
    width = 0.38
    axes[0, 0].bar(positions - width / 2, boundary_summary["raw_median_abs_standardized_jump"], width, label="raw", color="#999999")
    axes[0, 0].bar(positions + width / 2, boundary_summary["adjusted_median_abs_standardized_jump"], width, label="adjusted", color="#2C7FB8")
    axes[0, 0].set_xticks(positions, boundary_summary["endpoint"], rotation=35, ha="right", fontsize=8)
    axes[0, 0].set_ylabel("Median absolute standardized jump")
    axes[0, 0].set_title("Session-boundary discontinuity")
    axes[0, 0].legend()

    for zone in ZONES:
        subset = pseudocount[pseudocount["zone"] == zone]
        axes[0, 1].plot(subset["pseudocount"], subset["spearman_rho_vs_date"], marker="o", label=zone.replace("_", " "), color=ZONE_COLORS[zone])
    axes[0, 1].axvline(PRIMARY_EPSILON, color="#333333", linestyle="--", linewidth=1)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Pseudocount")
    axes[0, 1].set_ylabel("Longitudinal Spearman rho")
    axes[0, 1].set_title("Pseudocount sensitivity")
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].hist(assignment["center_bottom_agreement"], bins=np.linspace(0.7, 1.0, 13), color="#4C956C", edgecolor="white")
    axes[1, 0].axvline(assignment["center_bottom_agreement"].median(), color="#222222", linestyle="--")
    axes[1, 0].set_xlabel("Center vs bottom-center assignment agreement")
    axes[1, 0].set_ylabel("Sessions")
    axes[1, 0].set_title("Assignment-operator sensitivity")

    grouped = perturbation.groupby("perturbation_pixels")["assignment_agreement_with_original"].agg(["median", "min", "max"]).reset_index()
    axes[1, 1].errorbar(grouped["perturbation_pixels"], grouped["median"], yerr=np.vstack([grouped["median"] - grouped["min"], grouped["max"] - grouped["median"]]), marker="o", capsize=4, color="#C05A73")
    axes[1, 1].axhline(1, color="#777777", linewidth=0.8)
    axes[1, 1].set_xlabel("Polygon erosion (-) or dilation (+), pixels")
    axes[1, 1].set_ylabel("Agreement with original assignment")
    axes[1, 1].set_title("Polygon-boundary perturbation")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_s1_normalization_sensitivity.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "fig_s1_normalization_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for directory in [TABLES, FIGURES, AUDIT]:
        directory.mkdir(parents=True, exist_ok=True)
    columns = [
        "session_id", "start_dt", "quality_zone", "quality_position", "quality_video",
        *[f"zone_{zone}_fraction" for zone in ZONES],
        *[f"zone_{zone}_conditional_fraction" for zone in ZONES],
        *[f"zone_{zone}_log_selectivity" for zone in ZONES],
        *[f"{zone}_semantic_area_share" for zone in ZONES],
        "flock_spread_rms", "flock_spread_camera_normalized", "activity_mean", "video_activity_camera_normalized",
    ]
    data = pd.read_parquet(NORMALIZED, columns=columns)
    metric_quality = {
        **{f"zone_{zone}_fraction": "quality_zone" for zone in ZONES},
        **{f"zone_{zone}_log_selectivity": "quality_zone" for zone in ZONES},
        "flock_spread_rms": "quality_position",
        "flock_spread_camera_normalized": "quality_position",
        "activity_mean": "quality_video",
        "video_activity_camera_normalized": "quality_video",
    }
    daily = build_hour_balanced_daily(data, metric_quality)
    daily.to_csv(TABLES / "normalization_daily_hour_balanced.csv", index=False)
    comparison_rows = []
    comparison_rows.extend(trend_rows(daily, [f"zone_{zone}_fraction" for zone in ZONES], "raw_zone_fraction"))
    comparison_rows.extend(trend_rows(daily, [f"zone_{zone}_log_selectivity" for zone in ZONES], "area_adjusted_zone_selectivity"))
    comparison_rows.extend(trend_rows(daily, ["flock_spread_rms", "activity_mean"], "raw_camera_dependent"))
    comparison_rows.extend(trend_rows(daily, ["flock_spread_camera_normalized", "video_activity_camera_normalized"], "camera_adjusted"))
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(TABLES / "normalization_raw_area_camera_comparison.csv", index=False)

    pairs = [
        *[(zone, f"zone_{zone}_fraction", f"zone_{zone}_log_selectivity") for zone in ZONES],
        ("flock spread", "flock_spread_rms", "flock_spread_camera_normalized"),
        ("video activity", "activity_mean", "video_activity_camera_normalized"),
    ]
    boundary_detail, boundary_summary = boundary_analysis(daily, pairs)
    boundary_detail.to_csv(TABLES / "session_boundary_discontinuity_detail.csv", index=False)
    boundary_summary.to_csv(TABLES / "session_boundary_discontinuity_summary.csv", index=False)
    pseudocount = pseudocount_sensitivity(data)
    pseudocount.to_csv(TABLES / "pseudocount_sensitivity.csv", index=False)

    geometry = pd.read_csv(GEOMETRY_PATH, low_memory=False)
    geometry["session_start"] = pd.to_datetime(geometry["session_start"])
    stable_camera, stable_sessions, median_camera_scale = similar_camera_scale_sensitivity(daily, geometry)
    stable_camera.to_csv(TABLES / "similar_camera_scale_session_sensitivity.csv", index=False)
    manifest = build_reference_manifest(geometry)
    manifest.to_csv(AUDIT / "session_reference_manifest.csv", index=False)
    assignment, perturbation, example = assignment_and_polygon_sensitivity(geometry)
    assignment.to_csv(TABLES / "center_bottom_assignment_sensitivity.csv", index=False)
    perturbation.to_csv(TABLES / "polygon_perturbation_sensitivity.csv", index=False)
    if example is not None:
        pd.DataFrame(
            [
                {
                    "session_id": example["session_id"],
                    "video_file": example["video_file"],
                    "video_path": example["video_path"],
                    "media_key": example["media_key"],
                    "media_signature": example["media_signature"],
                    "timestamp_sec": example["timestamp_sec"],
                    "timestamp_bin_sec": example["timestamp_bin_sec"],
                    "track_id": example["track_id"],
                    "center_zone": example["center_zone"],
                    "bottom_zone": example["bottom_zone"],
                }
            ]
        ).to_csv(AUDIT / "figure_3_assignment_example.csv", index=False)
    make_session_figure(geometry, example)
    make_sensitivity_figure(boundary_summary, pseudocount, assignment, perturbation)

    summary = {
        "windows": int(len(data)),
        "sessions": int(geometry["session_id"].nunique()),
        "reference_images_hashed": int(len(manifest)),
        "center_bottom_sampled_sessions": int(len(assignment)),
        "center_bottom_sampled_detections": int(assignment["sampled_detections"].sum()),
        "center_bottom_median_agreement": float(assignment["center_bottom_agreement"].median()),
        "center_bottom_min_agreement": float(assignment["center_bottom_agreement"].min()),
        "polygon_perturbation_median_agreement": {
            str(key): float(value)
            for key, value in perturbation.groupby("perturbation_pixels")["assignment_agreement_with_original"].median().items()
        },
        "pseudocount_min_rank_agreement": float(pseudocount["daily_rank_agreement_with_eps_0_005"].min()),
        "similar_camera_scale_definition": "session camera linear scale within 2% of the all-session median",
        "similar_camera_scale_median": median_camera_scale,
        "similar_camera_scale_sessions": len(stable_sessions),
        "similar_camera_scale_sensitivity": stable_camera.to_dict("records"),
        "boundary_summary": boundary_summary.to_dict("records"),
        "second_polygon_annotator": "not available; all 41 primary annotations were hash-audited, and operator/perturbation sensitivity was assessed instead",
        "route_annotation": "entry-door/caretaker-route coordinates were not available and were not inferred from images",
    }
    (ROOT / "analysis" / "normalization_validation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
