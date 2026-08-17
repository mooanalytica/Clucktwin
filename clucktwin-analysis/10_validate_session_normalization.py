"""Validate session normalization against geometry, rather than temporal smoothness.

The primary test holds a biologically plausible zone-density profile fixed while
replaying it through every observed session geometry. This isolates the nuisance
factor that session normalization is intended to remove. Secondary analyses test
signed coupling between session-boundary feature changes and geometry changes,
and summarize existing polygon-perturbation checks on real detections.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
SOURCE = WORKSPACE / "experiments" / "room1_normalized_zone_quality_hmm_20260727"
LEGACY_VALIDATION = ROOT / "analysis" / "tables"

NORMALIZED_PATH = SOURCE / "processed" / "normalized_multimodal_30s.parquet"
GEOMETRY_PATH = SOURCE / "audit" / "session_geometry_normalization.csv"
PERTURBATION_PATH = LEGACY_VALIDATION / "polygon_perturbation_sensitivity.csv"
ASSIGNMENT_PATH = LEGACY_VALIDATION / "center_bottom_assignment_sensitivity.csv"
SESSION_METRICS_PATH = SOURCE / "tables" / "normalized_session_metrics.csv"
HMM_SENSITIVITY_PATH = LEGACY_VALIDATION / "hmm_k5_modality_weighting_normalization_sensitivity.csv"

TABLES = ROOT / "analysis" / "tables"
FIGURES = ROOT / "analysis" / "figures"
AUDIT = ROOT / "analysis" / "audit"

ZONES = ["drinking", "feeding", "resting", "open_movement"]
ZONE_LABELS = {
    "drinking": "Drinking",
    "feeding": "Feeding",
    "resting": "Resting",
    "open_movement": "Open movement",
}
PSEUDOCOUNT = 0.005
BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 20260817

ZONE_RAW = [f"zone_{zone}_conditional_fraction" for zone in ZONES]
ZONE_ADJUSTED = [f"zone_{zone}_log_selectivity" for zone in ZONES]
ZONE_AREAS = [f"{zone}_semantic_area_share" for zone in ZONES]
METRIC_QUALITY = {
    **{column: "quality_zone" for column in ZONE_RAW + ZONE_ADJUSTED},
    "flock_spread_rms": "quality_position",
    "flock_spread_camera_normalized": "quality_position",
    "activity_mean": "quality_video",
    "video_activity_camera_normalized": "quality_video",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bh_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return result
    count = len(valid)
    adjusted = valid.to_numpy() * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[valid.index] = np.minimum(adjusted, 1.0)
    return result


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    value_array = values.to_numpy(dtype=float)
    weight_array = weights.to_numpy(dtype=float)
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0)
    if not valid.any() or weight_array[valid].sum() < 2.0:
        return np.nan
    return float(np.average(value_array[valid], weights=weight_array[valid]))


def build_hour_balanced_daily() -> pd.DataFrame:
    columns = [
        "session_id",
        "start_dt",
        "is_daylight_schedule",
        "caretaker_or_management_overlap",
        *METRIC_QUALITY,
        *sorted(set(METRIC_QUALITY.values())),
    ]
    frame = (
        pl.scan_parquet(NORMALIZED_PATH)
        .select(columns)
        .filter(
            pl.col("is_daylight_schedule").fill_null(False)
            & ~pl.col("caretaker_or_management_overlap").fill_null(False)
        )
        .collect()
        .to_pandas()
    )
    frame["start_dt"] = pd.to_datetime(frame["start_dt"])
    frame["date"] = frame["start_dt"].dt.floor("D")
    frame["hour"] = frame["start_dt"].dt.hour

    hourly_rows: list[dict] = []
    for keys, group in frame.groupby(["session_id", "date", "hour"], sort=True):
        row = dict(zip(["session_id", "date", "hour"], keys))
        for metric, quality in METRIC_QUALITY.items():
            row[metric] = weighted_mean(group[metric], group[quality])
        hourly_rows.append(row)

    hourly = pd.DataFrame(hourly_rows)
    daily = (
        hourly.groupby(["session_id", "date"], as_index=False)[list(METRIC_QUALITY)]
        .mean()
        .sort_values(["date", "session_id"])
    )
    daily["days_since_start"] = (daily["date"] - daily["date"].min()).dt.days
    return daily


def variance_decomposition(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balanced two-way decomposition for profile x geometry x feature arrays."""
    grand_mean = values.mean(axis=(0, 1))
    profile_mean = values.mean(axis=1)
    geometry_mean = values.mean(axis=0)
    total_ss = ((values - grand_mean) ** 2).sum(axis=(0, 1))
    geometry_ss = values.shape[0] * ((geometry_mean - grand_mean) ** 2).sum(axis=0)
    profile_ss = values.shape[1] * ((profile_mean - grand_mean) ** 2).sum(axis=0)
    interaction_ss = np.maximum(total_ss - geometry_ss - profile_ss, 0.0)
    denominator = np.maximum(total_ss, np.finfo(float).eps)
    return geometry_ss / denominator, profile_ss / denominator, interaction_ss / denominator


def build_zone_counterfactual(
    daily: pd.DataFrame,
    geometry: pd.DataFrame,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    geometry_index = geometry.set_index("session_id")
    profiles: list[np.ndarray] = []
    profile_rows: list[dict] = []
    for row in daily.dropna(subset=ZONE_RAW).itertuples(index=False):
        if row.session_id not in geometry_index.index:
            continue
        occupancy = np.asarray([getattr(row, column) for column in ZONE_RAW], dtype=float)
        if not np.isfinite(occupancy).all() or occupancy.sum() <= 0:
            continue
        occupancy = occupancy / occupancy.sum()
        source_area = geometry_index.loc[row.session_id, ZONE_AREAS].to_numpy(dtype=float)
        density = (occupancy + epsilon) / (source_area + epsilon)
        density /= np.exp(np.mean(np.log(density)))
        profiles.append(density)
        profile_rows.append(
            {
                "profile_id": len(profile_rows),
                "source_session_id": row.session_id,
                "date": row.date,
                **{f"latent_density_{zone}": density[index] for index, zone in enumerate(ZONES)},
            }
        )

    densities = np.stack(profiles)
    target_areas = geometry[ZONE_AREAS].to_numpy(dtype=float)
    expected = densities[:, None, :] * target_areas[None, :, :]
    expected /= expected.sum(axis=2, keepdims=True)
    adjusted = np.log(expected + epsilon) - np.log(target_areas[None, :, :] + epsilon)
    return expected, adjusted, pd.DataFrame(profile_rows)


def counterfactual_analysis(
    daily: pd.DataFrame,
    geometry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw, adjusted, profiles = build_zone_counterfactual(daily, geometry, PSEUDOCOUNT)
    raw_geometry, raw_profile, raw_interaction = variance_decomposition(raw)
    adjusted_geometry, adjusted_profile, adjusted_interaction = variance_decomposition(adjusted)

    rows = []
    for index, zone in enumerate(ZONES):
        rows.append(
            {
                "endpoint": zone,
                "raw_geometry_eta2": raw_geometry[index],
                "normalized_geometry_eta2": adjusted_geometry[index],
                "geometry_eta2_reduction_fraction": 1.0 - adjusted_geometry[index] / raw_geometry[index],
                "raw_profile_eta2": raw_profile[index],
                "normalized_profile_eta2": adjusted_profile[index],
                "raw_interaction_eta2": raw_interaction[index],
                "normalized_interaction_eta2": adjusted_interaction[index],
            }
        )
    summary = pd.DataFrame(rows)

    rng = np.random.default_rng(RANDOM_SEED)
    reductions = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, raw.shape[0], raw.shape[0])
        raw_eta, _, _ = variance_decomposition(raw[indices])
        adjusted_eta, _, _ = variance_decomposition(adjusted[indices])
        reductions[replicate] = 1.0 - adjusted_eta.mean() / raw_eta.mean()

    sensitivity_rows = []
    for epsilon in [0.0001, 0.001, 0.005, 0.01, 0.02]:
        sensitivity_raw, sensitivity_adjusted, _ = build_zone_counterfactual(daily, geometry, epsilon)
        raw_eta, _, _ = variance_decomposition(sensitivity_raw)
        adjusted_eta, _, _ = variance_decomposition(sensitivity_adjusted)
        sensitivity_rows.append(
            {
                "pseudocount": epsilon,
                "mean_raw_geometry_eta2": raw_eta.mean(),
                "mean_normalized_geometry_eta2": adjusted_eta.mean(),
                "mean_geometry_eta2_reduction_fraction": 1.0 - adjusted_eta.mean() / raw_eta.mean(),
                "median_raw_geometry_eta2": np.median(raw_eta),
                "median_normalized_geometry_eta2": np.median(adjusted_eta),
            }
        )

    aggregate = {
        "profiles": int(raw.shape[0]),
        "source_sessions": int(profiles["source_session_id"].nunique()),
        "target_geometries": int(raw.shape[1]),
        "mean_raw_geometry_eta2": float(raw_geometry.mean()),
        "mean_normalized_geometry_eta2": float(adjusted_geometry.mean()),
        "mean_geometry_eta2_reduction_fraction": float(1.0 - adjusted_geometry.mean() / raw_geometry.mean()),
        "median_raw_geometry_eta2": float(np.median(raw_geometry)),
        "median_normalized_geometry_eta2": float(np.median(adjusted_geometry)),
        "bootstrap_reduction_ci_low": float(np.quantile(reductions, 0.025)),
        "bootstrap_reduction_ci_high": float(np.quantile(reductions, 0.975)),
        "bootstrap_probability_reduction_le_zero": float(np.mean(reductions <= 0)),
    }
    return summary, pd.DataFrame(sensitivity_rows), profiles, aggregate


def safe_correlation(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return np.nan, np.nan
    result = pearsonr(x[valid], y[valid]) if method == "pearson" else spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue)


def boundary_geometry_coupling(
    daily: pd.DataFrame,
    geometry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    geometry_index = geometry.set_index("session_id")
    order = daily.groupby("session_id")["date"].agg(["min", "max"]).sort_values("min")
    detail_rows = []
    for boundary_index in range(1, len(order)):
        previous_session = order.index[boundary_index - 1]
        current_session = order.index[boundary_index]
        previous = daily[daily["session_id"] == previous_session].sort_values("date").iloc[-1]
        current = daily[daily["session_id"] == current_session].sort_values("date").iloc[0]
        for zone, raw_column, adjusted_column, area_column in zip(ZONES, ZONE_RAW, ZONE_ADJUSTED, ZONE_AREAS):
            detail_rows.append(
                {
                    "boundary_id": boundary_index,
                    "previous_session": previous_session,
                    "current_session": current_session,
                    "previous_date": previous["date"],
                    "current_date": current["date"],
                    "date_gap_days": (current["date"] - previous["date"]).days,
                    "zone": zone,
                    "delta_geometry_area_share": (
                        geometry_index.loc[current_session, area_column]
                        - geometry_index.loc[previous_session, area_column]
                    ),
                    "delta_raw_conditional_fraction": current[raw_column] - previous[raw_column],
                    "delta_normalized_log_selectivity": current[adjusted_column] - previous[adjusted_column],
                }
            )
    detail = pd.DataFrame(detail_rows).replace([np.inf, -np.inf], np.nan)

    summary_rows = []
    for zone, subset in detail.groupby("zone", sort=False):
        raw_pearson, raw_pearson_p = safe_correlation(
            subset["delta_geometry_area_share"], subset["delta_raw_conditional_fraction"], "pearson"
        )
        adjusted_pearson, adjusted_pearson_p = safe_correlation(
            subset["delta_geometry_area_share"], subset["delta_normalized_log_selectivity"], "pearson"
        )
        raw_spearman, raw_spearman_p = safe_correlation(
            subset["delta_geometry_area_share"], subset["delta_raw_conditional_fraction"], "spearman"
        )
        adjusted_spearman, adjusted_spearman_p = safe_correlation(
            subset["delta_geometry_area_share"], subset["delta_normalized_log_selectivity"], "spearman"
        )
        summary_rows.append(
            {
                "zone": zone,
                "boundaries": int(subset.dropna().shape[0]),
                "raw_pearson_r": raw_pearson,
                "raw_pearson_p": raw_pearson_p,
                "normalized_pearson_r": adjusted_pearson,
                "normalized_pearson_p": adjusted_pearson_p,
                "raw_spearman_rho": raw_spearman,
                "raw_spearman_p": raw_spearman_p,
                "normalized_spearman_rho": adjusted_spearman,
                "normalized_spearman_p": adjusted_spearman_p,
            }
        )
    summary = pd.DataFrame(summary_rows)
    for column in ["raw_pearson_p", "normalized_pearson_p", "raw_spearman_p", "normalized_spearman_p"]:
        summary[f"{column}_bh"] = bh_adjust(summary[column])

    arrays = {}
    for value_column in ["delta_raw_conditional_fraction", "delta_normalized_log_selectivity", "delta_geometry_area_share"]:
        arrays[value_column] = (
            detail.pivot(index="boundary_id", columns="zone", values=value_column)
            .reindex(columns=ZONES)
            .to_numpy(dtype=float)
        )
    valid_rows = np.isfinite(np.stack(list(arrays.values()), axis=2)).all(axis=(1, 2))
    raw_values = arrays["delta_raw_conditional_fraction"][valid_rows]
    adjusted_values = arrays["delta_normalized_log_selectivity"][valid_rows]
    geometry_values = arrays["delta_geometry_area_share"][valid_rows]

    def mean_zone_correlation(feature_values: np.ndarray, geometry_matrix: np.ndarray) -> float:
        return float(np.mean([pearsonr(feature_values[:, index], geometry_matrix[:, index]).statistic for index in range(len(ZONES))]))

    observed_raw = mean_zone_correlation(raw_values, geometry_values)
    observed_adjusted = mean_zone_correlation(adjusted_values, geometry_values)
    rng = np.random.default_rng(RANDOM_SEED + 1)
    reduction = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, raw_values.shape[0], raw_values.shape[0])
        reduction[replicate] = (
            mean_zone_correlation(raw_values[indices], geometry_values[indices])
            - mean_zone_correlation(adjusted_values[indices], geometry_values[indices])
        )
    aggregate = {
        "complete_boundaries": int(raw_values.shape[0]),
        "mean_zone_raw_pearson_r": observed_raw,
        "mean_zone_normalized_pearson_r": observed_adjusted,
        "mean_zone_signed_correlation_reduction": observed_raw - observed_adjusted,
        "mean_absolute_raw_spearman_rho": float(summary["raw_spearman_rho"].abs().mean()),
        "mean_absolute_normalized_spearman_rho": float(summary["normalized_spearman_rho"].abs().mean()),
        "bootstrap_reduction_ci_low": float(np.quantile(reduction, 0.025)),
        "bootstrap_reduction_ci_high": float(np.quantile(reduction, 0.975)),
        "bootstrap_probability_reduction_le_zero": float(np.mean(reduction <= 0)),
    }
    return detail, summary, aggregate


def scale_equivariance(daily: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    specifications = [
        ("flock_spread", "flock_spread_camera_normalized", "camera_linear_scale"),
        ("video_activity", "video_activity_camera_normalized", "semantic_area_fraction"),
    ]
    rows = []
    for endpoint, normalized_column, geometry_column in specifications:
        profile = daily[normalized_column].dropna().to_numpy(dtype=float)
        factor = geometry[geometry_column].to_numpy(dtype=float)
        raw_counterfactual = profile[:, None] * factor[None, :]
        recovered = raw_counterfactual / factor[None, :]
        raw_eta, raw_profile, raw_interaction = variance_decomposition(raw_counterfactual[:, :, None])
        adjusted_eta, adjusted_profile, adjusted_interaction = variance_decomposition(recovered[:, :, None])
        rows.append(
            {
                "endpoint": endpoint,
                "profiles": len(profile),
                "geometries": len(factor),
                "raw_geometry_eta2": raw_eta[0],
                "normalized_geometry_eta2": adjusted_eta[0],
                "raw_profile_eta2": raw_profile[0],
                "normalized_profile_eta2": adjusted_profile[0],
                "raw_interaction_eta2": raw_interaction[0],
                "normalized_interaction_eta2": adjusted_interaction[0],
                "maximum_absolute_recovery_error": float(np.max(np.abs(recovered - profile[:, None]))),
            }
        )
    return pd.DataFrame(rows)


def camera_scale_empirical_diagnostics(
    daily: pd.DataFrame,
    geometry: pd.DataFrame,
) -> pd.DataFrame:
    """Test whether observed scale-sensitive endpoints track the proposed proxy."""
    geometry_index = geometry.set_index("session_id")
    order = daily.groupby("session_id")["date"].agg(["min", "max"]).sort_values("min")
    specifications = [
        (
            "flock_spread",
            "flock_spread_rms",
            "flock_spread_camera_normalized",
            "camera_linear_scale",
        ),
        (
            "video_activity",
            "activity_mean",
            "video_activity_camera_normalized",
            "semantic_area_fraction",
        ),
    ]
    rows = []
    for endpoint, raw_column, normalized_column, geometry_column in specifications:
        changes = []
        for boundary_index in range(1, len(order)):
            previous_session = order.index[boundary_index - 1]
            current_session = order.index[boundary_index]
            previous = daily[daily["session_id"] == previous_session].sort_values("date").iloc[-1]
            current = daily[daily["session_id"] == current_session].sort_values("date").iloc[0]
            changes.append(
                {
                    "geometry": geometry_index.loc[current_session, geometry_column]
                    - geometry_index.loc[previous_session, geometry_column],
                    "raw": current[raw_column] - previous[raw_column],
                    "normalized": current[normalized_column] - previous[normalized_column],
                }
            )
        change_frame = pd.DataFrame(changes)
        raw_pearson, raw_pearson_p = safe_correlation(change_frame["geometry"], change_frame["raw"], "pearson")
        adjusted_pearson, adjusted_pearson_p = safe_correlation(
            change_frame["geometry"], change_frame["normalized"], "pearson"
        )
        raw_spearman, raw_spearman_p = safe_correlation(change_frame["geometry"], change_frame["raw"], "spearman")
        adjusted_spearman, adjusted_spearman_p = safe_correlation(
            change_frame["geometry"], change_frame["normalized"], "spearman"
        )
        rows.append(
            {
                "endpoint": endpoint,
                "geometry_proxy": geometry_column,
                "boundaries": int(change_frame.dropna().shape[0]),
                "raw_pearson_r": raw_pearson,
                "raw_pearson_p": raw_pearson_p,
                "normalized_pearson_r": adjusted_pearson,
                "normalized_pearson_p": adjusted_pearson_p,
                "raw_spearman_rho": raw_spearman,
                "raw_spearman_p": raw_spearman_p,
                "normalized_spearman_rho": adjusted_spearman,
                "normalized_spearman_p": adjusted_spearman_p,
            }
        )

    session = pd.read_csv(SESSION_METRICS_PATH, parse_dates=["session_start"]).sort_values("session_start")
    session = session.dropna(
        subset=[
            "mean_bird_bbox_area_fraction",
            "mean_bird_bbox_camera_normalized",
            "semantic_area_fraction",
        ]
    )
    bbox_geometry = session["semantic_area_fraction"].diff().iloc[1:]
    bbox_raw = session["mean_bird_bbox_area_fraction"].diff().iloc[1:]
    bbox_normalized = session["mean_bird_bbox_camera_normalized"].diff().iloc[1:]
    raw_pearson, raw_pearson_p = safe_correlation(bbox_geometry, bbox_raw, "pearson")
    adjusted_pearson, adjusted_pearson_p = safe_correlation(bbox_geometry, bbox_normalized, "pearson")
    raw_spearman, raw_spearman_p = safe_correlation(bbox_geometry, bbox_raw, "spearman")
    adjusted_spearman, adjusted_spearman_p = safe_correlation(bbox_geometry, bbox_normalized, "spearman")
    rows.append(
        {
            "endpoint": "mean_bird_bbox_area",
            "geometry_proxy": "semantic_area_fraction",
            "boundaries": len(bbox_geometry),
            "raw_pearson_r": raw_pearson,
            "raw_pearson_p": raw_pearson_p,
            "normalized_pearson_r": adjusted_pearson,
            "normalized_pearson_p": adjusted_pearson_p,
            "raw_spearman_rho": raw_spearman,
            "raw_spearman_p": raw_spearman_p,
            "normalized_spearman_rho": adjusted_spearman,
            "normalized_spearman_p": adjusted_spearman_p,
        }
    )
    output = pd.DataFrame(rows)
    for column in ["raw_pearson_p", "normalized_pearson_p", "raw_spearman_p", "normalized_spearman_p"]:
        output[f"{column}_bh"] = bh_adjust(output[column])
    return output


def strict_longitudinal_series(metric: str, quality: str) -> pd.DataFrame:
    frame = pl.read_parquet(NORMALIZED_PATH, columns=["start_dt", metric, quality]).to_pandas()
    frame["start_dt"] = pd.to_datetime(frame["start_dt"])
    frame = frame[
        frame["start_dt"].dt.hour.between(7, 16)
        & np.isfinite(frame[metric])
        & np.isfinite(frame[quality])
        & (frame[quality] > 0)
    ].copy()
    frame["date"] = frame["start_dt"].dt.floor("D")
    frame["hour"] = frame["start_dt"].dt.hour
    frame["weighted_value"] = frame[metric] * frame[quality]
    hourly = frame.groupby(["date", "hour"], as_index=False).agg(
        weighted_sum=("weighted_value", "sum"),
        effective_windows=(quality, "sum"),
    )
    hourly = hourly[hourly["effective_windows"] >= 2.0].copy()
    hourly["value"] = hourly["weighted_sum"] / hourly["effective_windows"]
    daily = hourly.groupby("date", as_index=False).agg(
        value=("value", "mean"),
        hour_bins=("hour", "nunique"),
    )
    daily = daily[daily["hour_bins"] >= 2].copy()
    daily["days_since_start"] = (daily["date"] - pd.Timestamp("2025-07-03")).dt.days
    return daily


def downstream_sensitivity() -> pd.DataFrame:
    rows = []
    for endpoint, raw_column, normalized_column, quality in [
        (
            "flock_spread_longitudinal",
            "flock_spread_rms",
            "flock_spread_camera_normalized",
            "quality__flock_spread_camera_normalized",
        ),
        (
            "video_activity_longitudinal",
            "activity_mean",
            "video_activity_camera_normalized",
            "quality__video_activity_camera_normalized",
        ),
    ]:
        raw_daily = strict_longitudinal_series(raw_column, quality)
        normalized_daily = strict_longitudinal_series(normalized_column, quality)
        raw_rho, raw_p = safe_correlation(
            raw_daily["days_since_start"], raw_daily["value"], "spearman"
        )
        normalized_rho, normalized_p = safe_correlation(
            normalized_daily["days_since_start"], normalized_daily["value"], "spearman"
        )
        rows.append(
            {
                "analysis": endpoint,
                "valid_days": len(raw_daily),
                "raw_result": raw_rho,
                "raw_p": raw_p,
                "normalized_result": normalized_rho,
                "normalized_p": normalized_p,
                "interpretation": "Spearman correlation with recording day",
            }
        )

    hmm = pd.read_csv(HMM_SENSITIVITY_PATH)
    raw_spatial = hmm[hmm["variant"] == "raw_spatial_features"].iloc[0]
    rows.append(
        {
            "analysis": "hmm_raw_spatial_sensitivity",
            "valid_days": np.nan,
            "raw_result": raw_spatial["adjusted_rand_index_vs_post_aug_reference"],
            "raw_p": np.nan,
            "normalized_result": raw_spatial["mapped_hard_state_agreement"],
            "normalized_p": np.nan,
            "interpretation": "raw_result=ARI; normalized_result=mapped hard-state agreement",
        }
    )
    return pd.DataFrame(rows)


def polygon_stability() -> tuple[pd.DataFrame, dict]:
    perturbation = pd.read_csv(PERTURBATION_PATH)
    rows = []
    for pixels, subset in perturbation.groupby("perturbation_pixels"):
        weights = subset["sampled_detections"].to_numpy(dtype=float)
        agreements = subset["assignment_agreement_with_original"].to_numpy(dtype=float)
        rows.append(
            {
                "perturbation_pixels": int(pixels),
                "sessions": int(subset["session_id"].nunique()),
                "sampled_detections": int(weights.sum()),
                "detection_weighted_assignment_agreement": float(np.average(agreements, weights=weights)),
                "session_median_assignment_agreement": float(np.median(agreements)),
                "session_q10_assignment_agreement": float(np.quantile(agreements, 0.10)),
            }
        )
    summary = pd.DataFrame(rows).sort_values("perturbation_pixels")

    assignment = pd.read_csv(ASSIGNMENT_PATH)
    assignment_weights = assignment["sampled_detections"].to_numpy(dtype=float)
    aggregate = {
        "sessions": int(assignment["session_id"].nunique()),
        "sampled_detections": int(assignment_weights.sum()),
        "detection_weighted_bottom_center_outside_fraction": float(
            np.average(assignment["bottom_center_outside_fraction"], weights=assignment_weights)
        ),
        "session_median_bottom_center_outside_fraction": float(
            assignment["bottom_center_outside_fraction"].median()
        ),
        "detection_weighted_center_bottom_agreement": float(
            np.average(assignment["center_bottom_agreement"], weights=assignment_weights)
        ),
    }
    return summary, aggregate


def standardized_pooled(detail: pd.DataFrame, value_column: str) -> tuple[np.ndarray, np.ndarray]:
    x_values = []
    y_values = []
    for _, subset in detail.groupby("zone", sort=False):
        subset = subset[["delta_geometry_area_share", value_column]].dropna()
        x = subset["delta_geometry_area_share"].to_numpy(dtype=float)
        y = subset[value_column].to_numpy(dtype=float)
        x_values.extend((x - x.mean()) / x.std(ddof=1))
        y_values.extend((y - y.mean()) / y.std(ddof=1))
    return np.asarray(x_values), np.asarray(y_values)


def make_figure(
    counterfactual: pd.DataFrame,
    boundary_detail: pd.DataFrame,
    polygon: pd.DataFrame,
    boundary_aggregate: dict,
) -> None:
    colors = {"raw": "#777777", "normalized": "#1F6F8B"}
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.1), gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})

    positions = np.arange(len(counterfactual))
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        counterfactual["raw_geometry_eta2"],
        width,
        color=colors["raw"],
        label="Raw occupancy",
    )
    axes[0].bar(
        positions + width / 2,
        counterfactual["normalized_geometry_eta2"],
        width,
        color=colors["normalized"],
        label="Log selectivity",
    )
    axes[0].set_xticks(positions, [ZONE_LABELS[zone] for zone in counterfactual["endpoint"]], rotation=28, ha="right")
    axes[0].set_ylabel(r"Geometry variance fraction ($\eta^2$)")
    axes[0].set_title("A  Fixed-state geometry replay")
    axes[0].legend(frameon=False, fontsize=8)

    for axis, value_column, title, color in [
        (axes[1], "delta_raw_conditional_fraction", "B  Raw boundary coupling", colors["raw"]),
        (axes[2], "delta_normalized_log_selectivity", "C  Normalized boundary coupling", colors["normalized"]),
    ]:
        x, y = standardized_pooled(boundary_detail, value_column)
        axis.scatter(x, y, s=18, alpha=0.58, color=color, edgecolors="none")
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        axis.plot(line_x, intercept + slope * line_x, color="#111111", linewidth=1.4)
        axis.axhline(0, color="#BBBBBB", linewidth=0.7)
        axis.axvline(0, color="#BBBBBB", linewidth=0.7)
        axis.set_xlabel(r"Change in zone area share ($z$)")
        axis.set_ylabel(r"Feature change ($z$)")
        axis.set_title(title)
    axes[1].text(
        0.03,
        0.96,
        f"mean within-zone r = {boundary_aggregate['mean_zone_raw_pearson_r']:.2f}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
    )
    axes[2].text(
        0.03,
        0.96,
        f"mean within-zone r = {boundary_aggregate['mean_zone_normalized_pearson_r']:.2f}",
        transform=axes[2].transAxes,
        va="top",
        fontsize=8,
    )

    axes[3].plot(
        polygon["perturbation_pixels"],
        polygon["detection_weighted_assignment_agreement"],
        color="#4C956C",
        marker="o",
        linewidth=1.8,
    )
    axes[3].axhline(0.95, color="#999999", linestyle="--", linewidth=0.9)
    axes[3].set_ylim(0.85, 1.005)
    axes[3].set_xlabel("Polygon dilation (+) / erosion (-), px")
    axes[3].set_ylabel("Assignment agreement")
    axes[3].set_title("D  Real-detection robustness")

    for axis in axes:
        axis.grid(alpha=0.18, linewidth=0.6)
    fig.suptitle("Session normalization tested against geometry-specific nuisance variation", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES / "fig_3b_session_normalization_geometric_validity.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "fig_3b_session_normalization_geometric_validity.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(
    counterfactual: pd.DataFrame,
    counterfactual_aggregate: dict,
    boundary: pd.DataFrame,
    boundary_aggregate: dict,
    scale: pd.DataFrame,
    camera_diagnostics: pd.DataFrame,
    downstream: pd.DataFrame,
    polygon: pd.DataFrame,
    polygon_aggregate: dict,
) -> None:
    raw_significant = int((boundary["raw_spearman_p_bh"] < 0.05).sum())
    normalized_significant = int((boundary["normalized_spearman_p_bh"] < 0.05).sum())
    five_pixel = polygon[polygon["perturbation_pixels"].abs() == 5]["detection_weighted_assignment_agreement"].min()
    ten_pixel = polygon[polygon["perturbation_pixels"].abs() == 10]["detection_weighted_assignment_agreement"].min()
    spread_trend = downstream[downstream["analysis"] == "flock_spread_longitudinal"].iloc[0]
    activity_trend = downstream[downstream["analysis"] == "video_activity_longitudinal"].iloc[0]
    hmm_sensitivity = downstream[downstream["analysis"] == "hmm_raw_spatial_sensitivity"].iloc[0]
    bbox_diagnostic = camera_diagnostics[camera_diagnostics["endpoint"] == "mean_bird_bbox_area"].iloc[0]
    lines = [
        "# Session-normalization geometric-validity analysis",
        "",
        "## Why this replaces boundary smoothness as the primary validation",
        "",
        "A raw before/after jump at a session boundary combines camera geometry with real elapsed-time, flock, sampling-hour, management, and layout changes. It therefore tests temporal smoothness, not whether the normalization operator removes the geometry nuisance it was designed to remove. This analysis instead fixes a biological spatial-density profile and replays it through all observed session geometries.",
        "",
        "## Primary result: fixed-state replay through actual geometries",
        "",
        f"The analysis used {counterfactual_aggregate['profiles']} quality-weighted, hour-balanced daily spatial profiles from {counterfactual_aggregate['source_sessions']} data-bearing sessions and replayed each profile through all {counterfactual_aggregate['target_geometries']} audited geometry objects. Across the four zones, the mean fraction of counterfactual variance attributable to geometry fell from {counterfactual_aggregate['mean_raw_geometry_eta2']:.3f} for raw conditional occupancy to {counterfactual_aggregate['mean_normalized_geometry_eta2']:.3f} for log selectivity, a {100 * counterfactual_aggregate['mean_geometry_eta2_reduction_fraction']:.1f}% reduction (profile-cluster bootstrap 95% CI {100 * counterfactual_aggregate['bootstrap_reduction_ci_low']:.1f}% to {100 * counterfactual_aggregate['bootstrap_reduction_ci_high']:.1f}%).",
        "",
        "This is the strongest current validation because the target nuisance is manipulated while the biological profile is held fixed. It supports image-plane area normalization under the stated density model; it is not a substitute for metric floor-plane calibration or an independent second polygon annotation.",
        "",
        "## Empirical corroboration: signed boundary coupling",
        "",
        f"Across {boundary_aggregate['complete_boundaries']} adjacent data-bearing session boundaries, the mean within-zone Pearson correlation between signed feature change and signed zone-area change was {boundary_aggregate['mean_zone_raw_pearson_r']:.3f} before adjustment and {boundary_aggregate['mean_zone_normalized_pearson_r']:.3f} after adjustment. The paired reduction was {boundary_aggregate['mean_zone_signed_correlation_reduction']:.3f} (boundary-cluster bootstrap 95% CI {boundary_aggregate['bootstrap_reduction_ci_low']:.3f} to {boundary_aggregate['bootstrap_reduction_ci_high']:.3f}). All {raw_significant}/4 raw Spearman associations and {normalized_significant}/4 normalized associations were significant after within-family BH correction.",
        "",
        "This boundary analysis asks whether feature changes track the geometry nuisance in the expected signed direction. It does not require biological trajectories to be smooth.",
        "",
        "## Operator and polygon checks",
        "",
        f"Under explicit uniform scale/coverage interventions, division by the recorded camera scale or semantic coverage recovered the held-fixed spread and activity values with maximum numerical errors of {scale['maximum_absolute_recovery_error'].max():.3g}. In the existing real-detection perturbation audit, detection-weighted assignment agreement remained at least {five_pixel:.3f} for +/-5 px and {ten_pixel:.3f} for +/-10 px polygon perturbations. The sampled bottom-center outside fraction was {polygon_aggregate['detection_weighted_bottom_center_outside_fraction']:.3f} across {polygon_aggregate['sampled_detections']:,} detections from {polygon_aggregate['sessions']} sessions.",
        "",
        "The scale/coverage result is an operator-equivariance check, not independent evidence that semantic coverage is a pure camera-scale proxy. The empirical boundary diagnostics are weaker for these endpoints. In particular, raw mean bird bounding-box area did not track semantic-coverage changes (Spearman rho "
        f"{bbox_diagnostic['raw_spearman_rho']:.3f}), whereas coverage division induced a strong negative association (rho {bbox_diagnostic['normalized_spearman_rho']:.3f}). This warns against presenting spread/activity scaling as calibrated physical normalization.",
        "",
        "## Does the scale-proxy limitation overturn downstream results?",
        "",
        f"Not by itself. The longitudinal correlation remained positive with similar magnitude for raw versus adjusted spread (rho {spread_trend['raw_result']:.3f} versus {spread_trend['normalized_result']:.3f}) and activity (rho {activity_trend['raw_result']:.3f} versus {activity_trend['normalized_result']:.3f}). The existing HMM raw-spatial sensitivity retained ARI {hmm_sensitivity['raw_result']:.3f} and mapped hard-state agreement {hmm_sensitivity['normalized_result']:.3f} relative to the normalized post-August reference. These checks support retaining the current downstream story with a sensitivity qualification, rather than rerunning or discarding the entire analysis solely because the coverage proxy is imperfect.",
        "",
        "## Recommended interpretation",
        "",
        "The old discontinuity table should be retained only as an endpoint-specific longitudinal sensitivity analysis. It should not be used as the deciding test of normalization validity. The fixed-state actual-geometry replay is the most direct primary test; signed boundary nuisance coupling and polygon perturbation are empirical corroboration. Together, these results strongly support zone log selectivity as the current session-normalized image-plane representation. Spread and activity division should be described more narrowly as coverage-adjusted sensitivity features, with raw-feature robustness reported, rather than as validated physical normalization.",
        "",
        "## Generated files",
        "",
        "- `analysis/tables/counterfactual_geometry_variance_decomposition.csv`",
        "- `analysis/tables/counterfactual_pseudocount_sensitivity.csv`",
        "- `analysis/tables/boundary_geometry_coupling_summary.csv`",
        "- `analysis/tables/boundary_geometry_coupling_detail.csv`",
        "- `analysis/tables/scale_equivariance.csv`",
        "- `analysis/tables/camera_scale_empirical_diagnostics.csv`",
        "- `analysis/tables/downstream_scale_normalization_sensitivity.csv`",
        "- `analysis/tables/polygon_perturbation_summary.csv`",
        "- `analysis/figures/fig_3b_session_normalization_geometric_validity.png` and `.pdf`",
        "- `analysis/audit/session_normalization_geometric_validity_manifest.json`",
    ]
    (ROOT / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for directory in [TABLES, FIGURES, AUDIT]:
        directory.mkdir(parents=True, exist_ok=True)

    daily = build_hour_balanced_daily()
    geometry = pd.read_csv(GEOMETRY_PATH, parse_dates=["session_start", "session_end"])
    if len(geometry) != 41 or not geometry["geometry_valid"].all():
        raise RuntimeError("Expected 41 valid session geometry objects")

    counterfactual, pseudocount, profiles, counterfactual_aggregate = counterfactual_analysis(daily, geometry)
    boundary_detail, boundary_summary, boundary_aggregate = boundary_geometry_coupling(daily, geometry)
    scale = scale_equivariance(daily, geometry)
    camera_diagnostics = camera_scale_empirical_diagnostics(daily, geometry)
    downstream = downstream_sensitivity()
    polygon, polygon_aggregate = polygon_stability()

    daily.to_csv(TABLES / "hour_balanced_daily_profiles.csv", index=False)
    profiles.to_csv(TABLES / "counterfactual_source_profiles.csv", index=False)
    counterfactual.to_csv(TABLES / "counterfactual_geometry_variance_decomposition.csv", index=False)
    pseudocount.to_csv(TABLES / "counterfactual_pseudocount_sensitivity.csv", index=False)
    boundary_detail.to_csv(TABLES / "boundary_geometry_coupling_detail.csv", index=False)
    boundary_summary.to_csv(TABLES / "boundary_geometry_coupling_summary.csv", index=False)
    scale.to_csv(TABLES / "scale_equivariance.csv", index=False)
    camera_diagnostics.to_csv(TABLES / "camera_scale_empirical_diagnostics.csv", index=False)
    downstream.to_csv(TABLES / "downstream_scale_normalization_sensitivity.csv", index=False)
    polygon.to_csv(TABLES / "polygon_perturbation_summary.csv", index=False)

    make_figure(counterfactual, boundary_detail, polygon, boundary_aggregate)
    write_summary(
        counterfactual,
        counterfactual_aggregate,
        boundary_summary,
        boundary_aggregate,
        scale,
        camera_diagnostics,
        downstream,
        polygon,
        polygon_aggregate,
    )

    manifest = {
        "analysis": "session_normalization_geometric_validity",
        "random_seed": RANDOM_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_pseudocount": PSEUDOCOUNT,
        "filters": {
            "daylight_schedule": True,
            "exclude_caretaker_or_management_overlap": True,
            "hourly_effective_quality_weight_minimum": 2.0,
            "daily_aggregation": "equal mean across available local hours",
        },
        "inputs": [
            {"path": str(path), "sha256": sha256(path)}
            for path in [
                NORMALIZED_PATH,
                GEOMETRY_PATH,
                PERTURBATION_PATH,
                ASSIGNMENT_PATH,
                SESSION_METRICS_PATH,
                HMM_SENSITIVITY_PATH,
            ]
        ],
        "primary_counterfactual": counterfactual_aggregate,
        "boundary_geometry_coupling": boundary_aggregate,
        "camera_scale_empirical_diagnostics": json.loads(camera_diagnostics.to_json(orient="records")),
        "downstream_scale_normalization_sensitivity": json.loads(downstream.to_json(orient="records")),
        "polygon_assignment": polygon_aggregate,
    }
    (AUDIT / "session_normalization_geometric_validity_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
