"""Rebuild the longitudinal evidence on the current 41-session normalized table."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
SOURCE = WORKSPACE / "experiments" / "room1_normalized_zone_quality_hmm_20260727" / "processed" / "normalized_multimodal_30s.parquet"
TABLES = ROOT / "analysis" / "tables"
FIGURES = ROOT / "analysis" / "figures"
RNG = np.random.default_rng(20260816)

METRICS = {
    "zone_feeding_log_selectivity": ("quality__zone_feeding_log_selectivity", "Feeding-zone selectivity", "#C58B22"),
    "zone_drinking_log_selectivity": ("quality__zone_drinking_log_selectivity", "Drinking-zone selectivity", "#2B7A9B"),
    "zone_resting_log_selectivity": ("quality__zone_resting_log_selectivity", "Resting-zone selectivity", "#6C8E4E"),
    "zone_open_movement_log_selectivity": ("quality__zone_open_movement_log_selectivity", "Open-movement selectivity", "#8C644B"),
    "feeding_activity_fraction": ("quality_video", "Feeding-zone pixel activity share", "#C58B22"),
    "drinking_activity_fraction": ("quality_video", "Drinking-zone pixel activity share", "#2B7A9B"),
    "resting_activity_fraction": ("quality_video", "Resting-zone pixel activity share", "#6C8E4E"),
    "open_movement_activity_fraction": ("quality_video", "Open-movement pixel activity share", "#8C644B"),
    "flock_spread_camera_normalized": ("quality__flock_spread_camera_normalized", "Camera-normalized flock spread", "#28756D"),
    "position_grid_entropy": ("quality__position_grid_entropy", "Position-grid entropy", "#4B6FA8"),
    "zone_entropy": ("quality__zone_entropy", "Zone entropy", "#76567D"),
    "video_activity_camera_normalized": ("quality__video_activity_camera_normalized", "Camera-normalized video activity", "#C25B50"),
    "behavior_active_fraction": ("quality__behavior_active_fraction", "Active-behavior fraction", "#D47A2C"),
    "behavior_feeding_fraction": ("quality__behavior_feeding_fraction", "Feeding-behavior fraction", "#B48A2C"),
}


def bh_adjust(values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return output
    adjusted = valid.to_numpy() * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output.loc[valid.index] = np.minimum(adjusted, 1.0)
    return output


def aggregate_metric(data: pd.DataFrame, metric: str, quality: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = data[["start_dt", metric, quality]].copy()
    subset = subset[
        np.isfinite(subset[metric])
        & np.isfinite(subset[quality])
        & (subset[quality] > 0)
    ]
    subset["date"] = subset["start_dt"].dt.floor("D")
    subset["hour"] = subset["start_dt"].dt.hour
    subset["weighted_value"] = subset[metric] * subset[quality]
    hourly = subset.groupby(["date", "hour"], as_index=False).agg(
        weighted_sum=("weighted_value", "sum"),
        effective_windows=(quality, "sum"),
        raw_windows=(metric, "size"),
    )
    hourly = hourly[hourly["effective_windows"] >= 2.0].copy()
    hourly["hourly_mean"] = hourly["weighted_sum"] / hourly["effective_windows"]
    daily = hourly.groupby("date", as_index=False).agg(
        value=("hourly_mean", "mean"),
        hour_bins=("hour", "nunique"),
        effective_windows=("effective_windows", "sum"),
        raw_windows=("raw_windows", "sum"),
    )
    daily = daily[daily["hour_bins"] >= 2].copy()
    daily["metric"] = metric
    daily["days_since_first_recording"] = (daily["date"] - pd.Timestamp("2025-07-03")).dt.days
    daily["week_start"] = daily["date"] - pd.to_timedelta(daily["date"].dt.weekday, unit="D")

    weekly_rows = []
    for week, group in daily.groupby("week_start", sort=True):
        values = group["value"].to_numpy()
        if len(values) >= 2:
            samples = RNG.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
            ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
        else:
            ci_low = ci_high = np.nan
        weekly_rows.append(
            {
                "week_start": week,
                "metric": metric,
                "recorded_days": len(group),
                "hour_bins": int(group["hour_bins"].sum()),
                "effective_windows": group["effective_windows"].sum(),
                "weekly_mean": values.mean(),
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
            }
        )
    return daily, pd.DataFrame(weekly_rows)


def trend_tests(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, group in daily.groupby("metric"):
        group = group.sort_values("days_since_first_recording")
        rho, p_value = spearmanr(group["days_since_first_recording"], group["value"])
        edge = min(14, max(3, len(group) // 5))
        rows.append(
            {
                "metric": metric,
                "label": METRICS[metric][1],
                "valid_days": len(group),
                "spearman_rho": rho,
                "p_value": p_value,
                "edge_days": edge,
                "first_mean": group["value"].head(edge).mean(),
                "last_mean": group["value"].tail(edge).mean(),
                "last_minus_first": group["value"].tail(edge).mean() - group["value"].head(edge).mean(),
            }
        )
    output = pd.DataFrame(rows)
    output["q_value_bh"] = bh_adjust(output["p_value"])
    output["fdr_significant_0_05"] = output["q_value_bh"] <= 0.05
    return output.sort_values(["q_value_bh", "metric"])


def plot_metric(axis: plt.Axes, weekly: pd.DataFrame, trends: pd.DataFrame, metric: str) -> None:
    frame = weekly[weekly["metric"].eq(metric)].sort_values("week_start")
    row = trends.set_index("metric").loc[metric]
    _, label, color = METRICS[metric]
    axis.plot(frame["week_start"], frame["weekly_mean"], marker="o", markersize=3.2, linewidth=1.5, color=color, label=label)
    axis.fill_between(frame["week_start"], frame["bootstrap_95_ci_low"], frame["bootstrap_95_ci_high"], color=color, alpha=0.14, linewidth=0)
    axis.text(0.02, 0.96 - 0.10 * len(axis.lines), f"{label}: rho={row.spearman_rho:.2f}, q={row.q_value_bh:.2g}", transform=axis.transAxes, va="top", color=color, fontsize=8.5)


def make_figure(weekly: pd.DataFrame, trends: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.8), sharex=True)
    for metric in ["zone_feeding_log_selectivity", "zone_open_movement_log_selectivity"]:
        plot_metric(axes[0, 0], weekly, trends, metric)
    axes[0, 0].axhline(0, color="#777777", linewidth=0.8)
    axes[0, 0].set_title("Area-adjusted zone selectivity")
    axes[0, 0].set_ylabel("Log selectivity")

    for metric in ["position_grid_entropy", "zone_entropy"]:
        plot_metric(axes[0, 1], weekly, trends, metric)
    axes[0, 1].set_title("Spatial dispersion")
    axes[0, 1].set_ylabel("Normalized entropy")

    plot_metric(axes[1, 0], weekly, trends, "flock_spread_camera_normalized")
    axes[1, 0].set_title("Camera-adjusted flock spread")
    axes[1, 0].set_ylabel("Normalized image-plane spread")

    for metric in ["behavior_feeding_fraction", "behavior_active_fraction"]:
        plot_metric(axes[1, 1], weekly, trends, metric)
    axes[1, 1].set_title("Upstream behavior fractions (valid from Aug. 16)")
    axes[1, 1].set_ylabel("Fraction")

    for axis in axes.flat:
        axis.grid(alpha=0.18, linewidth=0.6)
        axis.xaxis.set_major_locator(mdates.MonthLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axis.tick_params(axis="x", rotation=25, labelsize=8)
    fig.suptitle("Longitudinal outputs of the session-normalized representation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_4_longitudinal_representation.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "fig_4_longitudinal_representation.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    columns = ["start_dt", *METRICS, *sorted({item[0] for item in METRICS.values()})]
    data = pl.read_parquet(SOURCE, columns=columns).to_pandas()
    data["start_dt"] = pd.to_datetime(data["start_dt"])
    data = data[data["start_dt"].dt.hour.between(7, 16)].copy()

    daily_parts = []
    weekly_parts = []
    for metric, (quality, _, _) in METRICS.items():
        daily, weekly = aggregate_metric(data, metric, quality)
        daily_parts.append(daily)
        weekly_parts.append(weekly)
    daily = pd.concat(daily_parts, ignore_index=True)
    weekly = pd.concat(weekly_parts, ignore_index=True)
    trends = trend_tests(daily)

    daily.to_csv(TABLES / "longitudinal_daily_hour_balanced_current.csv", index=False)
    weekly.to_csv(TABLES / "longitudinal_weekly_hour_balanced_current.csv", index=False)
    trends.to_csv(TABLES / "longitudinal_trend_statistics_current.csv", index=False)
    pd.DataFrame(
        [{"metric": metric, "quality": values[0], "label": values[1]} for metric, values in METRICS.items()]
    ).to_csv(TABLES / "longitudinal_metric_definitions_current.csv", index=False)
    make_figure(weekly, trends)

    summary = {
        "metrics": len(METRICS),
        "daylight_hours": "07:00-16:59 local time",
        "hour_rule": "quality-weighted mean with at least 2.0 effective windows",
        "day_rule": "equal-weight mean across at least two valid hourly bins",
        "week_rule": "equal-weight mean across recorded days with 2,000 day-bootstrap intervals",
        "trend_rule": "Spearman correlation with recording day; BH adjustment across 14 metrics",
        "selected_results": trends.loc[
            trends["metric"].isin([
                "zone_feeding_log_selectivity",
                "zone_open_movement_log_selectivity",
                "position_grid_entropy",
                "zone_entropy",
                "flock_spread_camera_normalized",
                "behavior_feeding_fraction",
                "behavior_active_fraction",
            ])
        ].to_dict("records"),
    }
    (ROOT / "analysis" / "longitudinal_current_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
