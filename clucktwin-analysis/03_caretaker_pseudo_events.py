"""Build matched pseudo-events and controlled caretaker-entry comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
NORMALIZED = WORKSPACE / "experiments" / "room1_normalized_zone_quality_hmm_20260727" / "processed" / "normalized_multimodal_30s.parquet"
VERIFIED = WORKSPACE / "experiments" / "room1_growth_caretaker_dynamics_verified_fall_20260727"
EVENTS_PATH = VERIFIED / "tables" / "caretaker_events_analysis.csv"
MEDIA_PATH = VERIFIED / "audit" / "media_inventory.csv"

TABLES = ROOT / "analysis" / "tables"
FIGURES = ROOT / "analysis" / "figures"

FEATURES = {
    "zone_drinking_log_selectivity": ("quality__zone_drinking_log_selectivity", "Drinking-zone selectivity"),
    "zone_feeding_log_selectivity": ("quality__zone_feeding_log_selectivity", "Feeding-zone selectivity"),
    "zone_resting_log_selectivity": ("quality__zone_resting_log_selectivity", "Resting-zone selectivity"),
    "zone_open_movement_log_selectivity": ("quality__zone_open_movement_log_selectivity", "Open-movement selectivity"),
    "flock_spread_camera_normalized": ("quality__flock_spread_camera_normalized", "Coverage-adjusted spread"),
    "video_activity_camera_normalized": ("quality__video_activity_camera_normalized", "Coverage-adjusted video activity"),
    "behavior_active_fraction": ("quality__behavior_active_fraction", "Active behavior fraction"),
    "behavior_feeding_fraction": ("quality__behavior_feeding_fraction", "Feeding behavior fraction"),
    "behavior_idle_fraction": ("quality__behavior_idle_fraction", "Idle behavior fraction"),
    "behavior_preening_fraction": ("quality__behavior_preening_fraction", "Preening behavior fraction"),
    "audio_rms_global_z": ("quality__audio_rms_global_z", "Audio RMS"),
    "audio_bird_event_occupancy": ("quality__audio_bird_event_occupancy", "Bird-event acoustic proxy"),
    "audio_nonbird_event_occupancy": ("quality_audio", "Non-bird acoustic proxy"),
    "audio_disturbance_event_occupancy": ("quality__audio_disturbance_event_occupancy", "Disturbance acoustic proxy"),
}
PHASES = {
    "baseline": ("start", -10, -2),
    "immediate": ("end", 0, 2),
    "recovery": ("end", 2, 10),
}
PRIMARY_ISOLATION = (-10, 15)
RNG = np.random.default_rng(20260816)


def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def bh_adjust(values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return output
    adjusted = valid.to_numpy() * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output.loc[valid.index] = np.minimum(adjusted, 1.0)
    return output


def bootstrap_mean(values: np.ndarray, iterations: int = 5000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    samples = RNG.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def session_cluster_bootstrap(frame: pd.DataFrame, value_column: str, iterations: int = 5000) -> tuple[float, float]:
    """Bootstrap sessions while retaining all event-day means within each draw."""
    frame = frame.dropna(subset=[value_column]).copy()
    sessions = frame["session_id"].drop_duplicates().to_numpy()
    if len(sessions) < 2:
        return np.nan, np.nan
    by_session = {
        session: frame.loc[frame["session_id"].eq(session), value_column].to_numpy()
        for session in sessions
    }
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled_sessions = RNG.choice(sessions, size=len(sessions), replace=True)
        estimates[index] = np.concatenate([by_session[session] for session in sampled_sessions]).mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = ["session_id", "media_key", "start_dt", "end_dt", "caretaker_or_management_overlap", *FEATURES]
    quality = sorted({item[0] for item in FEATURES.values()})
    columns.extend([column for column in quality if column not in columns])
    data = pd.read_parquet(NORMALIZED, columns=columns)
    data["start_dt"] = pd.to_datetime(data["start_dt"])
    data["end_dt"] = pd.to_datetime(data["end_dt"])
    events = pd.read_csv(EVENTS_PATH, low_memory=False)
    for column in ["event_start", "event_end", "mask_start", "mask_end"]:
        events[column] = pd.to_datetime(events[column], errors="coerce")
    for column in ["primary_event_eligible", "is_isolated", "video_mask_eligible"]:
        events[column] = parse_bool(events[column])
    media = pd.read_csv(MEDIA_PATH, low_memory=False)
    media_map = media.drop_duplicates("media_key").set_index("media_key")["session_id"]
    events["session_id"] = events["canonical_media_key"].map(media_map)
    return data, events, media


def operation_intervals(events: pd.DataFrame) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]]:
    intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]] = {}
    valid = events[events["video_mask_eligible"] & events["session_id"].notna() & events["mask_start"].notna() & events["mask_end"].notna()]
    for row in valid.itertuples(index=False):
        intervals.setdefault(row.session_id, []).append((row.mask_start, row.mask_end, row.event_id))
    return intervals


def overlaps_operation(intervals, start: pd.Timestamp, end: pd.Timestamp, ignore_event: str | None = None) -> bool:
    for op_start, op_end, event_id in intervals:
        if ignore_event is not None and event_id == ignore_event:
            continue
        if op_start < end and op_end > start:
            return True
    return False


def clock_distance_minutes(left: pd.Timestamp, right: pd.Timestamp) -> float:
    left_minute = left.hour * 60 + left.minute + left.second / 60
    right_minute = right.hour * 60 + right.minute + right.second / 60
    distance = abs(left_minute - right_minute)
    return min(distance, 1440 - distance)


def event_has_phase_coverage(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    checks = [
        (start - pd.Timedelta(minutes=10), start - pd.Timedelta(minutes=2), 4),
        (end, end + pd.Timedelta(minutes=2), 1),
        (end + pd.Timedelta(minutes=2), end + pd.Timedelta(minutes=10), 4),
    ]
    for left, right, minimum in checks:
        if ((frame["start_dt"] >= left) & (frame["start_dt"] < right)).sum() < minimum:
            return False
    return True


def build_matches(data: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    exact = events[
        events["event_category"].eq("Caretaker Entry Event")
        & events["primary_event_eligible"]
        & events["is_isolated"]
        & events["session_id"].notna()
    ].drop_duplicates("event_id").sort_values("event_start")
    intervals = operation_intervals(events)
    by_session = {key: group.sort_values("start_dt").copy() for key, group in data.groupby("session_id")}
    used: dict[str, list[pd.Timestamp]] = {}
    rows = []
    for event in exact.itertuples(index=False):
        session_data = by_session.get(event.session_id)
        if session_data is None or session_data.empty:
            continue
        duration = event.event_end - event.event_start
        candidates = session_data.loc[
            session_data["start_dt"].dt.minute.mod(5).eq(0)
            & session_data["start_dt"].dt.second.lt(30),
            "start_dt",
        ].drop_duplicates()
        ranked = []
        for candidate_start in candidates:
            candidate_end = candidate_start + duration
            clock_difference = clock_distance_minutes(event.event_start, candidate_start)
            if clock_difference > 120:
                continue
            if abs(candidate_start - event.event_start) < pd.Timedelta(minutes=30):
                continue
            exclusion_start = candidate_start - pd.Timedelta(minutes=10)
            exclusion_end = candidate_end + pd.Timedelta(minutes=15)
            if overlaps_operation(intervals.get(event.session_id, []), exclusion_start, exclusion_end):
                continue
            if any(abs(candidate_start - prior) < pd.Timedelta(minutes=20) for prior in used.get(event.session_id, [])):
                continue
            if not event_has_phase_coverage(session_data, candidate_start, candidate_end):
                continue
            same_date_penalty = 60 if candidate_start.date() == event.event_start.date() else 0
            day_distance = abs((candidate_start.normalize() - event.event_start.normalize()).days)
            score = clock_difference + same_date_penalty + 0.1 * day_distance
            ranked.append((score, candidate_start, candidate_end, clock_difference, same_date_penalty, day_distance))
        if not ranked:
            continue
        ranked.sort(key=lambda item: item[0])
        match_score, pseudo_start, pseudo_end, clock_difference, same_date_penalty, day_distance = ranked[0]
        used.setdefault(event.session_id, []).append(pseudo_start)
        rows.append(
            {
                "event_id": event.event_id,
                "session_id": event.session_id,
                "event_start": event.event_start,
                "event_end": event.event_end,
                "event_date": event.event_start.normalize(),
                "label_period": event.label_period,
                "pseudo_start": pseudo_start,
                "pseudo_end": pseudo_end,
                "pseudo_date": pseudo_start.normalize(),
                "clock_difference_minutes": clock_difference,
                "date_gap_days": day_distance,
                "same_date_control": same_date_penalty > 0,
                "matching_score": match_score,
                "event_duration_seconds": duration.total_seconds(),
            }
        )
    return pd.DataFrame(rows)


def phase_mean(frame: pd.DataFrame, feature: str, quality: str, left: pd.Timestamp, right: pd.Timestamp) -> tuple[float, float, int]:
    subset = frame[(frame["start_dt"] >= left) & (frame["start_dt"] < right)].copy()
    valid = subset[feature].notna() & np.isfinite(subset[feature]) & subset[quality].notna() & (subset[quality] > 0)
    expected = max(int(round((right - left).total_seconds() / 30)), 1)
    if valid.sum() < max(int(np.ceil(expected * 0.25)), 1):
        return np.nan, 0.0, int(valid.sum())
    weights = subset.loc[valid, quality].clip(lower=0)
    if weights.sum() < 1.0:
        return np.nan, float(weights.sum()), int(valid.sum())
    value = float(np.average(subset.loc[valid, feature], weights=weights))
    return value, float(weights.sum()), int(valid.sum())


def phase_bounds(start: pd.Timestamp, end: pd.Timestamp, phase: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    reference, begin, finish = PHASES[phase]
    anchor = start if reference == "start" else end
    return anchor + pd.Timedelta(minutes=begin), anchor + pd.Timedelta(minutes=finish)


def build_responses(data: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    by_session = {key: group.sort_values("start_dt") for key, group in data.groupby("session_id")}
    rows = []
    for match in matches.itertuples(index=False):
        frame = by_session[match.session_id]
        for feature, (quality, label) in FEATURES.items():
            values = {}
            for group_name, start, end in [
                ("true", match.event_start, match.event_end),
                ("pseudo", match.pseudo_start, match.pseudo_end),
            ]:
                for phase in PHASES:
                    left, right = phase_bounds(start, end, phase)
                    value, effective, windows = phase_mean(frame, feature, quality, left, right)
                    values[f"{group_name}_{phase}"] = value
                    values[f"{group_name}_{phase}_effective_weight"] = effective
                    values[f"{group_name}_{phase}_windows"] = windows
            true_immediate_change = values["true_immediate"] - values["true_baseline"]
            pseudo_immediate_change = values["pseudo_immediate"] - values["pseudo_baseline"]
            true_recovery_change = values["true_recovery"] - values["true_baseline"]
            pseudo_recovery_change = values["pseudo_recovery"] - values["pseudo_baseline"]
            rows.append(
                {
                    "event_id": match.event_id,
                    "session_id": match.session_id,
                    "event_date": match.event_date,
                    "feature": feature,
                    "feature_label": label,
                    **values,
                    "true_immediate_change": true_immediate_change,
                    "pseudo_immediate_change": pseudo_immediate_change,
                    "immediate_difference_in_differences": true_immediate_change - pseudo_immediate_change,
                    "true_recovery_change": true_recovery_change,
                    "pseudo_recovery_change": pseudo_recovery_change,
                    "recovery_difference_in_differences": true_recovery_change - pseudo_recovery_change,
                }
            )
    return pd.DataFrame(rows)


def response_statistics(responses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, group in responses.groupby("feature"):
        for phase, column in [
            ("immediate", "immediate_difference_in_differences"),
            ("recovery", "recovery_difference_in_differences"),
        ]:
            subset = group.dropna(subset=[column]).copy()
            daily = subset.groupby(["session_id", "event_date"], as_index=False)[column].mean()
            values = daily[column].to_numpy()
            if len(values) >= 5 and np.any(np.abs(values) > 0):
                statistic, p_value = wilcoxon(values, zero_method="wilcox", alternative="two-sided")
            else:
                statistic, p_value = np.nan, np.nan
            ci_low, ci_high = bootstrap_mean(values)
            cluster_low, cluster_high = session_cluster_bootstrap(daily, column)
            rows.append(
                {
                    "feature": feature,
                    "feature_label": FEATURES[feature][1],
                    "phase": phase,
                    "matched_events": subset["event_id"].nunique(),
                    "sessions": daily["session_id"].nunique(),
                    "event_days": len(daily),
                    "mean_difference_in_differences": np.mean(values) if len(values) else np.nan,
                    "median_difference_in_differences": np.median(values) if len(values) else np.nan,
                    "bootstrap_95_ci_low": ci_low,
                    "bootstrap_95_ci_high": ci_high,
                    "session_cluster_bootstrap_95_ci_low": cluster_low,
                    "session_cluster_bootstrap_95_ci_high": cluster_high,
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
    output = pd.DataFrame(rows)
    for phase in output["phase"].unique():
        mask = output["phase"].eq(phase)
        output.loc[mask, "q_value_bh"] = bh_adjust(output.loc[mask, "p_value"])
    return output


def build_trajectory(data: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    selected_features = [
        "zone_feeding_log_selectivity",
        "video_activity_camera_normalized",
        "behavior_active_fraction",
        "audio_rms_global_z",
        "audio_disturbance_event_occupancy",
    ]
    by_session = {key: group.sort_values("start_dt") for key, group in data.groupby("session_id")}
    rows = []
    for match in matches.itertuples(index=False):
        frame = by_session[match.session_id]
        for feature in selected_features:
            quality, label = FEATURES[feature]
            true_base_left = match.event_start - pd.Timedelta(minutes=10)
            true_base_right = match.event_start - pd.Timedelta(minutes=2)
            pseudo_base_left = match.pseudo_start - pd.Timedelta(minutes=10)
            pseudo_base_right = match.pseudo_start - pd.Timedelta(minutes=2)
            true_baseline, _, _ = phase_mean(frame, feature, quality, true_base_left, true_base_right)
            pseudo_baseline, _, _ = phase_mean(frame, feature, quality, pseudo_base_left, pseudo_base_right)
            for minute in range(-10, 10):
                true_left = match.event_end + pd.Timedelta(minutes=minute)
                pseudo_left = match.pseudo_end + pd.Timedelta(minutes=minute)
                true_value, _, _ = phase_mean(frame, feature, quality, true_left, true_left + pd.Timedelta(minutes=1))
                pseudo_value, _, _ = phase_mean(frame, feature, quality, pseudo_left, pseudo_left + pd.Timedelta(minutes=1))
                rows.append(
                    {
                        "event_id": match.event_id,
                        "session_id": match.session_id,
                        "event_date": match.event_date,
                        "feature": feature,
                        "feature_label": label,
                        "minute": minute + 0.5,
                        "difference_in_differences_from_baseline": (true_value - true_baseline) - (pseudo_value - pseudo_baseline),
                    }
                )
    pair_level = pd.DataFrame(rows)
    daily = pair_level.groupby(["session_id", "event_date", "feature", "feature_label", "minute"], as_index=False)["difference_in_differences_from_baseline"].mean()
    summary_rows = []
    for keys, group in daily.groupby(["feature", "feature_label", "minute"]):
        values = group["difference_in_differences_from_baseline"].dropna().to_numpy()
        ci_low, ci_high = bootstrap_mean(values, iterations=2000)
        summary_rows.append(
            {
                "feature": keys[0],
                "feature_label": keys[1],
                "minute": keys[2],
                "event_days": len(values),
                "mean_difference_in_differences": np.mean(values) if len(values) else np.nan,
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
            }
        )
    return pd.DataFrame(summary_rows)


def isolation_sensitivity(events: pd.DataFrame, matches: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    exact = events[
        events["event_category"].eq("Caretaker Entry Event")
        & events["primary_event_eligible"]
        & events["session_id"].notna()
    ].drop_duplicates("event_id")
    all_intervals = operation_intervals(events)
    settings = [(-10, 15), (-20, 20), (-30, 30)]
    features = ["zone_feeding_log_selectivity", "video_activity_camera_normalized", "audio_rms_global_z"]
    rows = []
    for before, after in settings:
        retained = []
        for event in exact.itertuples(index=False):
            start = event.event_start + pd.Timedelta(minutes=before)
            end = event.event_end + pd.Timedelta(minutes=after)
            if not overlaps_operation(all_intervals.get(event.session_id, []), start, end, ignore_event=event.event_id):
                retained.append(event.event_id)
        retained_matches = set(matches.loc[matches["event_id"].isin(retained), "event_id"])
        for feature in features:
            subset = responses[responses["event_id"].isin(retained_matches) & responses["feature"].eq(feature)].dropna(subset=["immediate_difference_in_differences"])
            daily = subset.groupby("event_date")["immediate_difference_in_differences"].mean()
            rows.append(
                {
                    "isolation_before_minutes": before,
                    "isolation_after_minutes": after,
                    "isolated_exact_events": len(retained),
                    "matched_events_with_endpoint": subset["event_id"].nunique(),
                    "event_days": len(daily),
                    "feature": feature,
                    "mean_immediate_difference_in_differences": daily.mean(),
                }
            )
    output = pd.DataFrame(rows)
    primary = output[(output["isolation_before_minutes"] == -10) & (output["isolation_after_minutes"] == 15)].set_index("feature")["mean_immediate_difference_in_differences"]
    output["same_direction_as_primary"] = output.apply(
        lambda row: np.sign(row["mean_immediate_difference_in_differences"]) == np.sign(primary.get(row["feature"], np.nan)), axis=1
    )
    return output


def make_figure(matches: pd.DataFrame, statistics: pd.DataFrame, trajectory: pd.DataFrame, isolation: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.5))
    axes[0, 0].hist(matches["clock_difference_minutes"], bins=np.arange(0, 126, 10), color="#2C7FB8", edgecolor="white")
    axes[0, 0].axvline(matches["clock_difference_minutes"].median(), color="#222222", linestyle="--", label=f"median={matches['clock_difference_minutes'].median():.1f} min")
    axes[0, 0].set_xlabel("True-pseudo clock-time difference (min)")
    axes[0, 0].set_ylabel("Matched entry events")
    axes[0, 0].set_title("Same-session time-of-day matching")
    axes[0, 0].legend(fontsize=8)

    colors = ["#C05A73", "#4C956C", "#756BB1", "#2C7FB8", "#D99A00"]
    for (feature, group), color in zip(trajectory.groupby("feature"), colors):
        axes[0, 1].plot(group["minute"], group["mean_difference_in_differences"], label=FEATURES[feature][1], color=color)
        axes[0, 1].fill_between(
            group["minute"],
            group["bootstrap_95_ci_low"],
            group["bootstrap_95_ci_high"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    axes[0, 1].axvline(0, color="#222222", linewidth=1)
    axes[0, 1].axhline(0, color="#777777", linewidth=0.8)
    axes[0, 1].set_xlim(-10, 10)
    axes[0, 1].set_xlabel("Minutes from caretaker exit")
    axes[0, 1].set_ylabel("True-pseudo baseline-adjusted change")
    axes[0, 1].set_title("Event-time trajectory")
    axes[0, 1].legend(fontsize=6.5)

    immediate = statistics[statistics["phase"].eq("immediate")].sort_values("mean_difference_in_differences")
    positions = np.arange(len(immediate))
    means = immediate["mean_difference_in_differences"].to_numpy()
    lower = means - immediate["bootstrap_95_ci_low"].to_numpy()
    upper = immediate["bootstrap_95_ci_high"].to_numpy() - means
    axes[1, 0].errorbar(means, positions, xerr=np.vstack([lower, upper]), fmt="o", color="#2C7FB8", capsize=3)
    axes[1, 0].axvline(0, color="#333333", linewidth=1)
    axes[1, 0].set_yticks(positions, immediate["feature_label"], fontsize=7)
    axes[1, 0].set_xlabel("Immediate difference-in-differences")
    axes[1, 0].set_title("Matched acute response")

    isolation_pivot = isolation.pivot(index="feature", columns=["isolation_before_minutes", "isolation_after_minutes"], values="mean_immediate_difference_in_differences")
    isolation_pivot.index = [FEATURES[feature][1] for feature in isolation_pivot.index]
    isolation_pivot.T.plot(kind="bar", ax=axes[1, 1], color=["#C05A73", "#4C956C", "#2C7FB8"])
    axes[1, 1].axhline(0, color="#555555", linewidth=0.8)
    axes[1, 1].set_xlabel("Isolation window (minutes)")
    axes[1, 1].set_ylabel("Mean immediate difference-in-differences")
    axes[1, 1].set_title("Isolation-window sensitivity")
    axes[1, 1].tick_params(axis="x", rotation=0, labelsize=7)
    axes[1, 1].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_6_caretaker_matched_analysis.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "fig_6_caretaker_matched_analysis.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data, events, _ = load_inputs()
    matches = build_matches(data, events)
    matches.to_csv(TABLES / "caretaker_matched_pseudo_events.csv", index=False)
    responses = build_responses(data, matches)
    responses.to_csv(TABLES / "caretaker_matched_event_responses.csv", index=False)
    statistics = response_statistics(responses)
    statistics.to_csv(TABLES / "caretaker_difference_in_differences_statistics.csv", index=False)
    statistics.loc[
        statistics["phase"].eq("immediate"),
        [
            "feature",
            "feature_label",
            "matched_events",
            "sessions",
            "event_days",
            "mean_difference_in_differences",
            "bootstrap_95_ci_low",
            "bootstrap_95_ci_high",
            "session_cluster_bootstrap_95_ci_low",
            "session_cluster_bootstrap_95_ci_high",
            "q_value_bh",
        ],
    ].to_csv(TABLES / "caretaker_immediate_publication.csv", index=False)
    trajectory = build_trajectory(data, matches)
    trajectory.to_csv(TABLES / "caretaker_event_time_trajectory.csv", index=False)
    isolation = isolation_sensitivity(events, matches, responses)
    isolation.to_csv(TABLES / "caretaker_isolation_window_sensitivity.csv", index=False)
    make_figure(matches, statistics, trajectory, isolation)

    exact = events[events["event_category"].eq("Caretaker Entry Event") & events["primary_event_eligible"]].drop_duplicates("event_id")
    isolated = exact[exact["is_isolated"]]
    significant = statistics[(statistics["phase"] == "immediate") & (statistics["q_value_bh"] <= 0.05)]
    date_gaps = matches["date_gap_days"]
    summary = {
        "exact_unique_caretaker_entries": int(len(exact)),
        "primary_isolated_entries": int(len(isolated)),
        "matched_true_pseudo_pairs": int(len(matches)),
        "matched_sessions": int(matches["session_id"].nunique()),
        "median_clock_difference_minutes": float(matches["clock_difference_minutes"].median()),
        "same_date_controls": int(matches["same_date_control"].sum()),
        "median_date_gap_days": float(date_gaps.median()),
        "maximum_date_gap_days": int(date_gaps.max()),
        "pseudo_event_rule": "same-session candidates within 120 clock minutes, at least 30 minutes from the true entry, no labelled operational overlap in the -10/+15 minute neighborhood, ranked by clock_difference + 60*same_date + 0.1*date_gap_days",
        "difference_in_differences_unit": "session-day mean",
        "trajectory_alignment": "one-minute bins from -10 to +10 minutes relative to caretaker exit; baseline remains -10 to -2 minutes relative to entry start",
        "cluster_sensitivity": "95% confidence intervals additionally resample sessions as clusters while retaining their session-day values",
        "significant_immediate_features_after_bh": significant[["feature", "mean_difference_in_differences", "q_value_bh"]].to_dict("records"),
        "caretaker_purpose": "not present in the manual label tables; feeding, watering, cleaning, and inspection could not be separated and were not inferred",
        "room_layout_interpretation": "the feeding and drinking zones are at the end of Room 1 opposite the entrance; their post-entry selectivity increase is interpreted as flock displacement away from the entering caretaker, not increased feeding or drinking",
        "route_annotation": "the entrance-side relationship is known from room layout, but caretaker paths and door coordinates were not digitized",
    }
    (ROOT / "analysis" / "caretaker_controlled_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
