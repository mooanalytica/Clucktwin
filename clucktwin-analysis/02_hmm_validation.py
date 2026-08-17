"""Rebuild HMM validation with consistent modalities and session-blocked folds."""

from __future__ import annotations

import importlib.util
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import linear_sum_assignment
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
SOURCE = WORKSPACE / "experiments" / "room1_normalized_zone_quality_hmm_20260727"
SOURCE_PARQUET = SOURCE / "processed" / "normalized_multimodal_30s.parquet"
SOURCE_HMM = SOURCE / "03_fit_quality_weighted_hmm.py"
OLD_K5_POSTERIOR = SOURCE / "processed" / "hmm_state_posteriors.parquet"

TABLES = ROOT / "analysis" / "tables"
FIGURES = ROOT / "analysis" / "figures"
MODELS = ROOT / "analysis" / "models"
PROCESSED = ROOT / "analysis" / "processed"

K_VALUES = list(range(2, 13))
CV_SEEDS = [20260816, 20260817, 20260818]
FULL_SEEDS = [20260816, 20260817, 20260818, 20260819, 20260820]
FOLDS = 3

CORE_FEATURES = [
    "zone_drinking_log_selectivity",
    "zone_feeding_log_selectivity",
    "zone_resting_log_selectivity",
    "zone_open_movement_log_selectivity",
    "zone_entropy",
    "position_grid_entropy",
    "flock_spread_camera_normalized",
    "video_activity_camera_normalized",
    "audio_rms_global_z",
    "audio_bird_event_occupancy",
    "audio_disturbance_event_occupancy",
]
BEHAVIOR_FEATURES = [
    "behavior_active_fraction",
    "behavior_feeding_fraction",
    "behavior_idle_fraction",
    "behavior_preening_fraction",
]
POST_FEATURES = CORE_FEATURES[:8] + BEHAVIOR_FEATURES + CORE_FEATURES[8:]
RAW_POST_FEATURES = [
    "zone_drinking_fraction",
    "zone_feeding_fraction",
    "zone_resting_fraction",
    "zone_open_movement_fraction",
    "zone_entropy",
    "position_grid_entropy",
    "flock_spread_rms",
    "activity_mean",
    *BEHAVIOR_FEATURES,
    "audio_rms_global_z",
    "audio_bird_event_occupancy",
    "audio_disturbance_event_occupancy",
]
MODALITIES = {
    "position-zone": POST_FEATURES[:7],
    "video": ["video_activity_camera_normalized"],
    "behavior": BEHAVIOR_FEATURES,
    "audio": POST_FEATURES[-3:],
}


def load_hmm_module():
    spec = importlib.util.spec_from_file_location("quality_hmm_second_draft", SOURCE_HMM)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load HMM implementation: {SOURCE_HMM}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MAX_ITERATIONS = 20
    return module


def quality_columns(features: list[str]) -> list[str]:
    raw_map = {
        "zone_drinking_fraction": "quality__zone_drinking_log_selectivity",
        "zone_feeding_fraction": "quality__zone_feeding_log_selectivity",
        "zone_resting_fraction": "quality__zone_resting_log_selectivity",
        "zone_open_movement_fraction": "quality__zone_open_movement_log_selectivity",
        "flock_spread_rms": "quality__flock_spread_camera_normalized",
        "activity_mean": "quality__video_activity_camera_normalized",
    }
    return [raw_map.get(feature, f"quality__{feature}") for feature in features]


def complete_frame(source: pl.DataFrame, features: list[str], post_august: bool) -> pl.DataFrame:
    conditions = [pl.col(column) > 0 for column in quality_columns(features)]
    condition = conditions[0]
    for item in conditions[1:]:
        condition = condition & item
    if post_august:
        condition = condition & (pl.col("start_dt") >= pl.lit(pd.Timestamp("2025-08-16")))
    return source.filter(condition).sort(["session_id", "media_key", "start_dt"])


def arrays(frame: pl.DataFrame, features: list[str], weighted: bool = True):
    values = frame.select(features).to_numpy().astype(np.float64)
    weights = frame.select(quality_columns(features)).to_numpy().astype(np.float64)
    weights[~np.isfinite(values)] = 0.0
    if not weighted:
        weights = (weights > 0).astype(np.float64)
    overall = weights.mean(axis=1)
    return values, weights, overall


def scale_with_training(values: np.ndarray, weights: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
    output = np.where(np.isfinite(values), (values - centers) / scales, 0.0)
    output[weights <= 0] = 0.0
    return output.astype(np.float64)


def score_model(module, frame, x, weights, overall, model) -> dict:
    starts, ends = module.sequence_boundaries(frame)
    emissions = module.emission_log_likelihood(x, weights, np.asarray(model["means"]), np.asarray(model["variances"]))
    gamma, _, _, log_likelihood = module.expectation_step(
        emissions,
        np.log(np.maximum(np.asarray(model["start_probability"]), 1e-12)),
        np.log(np.maximum(np.asarray(model["transition"]), 1e-12)),
        starts,
        ends,
        overall,
    )
    return {
        "gamma": gamma,
        "log_likelihood": float(log_likelihood),
        "log_likelihood_per_row": float(log_likelihood / len(frame)),
        "rows": len(frame),
        "sequences": len(starts),
    }


def contiguous_session_folds(frame: pl.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
    sessions = (
        frame.group_by("session_id")
        .agg(pl.col("start_dt").min().alias("start_dt"), pl.len().alias("rows"))
        .sort("start_dt")
        .to_pandas()
    )
    blocks = np.array_split(sessions["session_id"].to_numpy(), FOLDS)
    mapping = {str(session): fold for fold, block in enumerate(blocks) for session in block}
    sessions["fold"] = sessions["session_id"].map(mapping)
    return mapping, sessions


def align_profiles(reference: dict, candidate: dict) -> tuple[dict[int, int], float]:
    reference_means = np.asarray(reference["means"])
    candidate_means = np.asarray(candidate["means"])
    distance = np.zeros((len(candidate_means), len(reference_means)))
    for left in range(len(candidate_means)):
        for right in range(len(reference_means)):
            distance[left, right] = np.linalg.norm(candidate_means[left] - reference_means[right])
    candidate_states, reference_states = linear_sum_assignment(distance)
    mapping = dict(zip(candidate_states.tolist(), reference_states.tolist()))
    correlations = []
    for candidate_state, reference_state in mapping.items():
        correlations.append(np.corrcoef(candidate_means[candidate_state], reference_means[reference_state])[0, 1])
    return mapping, float(np.nanmedian(correlations))


def seed_stability(models: list[dict], hard_states: list[np.ndarray], model_name: str, fold: int, k: int) -> list[dict]:
    rows = []
    for left, right in combinations(range(len(models)), 2):
        _, profile_correlation = align_profiles(models[left], models[right])
        rows.append(
            {
                "model": model_name,
                "fold": fold,
                "n_states": k,
                "seed_a": int(models[left]["seed"]),
                "seed_b": int(models[right]["seed"]),
                "adjusted_rand_index": adjusted_rand_score(hard_states[left], hard_states[right]),
                "matched_profile_median_correlation": profile_correlation,
            }
        )
    return rows


def run_session_blocked_cv(module, frame: pl.DataFrame, features: list[str], model_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_path = TABLES / f"hmm_{model_name}_session_blocked_cv_all_seeds.csv"
    stability_path = TABLES / f"hmm_{model_name}_initialization_stability.csv"
    existing_scores = pd.read_csv(score_path) if score_path.exists() else pd.DataFrame()
    existing_stability = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()
    score_rows = existing_scores.to_dict("records")
    stability_rows = existing_stability.to_dict("records")
    fold_mapping, ledger = contiguous_session_folds(frame)
    ledger.to_csv(TABLES / f"hmm_{model_name}_session_fold_ledger.csv", index=False)
    for fold in range(FOLDS):
        test_sessions = [session for session, value in fold_mapping.items() if value == fold]
        train = frame.filter(~pl.col("session_id").is_in(test_sessions)).sort(["media_key", "start_dt"])
        test = frame.filter(pl.col("session_id").is_in(test_sessions)).sort(["media_key", "start_dt"])
        train_values, train_weights, train_overall = arrays(train, features)
        train_x, centers, scales = module.robust_scale(train_values, train_weights)
        test_values, test_weights, test_overall = arrays(test, features)
        test_x = scale_with_training(test_values, test_weights, centers, scales)
        starts, ends = module.sequence_boundaries(train)
        for k in K_VALUES:
            complete_scores = (
                not existing_scores.empty
                and len(existing_scores[(existing_scores["fold"] == fold) & (existing_scores["n_states"] == k)]) == len(CV_SEEDS)
            )
            complete_stability = (
                not existing_stability.empty
                and len(existing_stability[(existing_stability["fold"] == fold) & (existing_stability["n_states"] == k)]) == math.comb(len(CV_SEEDS), 2)
            )
            if complete_scores and complete_stability:
                print(f"Resume {model_name}: fold={fold + 1}, K={k}", flush=True)
                continue
            score_rows = [row for row in score_rows if not (int(row["fold"]) == fold and int(row["n_states"]) == k)]
            stability_rows = [row for row in stability_rows if not (int(row["fold"]) == fold and int(row["n_states"]) == k)]
            models = []
            hard_states = []
            for seed_offset, seed_base in enumerate(CV_SEEDS):
                seed = seed_base + fold * 100 + k * 10
                print(f"Fit {model_name}: fold={fold + 1}/{FOLDS}, K={k}, seed={seed}", flush=True)
                model = module.fit_model(train_x, train_weights, train_overall, starts, ends, k, seed)
                heldout = score_model(module, test, test_x, test_weights, test_overall, model)
                models.append(model)
                hard_states.append(np.asarray(model["gamma"]).argmax(axis=1))
                score_rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "n_states": k,
                        "seed": seed,
                        "train_sessions": train["session_id"].n_unique(),
                        "test_sessions": test["session_id"].n_unique(),
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "train_log_likelihood": model["log_likelihood"],
                        "train_bic": model["bic"],
                        "train_min_state_occupancy": model["min_state_occupancy"],
                        "train_converged": model["converged"],
                        "test_log_likelihood": heldout["log_likelihood"],
                        "test_log_likelihood_per_row": heldout["log_likelihood_per_row"],
                    }
                )
            stability_rows.extend(seed_stability(models, hard_states, model_name, fold, k))
            pd.DataFrame(score_rows).sort_values(["fold", "n_states", "seed"]).to_csv(score_path, index=False)
            pd.DataFrame(stability_rows).sort_values(["fold", "n_states", "seed_a", "seed_b"]).to_csv(stability_path, index=False)
    scores = pd.DataFrame(score_rows)
    stability = pd.DataFrame(stability_rows)
    best_seed = scores.loc[scores.groupby(["fold", "n_states"])["train_log_likelihood"].idxmax()].copy()
    summary = (
        best_seed.groupby("n_states", as_index=False)
        .agg(
            mean_heldout_ll_per_row=("test_log_likelihood_per_row", "mean"),
            sd_heldout_ll_per_row=("test_log_likelihood_per_row", "std"),
            min_occupancy_across_folds=("train_min_state_occupancy", "min"),
            converged_folds=("train_converged", "sum"),
        )
        .sort_values("n_states")
    )
    summary["se_heldout_ll_per_row"] = summary["sd_heldout_ll_per_row"] / math.sqrt(FOLDS)
    stability_summary = (
        stability.groupby("n_states", as_index=False)
        .agg(
            median_seed_ari=("adjusted_rand_index", "median"),
            minimum_seed_ari=("adjusted_rand_index", "min"),
            median_profile_correlation=("matched_profile_median_correlation", "median"),
        )
        .sort_values("n_states")
    )
    summary.to_csv(TABLES / f"hmm_{model_name}_session_blocked_cv_summary.csv", index=False)
    stability_summary.to_csv(TABLES / f"hmm_{model_name}_initialization_stability_summary.csv", index=False)
    return scores, summary, stability, stability_summary


def fit_full(module, frame: pl.DataFrame, features: list[str], model_name: str, weighted: bool = True, seeds: list[int] | None = None) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    seeds = seeds or FULL_SEEDS
    values, weights, overall = arrays(frame, features, weighted=weighted)
    x, centers, scales = module.robust_scale(values, weights)
    starts, ends = module.sequence_boundaries(frame)
    models = []
    rows = []
    for seed in seeds:
        print(f"Full K=5 {model_name}: seed={seed}", flush=True)
        model = module.fit_model(x, weights, overall, starts, ends, 5, seed)
        models.append(model)
        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "log_likelihood": model["log_likelihood"],
                "bic": model["bic"],
                "iterations": model["iterations"],
                "converged": model["converged"],
                "min_state_occupancy": model["min_state_occupancy"],
                "quality_weighted_median_max_posterior": model["quality_weighted_median_max_posterior"],
                "mean_normalized_posterior_entropy": model["mean_normalized_posterior_entropy"],
            }
        )
    fits = pd.DataFrame(rows)
    selected_index = int(fits["log_likelihood"].idxmax())
    fits["selected"] = fits.index == selected_index
    fits.to_csv(TABLES / f"hmm_{model_name}_full_k5_seeds.csv", index=False)
    reference = models[selected_index]
    stability_rows = seed_stability(
        models,
        [np.asarray(model["gamma"]).argmax(axis=1) for model in models],
        model_name,
        -1,
        5,
    )
    pd.DataFrame(stability_rows).to_csv(TABLES / f"hmm_{model_name}_full_k5_stability.csv", index=False)
    return reference, centers, scales, overall


def save_model_output(frame: pl.DataFrame, features: list[str], model_name: str, model: dict, centers: np.ndarray, scales: np.ndarray, overall: np.ndarray) -> pl.DataFrame:
    gamma = np.asarray(model["gamma"])
    hard = gamma.argmax(axis=1).astype(np.int16)
    entropy = -np.sum(gamma * np.log(np.maximum(gamma, 1e-12)), axis=1) / math.log(5)
    output = frame.with_columns(
        [
            pl.Series("hmm_state", hard),
            pl.Series("hmm_max_posterior", gamma.max(axis=1)),
            pl.Series("hmm_normalized_posterior_entropy", entropy),
            pl.Series("hmm_state_assignment_score", gamma.max(axis=1) * overall),
            *[pl.Series(f"state_{state}_posterior", gamma[:, state]) for state in range(5)],
        ]
    )
    output.write_parquet(PROCESSED / f"hmm_{model_name}_k5_posteriors.parquet")
    profile = pd.DataFrame(np.asarray(model["means"]).T, columns=[f"state_{state}" for state in range(5)])
    profile.insert(0, "feature", features)
    profile.to_csv(TABLES / f"hmm_{model_name}_k5_profile.csv", index=False)
    pd.DataFrame(model["transition"], index=[f"state_{i}" for i in range(5)], columns=[f"state_{i}" for i in range(5)]).to_csv(
        TABLES / f"hmm_{model_name}_k5_transition.csv", index_label="from_state"
    )
    np.savez_compressed(
        MODELS / f"hmm_{model_name}_k5.npz",
        means=model["means"],
        variances=model["variances"],
        transition=model["transition"],
        start_probability=model["start_probability"],
        centers=centers,
        scales=scales,
        occupancy=model["occupancy"],
    )
    metadata = {
        "model": model_name,
        "features": features,
        "rows": len(frame),
        "sessions": frame["session_id"].n_unique(),
        "seed": int(model["seed"]),
        "states": 5,
        "interpretation": "exploratory temporal resolution; not a validated biological taxonomy or unique optimum",
        "state_names": [f"State {letter}" for letter in "ABCDE"],
        "quality_weighting": "feature-specific diagonal-Gaussian emission weights; all retained rows have every selected modality available",
    }
    (MODELS / f"hmm_{model_name}_k5.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def map_assignments(reference_model: dict, candidate_model: dict) -> tuple[np.ndarray, float, float]:
    reference_hard = np.asarray(reference_model["gamma"]).argmax(axis=1)
    candidate_hard = np.asarray(candidate_model["gamma"]).argmax(axis=1)
    contingency = np.zeros((5, 5), dtype=np.int64)
    np.add.at(contingency, (candidate_hard, reference_hard), 1)
    candidate_states, reference_states = linear_sum_assignment(-contingency)
    mapping = dict(zip(candidate_states.tolist(), reference_states.tolist()))
    mapped = np.array([mapping[state] for state in candidate_hard], dtype=np.int16)
    return mapped, float(adjusted_rand_score(reference_hard, candidate_hard)), float(np.mean(mapped == reference_hard))


def sensitivity_models(module, frame: pl.DataFrame, reference_model: dict) -> pd.DataFrame:
    variants = []
    for modality, removed in MODALITIES.items():
        variants.append((f"leave_out_{modality}", [feature for feature in POST_FEATURES if feature not in removed], True))
    variants.extend(
        [
            ("unweighted", POST_FEATURES, False),
            ("raw_spatial_features", RAW_POST_FEATURES, True),
        ]
    )
    rows = []
    for name, features, weighted in variants:
        model, _, _, _ = fit_full(module, frame, features, f"sensitivity_{name}", weighted=weighted, seeds=CV_SEEDS)
        mapped, ari, agreement = map_assignments(reference_model, model)
        rows.append(
            {
                "variant": name,
                "features": len(features),
                "weighted": weighted,
                "adjusted_rand_index_vs_post_aug_reference": ari,
                "mapped_hard_state_agreement": agreement,
                "median_max_posterior": float(np.median(np.asarray(model["gamma"]).max(axis=1))),
                "mean_normalized_posterior_entropy": float(model["mean_normalized_posterior_entropy"]),
                "minimum_state_occupancy": float(model["min_state_occupancy"]),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(TABLES / "hmm_k5_modality_weighting_normalization_sensitivity.csv", index=False)
    return output


def modality_availability(frame: pl.DataFrame, model_label: str) -> pd.DataFrame:
    data = frame.select(
        [
            "start_dt", "session_id", "hmm_state", "quality_position", "quality_zone", "quality_video", "quality_behavior", "quality_audio",
        ]
    ).to_pandas()
    data["start_dt"] = pd.to_datetime(data["start_dt"])
    data["month"] = data["start_dt"].dt.to_period("M").astype(str)
    for modality in ["position", "zone", "video", "behavior", "audio"]:
        data[f"available_{modality}"] = data[f"quality_{modality}"] > 0
    grouped = (
        data.groupby(["month", "hmm_state"], as_index=False)
        .agg(
            windows=("hmm_state", "size"),
            **{f"availability_{modality}": (f"available_{modality}", "mean") for modality in ["position", "zone", "video", "behavior", "audio"]},
            mean_quality=("quality_audio", "mean"),
        )
    )
    grouped.insert(0, "model", model_label)
    return grouped


def predictability_diagnostic(
    frame: pl.DataFrame,
    model_label: str,
    included_modalities: list[str],
) -> pd.DataFrame:
    data = frame.select(
        [
            "media_key", "session_id", "start_dt", "hmm_state", "quality_position", "quality_zone", "quality_video", "quality_behavior", "quality_audio",
        ]
    ).to_pandas()
    data["start_dt"] = pd.to_datetime(data["start_dt"])
    if len(data) > 60_000:
        samples = []
        for _, group in data.groupby("hmm_state", sort=False):
            samples.append(group.sample(min(len(group), 12_000), random_state=20260816))
        data = pd.concat(samples, ignore_index=True)
    data["day_index"] = (data["start_dt"] - pd.Timestamp("2025-07-03")).dt.total_seconds() / 86400.0
    quality_features = [f"quality_{modality}" for modality in included_modalities]
    missing_features = []
    for column in quality_features:
        name = column.replace("quality_", "missing_")
        data[name] = (data[column] <= 0).astype(int)
        missing_features.append(name)
    y = data["hmm_state"].to_numpy()
    groups = data["media_key"].to_numpy()
    folds = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=folds)
    blocks = {
        "session": ["session_id"],
        "date": ["day_index"],
        "quality": quality_features,
        "missingness_mask": missing_features,
        "all_nuisance": ["session_id", "day_index", *quality_features, *missing_features],
    }
    rows = []
    for block, features in blocks.items():
        if data[features].nunique(dropna=False).sum() <= len(features):
            rows.append({"model": model_label, "predictor_block": block, "folds": folds, "balanced_accuracy": np.nan, "macro_f1": np.nan, "note": "predictors constant in this complete-modality cohort"})
            continue
        categorical = [column for column in features if column == "session_id"]
        numeric = [column for column in features if column not in categorical]
        transformer = ColumnTransformer(
            [
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("numeric", StandardScaler(), numeric),
            ],
            remainder="drop",
        )
        predictions = np.full(len(data), -1, dtype=int)
        for train_index, test_index in splitter.split(data, y, groups):
            pipeline = Pipeline(
                [
                    ("transform", transformer),
                    ("model", LogisticRegression(max_iter=300, class_weight="balanced", solver="lbfgs")),
                ]
            )
            pipeline.fit(data.iloc[train_index][features], y[train_index])
            predictions[test_index] = pipeline.predict(data.iloc[test_index][features])
        rows.append(
            {
                "model": model_label,
                "predictor_block": block,
                "folds": folds,
                "balanced_accuracy": balanced_accuracy_score(y, predictions),
                "macro_f1": f1_score(y, predictions, average="macro"),
                "note": "media-grouped diagnostic; the HMM itself is validated with held-out session blocks",
            }
        )
    return pd.DataFrame(rows)


def weekly_occupancy(output: pl.DataFrame, model_name: str) -> pd.DataFrame:
    data = output.select(["start_dt", *[f"state_{state}_posterior" for state in range(5)]]).to_pandas()
    data["start_dt"] = pd.to_datetime(data["start_dt"])
    data["week"] = data["start_dt"].dt.to_period("W-MON").dt.start_time
    weekly = data.groupby("week", as_index=False)[[f"state_{state}_posterior" for state in range(5)]].mean()
    weekly.insert(0, "model", model_name)
    return weekly


def make_figures(core_summary, post_summary, core_stability, post_stability, core_profile, post_profile, predictability, sensitivity, core_weekly, post_weekly):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    for summary, label, color in [(core_summary, "consistent core", "#2C7FB8"), (post_summary, "post-Aug 16 + behavior", "#C05A73")]:
        axes[0, 0].errorbar(summary["n_states"], summary["mean_heldout_ll_per_row"], yerr=summary["se_heldout_ll_per_row"], marker="o", capsize=3, label=label, color=color)
    axes[0, 0].axvline(5, color="#444444", linestyle="--", linewidth=1)
    axes[0, 0].set_xlabel("Number of states K")
    axes[0, 0].set_ylabel("Held-out log likelihood / row")
    axes[0, 0].set_title("Session-blocked model score")
    axes[0, 0].legend(fontsize=8)

    for stability, label, color in [(core_stability, "consistent core", "#2C7FB8"), (post_stability, "post-Aug 16 + behavior", "#C05A73")]:
        axes[0, 1].plot(stability["n_states"], stability["median_seed_ari"], marker="o", label=label, color=color)
    axes[0, 1].axvline(5, color="#444444", linestyle="--", linewidth=1)
    axes[0, 1].set_ylim(0, 1.03)
    axes[0, 1].set_xlabel("Number of states K")
    axes[0, 1].set_ylabel("Median seed-pair ARI")
    axes[0, 1].set_title("Initialization stability")
    axes[0, 1].legend(fontsize=8)

    prediction_pivot = predictability.pivot(index="predictor_block", columns="model", values="balanced_accuracy")
    prediction_pivot.plot(kind="bar", ax=axes[0, 2], color=["#999999", "#2C7FB8", "#C05A73"], width=0.75)
    axes[0, 2].axhline(0.2, color="#555555", linestyle=":", linewidth=1)
    axes[0, 2].set_ylabel("Grouped balanced accuracy")
    axes[0, 2].set_xlabel("")
    axes[0, 2].set_title("Can nuisance variables predict state?")
    axes[0, 2].tick_params(axis="x", rotation=35, labelsize=8)
    axes[0, 2].legend(fontsize=7)

    core_matrix = core_profile.set_index("feature").to_numpy()
    image = axes[1, 0].imshow(core_matrix, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    axes[1, 0].set_xticks(range(5), [f"A{i + 1}" for i in range(5)])
    axes[1, 0].set_yticks(range(len(core_profile)), [value.replace("_", " ") for value in core_profile["feature"]], fontsize=6.5)
    axes[1, 0].set_title("Consistent-core K=5 profiles")

    post_matrix = post_profile.set_index("feature").to_numpy()
    axes[1, 1].imshow(post_matrix, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    axes[1, 1].set_xticks(range(5), [f"B{i + 1}" for i in range(5)])
    axes[1, 1].set_yticks(range(len(post_profile)), [value.replace("_", " ") for value in post_profile["feature"]], fontsize=6.5)
    axes[1, 1].set_title("Post-August-16 K=5 profiles")
    fig.colorbar(image, ax=[axes[1, 0], axes[1, 1]], fraction=0.025, pad=0.02, label="Robust standardized mean")

    ordered = sensitivity.sort_values("adjusted_rand_index_vs_post_aug_reference")
    axes[1, 2].barh(ordered["variant"], ordered["adjusted_rand_index_vs_post_aug_reference"], color="#4C956C")
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_xlabel("ARI vs reference K=5")
    axes[1, 2].set_title("Modality/weighting/normalization sensitivity")
    axes[1, 2].tick_params(axis="y", labelsize=7)
    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.08, right=0.98, hspace=0.42, wspace=0.38)
    fig.savefig(FIGURES / "fig_4_hmm_validation.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / "fig_4_hmm_validation.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for axis, weekly, prefix, title in [
        (axes[0], core_weekly, "A", "Consistent-core state occupancy"),
        (axes[1], post_weekly, "B", "Post-August-16 state occupancy with behavior"),
    ]:
        for state in range(5):
            axis.plot(pd.to_datetime(weekly["week"]), weekly[f"state_{state}_posterior"], marker=".", linewidth=1.1, label=f"State {prefix}{state + 1}")
        axis.set_ylabel("Mean posterior occupancy")
        axis.set_title(title)
        axis.legend(ncol=5, fontsize=7)
        axis.grid(alpha=0.2)
    axes[1].set_xlabel("Week")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_5_hmm_temporal_structure.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / "fig_5_hmm_temporal_structure.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for directory in [TABLES, FIGURES, MODELS, PROCESSED]:
        directory.mkdir(parents=True, exist_ok=True)
    module = load_hmm_module()
    source = pl.read_parquet(SOURCE_PARQUET)
    core = complete_frame(source, CORE_FEATURES, post_august=False)
    post = complete_frame(source, POST_FEATURES, post_august=True)
    print(f"Complete core rows={len(core):,}; post-August complete rows={len(post):,}", flush=True)

    _, core_summary, _, core_stability = run_session_blocked_cv(module, core, CORE_FEATURES, "consistent_core")
    _, post_summary, _, post_stability = run_session_blocked_cv(module, post, POST_FEATURES, "post_august_behavior")

    core_model, core_centers, core_scales, core_overall = fit_full(module, core.sort(["media_key", "start_dt"]), CORE_FEATURES, "consistent_core")
    core_output = save_model_output(core.sort(["media_key", "start_dt"]), CORE_FEATURES, "consistent_core", core_model, core_centers, core_scales, core_overall)
    post_model, post_centers, post_scales, post_overall = fit_full(module, post.sort(["media_key", "start_dt"]), POST_FEATURES, "post_august_behavior")
    post_output = save_model_output(post.sort(["media_key", "start_dt"]), POST_FEATURES, "post_august_behavior", post_model, post_centers, post_scales, post_overall)
    sensitivity = sensitivity_models(module, post.sort(["media_key", "start_dt"]), post_model)

    old = pl.read_parquet(OLD_K5_POSTERIOR).sort(["media_key", "start_dt"])
    availability = pd.concat(
        [
            modality_availability(old, "legacy_missing-modality K=5"),
            modality_availability(core_output, "consistent-core K=5"),
            modality_availability(post_output, "post-August behavior K=5"),
        ],
        ignore_index=True,
    )
    availability.to_csv(TABLES / "hmm_modality_availability_by_month_state.csv", index=False)
    predictability = pd.concat(
        [
            predictability_diagnostic(old, "legacy K=5", ["position", "zone", "video", "behavior", "audio"]),
            predictability_diagnostic(core_output, "consistent-core K=5", ["position", "zone", "video", "audio"]),
            predictability_diagnostic(post_output, "post-August behavior K=5", ["position", "zone", "video", "behavior", "audio"]),
        ],
        ignore_index=True,
    )
    predictability.to_csv(TABLES / "hmm_state_nuisance_predictability.csv", index=False)
    core_weekly = weekly_occupancy(core_output, "consistent_core")
    post_weekly = weekly_occupancy(post_output, "post_august_behavior")
    pd.concat([core_weekly, post_weekly], ignore_index=True).to_csv(TABLES / "hmm_weekly_occupancy_rebuilt.csv", index=False)

    core_profile = pd.read_csv(TABLES / "hmm_consistent_core_k5_profile.csv")
    post_profile = pd.read_csv(TABLES / "hmm_post_august_behavior_k5_profile.csv")
    make_figures(core_summary, post_summary, core_stability, post_stability, core_profile, post_profile, predictability, sensitivity, core_weekly, post_weekly)

    summary = {
        "legacy_eligible_windows": int(old.height),
        "consistent_core_windows": int(core.height),
        "consistent_core_sessions": int(core["session_id"].n_unique()),
        "post_august_behavior_windows": int(post.height),
        "post_august_behavior_sessions": int(post["session_id"].n_unique()),
        "candidate_states": K_VALUES,
        "folds": FOLDS,
        "random_initializations_per_fold_k": len(CV_SEEDS),
        "validation_grouping": "chronologically contiguous, non-overlapping session blocks",
        "core_best_tested_k": int(core_summary.loc[core_summary["mean_heldout_ll_per_row"].idxmax(), "n_states"]),
        "post_best_tested_k": int(post_summary.loc[post_summary["mean_heldout_ll_per_row"].idxmax(), "n_states"]),
        "k5_status": "exploratory temporal resolution chosen for interpretability; not a unique optimum or biological taxonomy",
        "neutral_state_names": ["State A1-A5", "State B1-B5"],
        "core_uncertainty": {
            "quality_weighted_median_max_posterior": float(core_model["quality_weighted_median_max_posterior"]),
            "mean_normalized_posterior_entropy": float(core_model["mean_normalized_posterior_entropy"]),
            "minimum_state_occupancy": float(core_model["min_state_occupancy"]),
        },
        "post_uncertainty": {
            "quality_weighted_median_max_posterior": float(post_model["quality_weighted_median_max_posterior"]),
            "mean_normalized_posterior_entropy": float(post_model["mean_normalized_posterior_entropy"]),
            "minimum_state_occupancy": float(post_model["min_state_occupancy"]),
        },
        "sensitivity": sensitivity.to_dict("records"),
        "nuisance_predictability": predictability.to_dict("records"),
    }
    (ROOT / "analysis" / "hmm_validation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
