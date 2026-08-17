"""Apply the canonical bottom-center/outside contract to the 50-window validation set."""

from __future__ import annotations

import json
import math
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from jsonschema import Draft202012Validator, FormatChecker
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
SOURCE_GEOMETRY = WORKSPACE / "experiments" / "room1_normalized_zone_quality_hmm_20260727" / "audit" / "session_geometry_normalization.csv"
NORMALIZED = WORKSPACE / "experiments" / "room1_normalized_zone_quality_hmm_20260727" / "processed" / "normalized_multimodal_30s.parquet"
MEDIA_PATH = WORKSPACE / "experiments" / "room1_growth_caretaker_dynamics_verified_fall_20260727" / "audit" / "media_inventory.csv"
MANUAL_PATH = WORKSPACE / "experiments" / "physical_virtual_fidelity_room1_20260708" / "outputs_daytime_50_outline_final" / "clip_annotations.csv"

TABLES = ROOT / "analysis" / "tables"
FIGURES = ROOT / "analysis" / "figures"
SCHEMA_DIR = ROOT / "analysis" / "schema"
JSON_DIR = ROOT / "analysis" / "json"

ZONES = ["drinking", "feeding", "resting", "open_movement"]
ZONE_IDS = {"drinking": "drinking", "feeding": "feeding", "resting": "resting", "open_movement": "open_movement", "outside": "outside"}
BEHAVIOR_ORDER = ["active", "drinking", "feeding", "idle", "perching", "preening", "wing_flapping"]
LOCAL_TIMEZONE = ZoneInfo("America/Halifax")


def local_iso(value: object) -> str:
    timestamp = pd.Timestamp(value).to_pydatetime()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=LOCAL_TIMEZONE)
    return timestamp.isoformat()


def finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def load_config(path: Path, session_id: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    configs = payload if isinstance(payload, list) else [payload]
    for config in configs:
        if str(config.get("session_id", "")) == session_id:
            return config
    raise RuntimeError(f"No zone config for {session_id}")


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


def assigned_mask(config: dict) -> np.ndarray:
    width = int(config["image_width"])
    height = int(config["image_height"])
    assigned = np.zeros((height, width), dtype=np.uint8)
    polygons = zone_polygons(config)
    for code, zone in enumerate(ZONES, start=1):
        mask = np.zeros((height, width), dtype=np.uint8)
        points = []
        for polygon in polygons[zone]:
            clipped = polygon.copy()
            clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
            clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
            points.append(clipped.reshape((-1, 1, 2)))
        if points:
            cv2.fillPoly(mask, points, color=1)
        assigned[(mask > 0) & (assigned == 0)] = code
    return assigned


def parse_range(value: str) -> tuple[float, float]:
    text = str(value).strip().replace("–", "~").replace("-", "~")
    values = [float(item) for item in text.split("~") if item.strip()]
    if len(values) == 1:
        return values[0], values[0]
    return min(values), max(values)


def interval_error(value: float, lower: float, upper: float) -> float:
    if lower <= value <= upper:
        return 0.0
    return lower - value if value < lower else value - upper


def classify(mask: np.ndarray, x: pd.Series, y: pd.Series) -> np.ndarray:
    xi = np.rint(x.to_numpy()).astype(int).clip(0, mask.shape[1] - 1)
    yi = np.rint(y.to_numpy()).astype(int).clip(0, mask.shape[0] - 1)
    return mask[yi, xi]


def load_track_window(media: pd.Series, clip_start: pd.Timestamp, clip_end: pd.Timestamp, config: dict) -> pd.DataFrame:
    media_start = pd.to_datetime(media["start_time"])
    offset_start = (clip_start - media_start).total_seconds()
    offset_end = (clip_end - media_start).total_seconds()
    columns = ["timestamp_sec", "timestamp_bin_sec", "track_id", "behavior", "confidence", "x1", "y1", "x2", "y2"]
    chunks = []
    for chunk in pd.read_csv(Path(str(media["track_path"])), usecols=columns, chunksize=250_000):
        timestamp = pd.to_numeric(chunk["timestamp_sec"], errors="coerce")
        keep = timestamp.ge(offset_start) & timestamp.lt(offset_end)
        if keep.any():
            chunks.append(chunk.loc[keep].copy())
    if not chunks:
        return pd.DataFrame(columns=columns)
    frame = pd.concat(chunks, ignore_index=True)
    frame["second"] = np.floor(pd.to_numeric(frame["timestamp_sec"], errors="coerce") - offset_start).clip(0, 29).astype(int)
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0)
    frame = frame.sort_values("confidence").drop_duplicates(["second", "track_id"], keep="last")
    scale_x = int(config["image_width"]) / float(media["behavior_source_width"])
    scale_y = int(config["image_height"]) / float(media["behavior_source_height"])
    frame["bottom_x"] = ((frame["x1"] + frame["x2"]) * 0.5 * scale_x).clip(0, int(config["image_width"]) - 1)
    frame["bottom_y"] = (frame["y2"] * scale_y).clip(0, int(config["image_height"]) - 1)
    codes = classify(assigned_mask(config), frame["bottom_x"], frame["bottom_y"])
    labels = np.array(["outside", *ZONES], dtype=object)
    frame["canonical_zone"] = labels[codes]
    return frame


def summarize_window(frame: pd.DataFrame) -> dict:
    seconds = 30
    if frame.empty:
        return {
            "visible_bird_count_mean": 0.0,
            **{f"{zone}_bird_count_mean": 0.0 for zone in [*ZONES, "outside"]},
            "dominant_behavior": "unavailable",
        }
    visible = frame.groupby("second")["track_id"].nunique().reindex(range(seconds), fill_value=0)
    summary = {"visible_bird_count_mean": float(visible.sum() / seconds)}
    for zone in [*ZONES, "outside"]:
        counts = frame[frame["canonical_zone"].eq(zone)].groupby("second")["track_id"].nunique().reindex(range(seconds), fill_value=0)
        summary[f"{zone}_bird_count_mean"] = float(counts.sum() / seconds)
    behavior = frame["behavior"].astype(str).str.lower().value_counts()
    summary["dominant_behavior"] = behavior.index[0] if len(behavior) else "unavailable"
    return summary


def canonical_schema() -> dict:
    zone_object = {
        "type": "object",
        "additionalProperties": False,
        "required": ["zone_id", "bird_count_mean"],
        "properties": {
            "zone_id": {"type": "string", "enum": ["drinking", "feeding", "resting", "open_movement", "outside"]},
            "bird_count_mean": {"type": ["number", "null"], "minimum": 0, "description": "Mean uniquely detected birds per second during the 30 s record."},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://clucktwin.local/schema/clucktwin-shadow-1.0.0.json",
        "title": "CluckTwin digital-shadow record",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "record_id", "room_id", "session_id", "start_time", "end_time", "update_cadence_seconds", "spatial_assignment", "flock", "zones", "provenance"],
        "properties": {
            "schema_version": {"const": "clucktwin.shadow/1.0.0"},
            "record_id": {"type": "string"},
            "room_id": {"type": "string"},
            "session_id": {"type": "string"},
            "start_time": {"type": "string", "format": "date-time"},
            "end_time": {"type": "string", "format": "date-time"},
            "update_cadence_seconds": {"type": "number", "const": 30},
            "spatial_assignment": {
                "type": "object",
                "additionalProperties": False,
                "required": ["point", "priority", "out_of_polygon"],
                "properties": {
                    "point": {"const": "bounding_box_bottom_center"},
                    "priority": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "prefixItems": [
                            {"const": "drinking"},
                            {"const": "feeding"},
                            {"const": "resting"},
                            {"const": "open_movement"},
                        ],
                        "items": False,
                    },
                    "out_of_polygon": {"const": "outside"},
                },
            },
            "flock": {
                "type": "object",
                "additionalProperties": False,
                "required": ["visible_bird_count_mean", "dominant_behavior"],
                "properties": {
                    "visible_bird_count_mean": {"type": ["number", "null"], "minimum": 0},
                    "dominant_behavior": {"type": ["string", "null"]},
                },
            },
            "zones": {"type": "array", "minItems": 5, "maxItems": 5, "items": zone_object},
            "provenance": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source_video", "source_window_id", "producer"],
                "properties": {
                    "source_video": {"type": "string"},
                    "source_window_id": {"type": "string"},
                    "producer": {"const": "Temporal Behavior Classification Model downstream canonicalizer"},
                },
            },
        },
    }


def validate_record(record: dict) -> list[str]:
    errors = []
    required = ["schema_version", "record_id", "room_id", "session_id", "start_time", "end_time", "update_cadence_seconds", "spatial_assignment", "flock", "zones", "provenance"]
    for key in required:
        if key not in record:
            errors.append(f"missing:{key}")
    if record.get("schema_version") != "clucktwin.shadow/1.0.0":
        errors.append("schema_version")
    for field in ["start_time", "end_time"]:
        try:
            parsed = pd.Timestamp(record.get(field))
            if parsed.tzinfo is None:
                errors.append(f"timezone_missing:{field}")
        except Exception:
            errors.append(f"invalid_timestamp:{field}")
    spatial = record.get("spatial_assignment", {})
    if spatial.get("point") != "bounding_box_bottom_center" or spatial.get("out_of_polygon") != "outside":
        errors.append("spatial_assignment")
    zones = record.get("zones", [])
    if [item.get("zone_id") for item in zones] != [*ZONES, "outside"]:
        errors.append("zone_order")
    if "welfare" in record or "state" in record:
        errors.append("deprecated_heuristic_field")
    return errors


def normalized_record(row: dict) -> dict:
    visible = finite_or_none(row["observed_birds_mean"])
    zone_counts = {}
    for zone in [*ZONES, "outside"]:
        fraction = finite_or_none(row[f"zone_{zone}_fraction"])
        zone_counts[zone] = visible * fraction if visible is not None and fraction is not None else None

    behaviors = {
        behavior: finite_or_none(row[f"behavior_{behavior}_fraction"])
        for behavior in BEHAVIOR_ORDER
    }
    available_behaviors = {key: value for key, value in behaviors.items() if value is not None}
    dominant = max(available_behaviors, key=available_behaviors.get) if available_behaviors else None
    return {
        "schema_version": "clucktwin.shadow/1.0.0",
        "record_id": f"{row['session_id']}_{row['window_id']}",
        "room_id": "room_1",
        "session_id": row["session_id"],
        "start_time": local_iso(row["start_dt"]),
        "end_time": local_iso(row["end_dt"]),
        "update_cadence_seconds": 30,
        "spatial_assignment": {
            "point": "bounding_box_bottom_center",
            "priority": ["drinking", "feeding", "resting", "open_movement"],
            "out_of_polygon": "outside",
        },
        "flock": {
            "visible_bird_count_mean": visible,
            "dominant_behavior": dominant,
        },
        "zones": [
            {"zone_id": zone, "bird_count_mean": zone_counts[zone]}
            for zone in [*ZONES, "outside"]
        ],
        "provenance": {
            "source_video": row["media_key"],
            "source_window_id": row["window_id"],
            "producer": "Temporal Behavior Classification Model downstream canonicalizer",
        },
    }


def full_table_json_audit(schema_validator: Draft202012Validator) -> pd.DataFrame:
    columns = [
        "window_id",
        "media_key",
        "session_id",
        "start_dt",
        "end_dt",
        "observed_birds_mean",
        *[f"zone_{zone}_fraction" for zone in [*ZONES, "outside"]],
        *[f"behavior_{behavior}_fraction" for behavior in BEHAVIOR_ORDER],
    ]
    frame = pl.read_parquet(NORMALIZED, columns=columns)
    schema_invalid_records = 0
    custom_contract_invalid_records = 0
    timestamp_mismatches = 0
    deprecated_fields = 0
    maximum_numeric_error = 0.0
    record_ids: set[str] = set()
    duplicate_ids = 0
    for row in frame.iter_rows(named=True):
        record = normalized_record(row)
        reopened = json.loads(json.dumps(record, allow_nan=False, separators=(",", ":")))
        if next(schema_validator.iter_errors(reopened), None) is not None:
            schema_invalid_records += 1
        if validate_record(reopened):
            custom_contract_invalid_records += 1
        if reopened["record_id"] in record_ids:
            duplicate_ids += 1
        record_ids.add(reopened["record_id"])
        if reopened["start_time"] != record["start_time"] or reopened["end_time"] != record["end_time"]:
            timestamp_mismatches += 1
        deprecated_fields += int("welfare" in reopened or "state" in reopened)
        numeric_pairs = [(reopened["flock"]["visible_bird_count_mean"], record["flock"]["visible_bird_count_mean"])]
        numeric_pairs.extend((left["bird_count_mean"], right["bird_count_mean"]) for left, right in zip(reopened["zones"], record["zones"]))
        for left, right in numeric_pairs:
            if left is not None and right is not None:
                maximum_numeric_error = max(maximum_numeric_error, abs(left - right))
    return pd.DataFrame(
        [
            {
                "records_expected": len(frame),
                "records_serialized_and_reopened": len(frame),
                "unique_record_ids": len(record_ids),
                "duplicate_record_ids": duplicate_ids,
                "json_schema_draft_2020_12_invalid_records": schema_invalid_records,
                "custom_contract_invalid_records": custom_contract_invalid_records,
                "schema_or_contract_invalid_records": schema_invalid_records + custom_contract_invalid_records,
                "jsonschema_package_version": version("jsonschema"),
                "format_checker_enabled": True,
                "timestamp_roundtrip_mismatches": timestamp_mismatches,
                "maximum_numeric_roundtrip_error": maximum_numeric_error,
                "deprecated_welfare_or_state_fields": deprecated_fields,
                "coverage_start": str(frame["start_dt"].min()),
                "coverage_end": str(frame["end_dt"].max()),
                "sessions": frame["session_id"].n_unique(),
            }
        ]
    )


def behavior_metrics(table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    labels = sorted(set(table["manual_primary_behavior"]) | set(table["analytical_dominant_behavior"]))
    precision, recall, f1, support = precision_recall_fscore_support(
        table["manual_primary_behavior"], table["analytical_dominant_behavior"], labels=labels, zero_division=0
    )
    class_table = pd.DataFrame({"behavior": labels, "precision": precision, "recall": recall, "f1": f1, "support": support})
    summary = {
        "balanced_recall": float(balanced_accuracy_score(table["manual_primary_behavior"], table["analytical_dominant_behavior"])),
        "macro_f1": float(f1_score(table["manual_primary_behavior"], table["analytical_dominant_behavior"], average="macro", zero_division=0)),
        "overall_accuracy": float((table["manual_primary_behavior"] == table["analytical_dominant_behavior"]).mean()),
        "labels": labels,
        "confusion_matrix": confusion_matrix(table["manual_primary_behavior"], table["analytical_dominant_behavior"], labels=labels).tolist(),
    }
    return class_table, summary


def count_metrics(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    endpoints = ["visible", "feeding", "drinking", "resting", "open_movement"]
    for endpoint in endpoints:
        lower = table[f"manual_{endpoint}_lower"]
        upper = table[f"manual_{endpoint}_upper"]
        analytical = table[f"analytical_{endpoint}_mean"]
        midpoint = (lower + upper) / 2
        error = analytical - midpoint
        interval = np.array([interval_error(value, low, high) for value, low, high in zip(analytical, lower, upper)])
        rows.append(
            {
                "endpoint": endpoint,
                "windows": len(table),
                "mean_bias_vs_manual_midpoint": error.mean(),
                "mae_vs_manual_midpoint": np.abs(error).mean(),
                "rmse_vs_manual_midpoint": float(np.sqrt(np.mean(error**2))),
                "agreement_lower": error.mean() - 1.96 * error.std(ddof=1),
                "agreement_upper": error.mean() + 1.96 * error.std(ddof=1),
                "median_interval_error": np.median(interval),
                "mean_interval_error": np.mean(interval),
                "within_manual_range_fraction": np.mean(interval == 0),
            }
        )
    return pd.DataFrame(rows)


def contract_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("schema_version", "string", "version identifier", False, "per record"),
            ("record_id", "string", "unique record identifier", False, "per record"),
            ("room_id/session_id", "string", "recording provenance", False, "per record"),
            ("start_time/end_time", "ISO-8601 string", "local timestamp with offset at deployment", False, "30 s record"),
            ("update_cadence_seconds", "number", "seconds", False, "30 s"),
            ("spatial_assignment.point", "enum", "bounding-box bottom-center", False, "schema constant"),
            ("spatial_assignment.out_of_polygon", "enum", "outside", False, "schema constant"),
            ("flock.visible_bird_count_mean", "number|null", "birds per second, averaged over record", True, "30 s"),
            ("flock.dominant_behavior", "string|null", "upstream behavior class", True, "30 s"),
            ("zones[].bird_count_mean", "number|null", "birds per second, averaged over record", True, "30 s"),
            ("provenance", "object", "source video/window and producer", False, "per record"),
        ],
        columns=["field", "type", "units_or_meaning", "nullable", "update_cadence"],
    )


def make_figure(counts: pd.DataFrame, behavior_classes: pd.DataFrame, table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    positions = np.arange(len(counts))
    axes[0].bar(positions, counts["mae_vs_manual_midpoint"], color="#2C7FB8")
    axes[0].set_xticks(positions, counts["endpoint"], rotation=35, ha="right")
    axes[0].set_ylabel("MAE (birds)")
    axes[0].set_title("Physical-to-analytical count error")

    axes[1].bar(positions, 100 * counts["within_manual_range_fraction"], color="#4C956C")
    axes[1].set_xticks(positions, counts["endpoint"], rotation=35, ha="right")
    axes[1].set_ylabel("Within manual range (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Range agreement")

    behavior_classes = behavior_classes.sort_values("support", ascending=True)
    axes[2].barh(behavior_classes["behavior"], 100 * behavior_classes["recall"], color="#C05A73")
    for position, row in enumerate(behavior_classes.itertuples(index=False)):
        axes[2].text(min(100 * row.recall + 2, 96), position, f"n={row.support}", va="center", fontsize=8)
    axes[2].set_xlim(0, 100)
    axes[2].set_xlabel("Recall (%)")
    axes[2].set_title("Behavior class recall and support")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_7_physical_analytical_json_validation.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "fig_7_physical_analytical_json_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for directory in [TABLES, FIGURES, SCHEMA_DIR, JSON_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    manual = pd.read_csv(MANUAL_PATH)
    manual["start_time"] = pd.to_datetime(manual["start_time"])
    manual["end_time"] = pd.to_datetime(manual["end_time"])
    geometry = pd.read_csv(SOURCE_GEOMETRY).set_index("session_id")
    media = pd.read_csv(MEDIA_PATH, low_memory=False)
    media = media[media["primary_media_eligible"].astype(str).str.lower() == "true"]
    media_lookup = media.set_index(["session_folder", "video_file"])

    rows = []
    records = []
    exclusions = []
    for clip in manual.itertuples(index=False):
        key = (clip.session, clip.video_name)
        if key not in media_lookup.index:
            exclusions.append({"clip_id": clip.clip_id, "reason": "media inventory match unavailable"})
            continue
        media_row = media_lookup.xs(key)
        if isinstance(media_row, pd.DataFrame):
            media_row = media_row.iloc[0]
        track_path = Path(str(media_row.get("track_path", "")))
        if str(media_row.get("track_exists", "")).lower() != "true" or not track_path.is_file():
            exclusions.append({"clip_id": clip.clip_id, "reason": "track-level CSV unavailable for canonical reprocessing"})
            continue
        session_id = media_row["session_id"]
        geometry_row = geometry.loc[session_id]
        config = load_config(Path(geometry_row["zone_configs_path"]), session_id)
        track = load_track_window(media_row, clip.start_time, clip.end_time, config)
        analytical = summarize_window(track)
        row = {
            "clip_id": clip.clip_id,
            "session_id": session_id,
            "session": clip.session,
            "video_name": clip.video_name,
            "window_id": clip.window_id,
            "start_time": clip.start_time,
            "end_time": clip.end_time,
            "manual_primary_behavior": str(clip.dominant_behavior).split("/")[0].strip().lower(),
            "analytical_dominant_behavior": analytical["dominant_behavior"],
            "analytical_visible_mean": analytical["visible_bird_count_mean"],
        }
        range_columns = {
            "visible": clip.visible_chicken_count,
            "feeding": clip.feeding_zone_count,
            "drinking": clip.drinking_zone_count,
            "resting": clip.resting_zone_count,
            "open_movement": clip.open_movement_zone_count,
        }
        for endpoint, value in range_columns.items():
            lower, upper = parse_range(value)
            row[f"manual_{endpoint}_lower"] = lower
            row[f"manual_{endpoint}_upper"] = upper
            if endpoint != "visible":
                row[f"analytical_{endpoint}_mean"] = analytical[f"{endpoint}_bird_count_mean"]
        row["analytical_outside_mean"] = analytical["outside_bird_count_mean"]
        rows.append(row)
        records.append(
            {
                "schema_version": "clucktwin.shadow/1.0.0",
                "record_id": f"{session_id}_{clip.window_id}",
                "room_id": "room_1",
                "session_id": session_id,
                "start_time": local_iso(clip.start_time),
                "end_time": local_iso(clip.end_time),
                "update_cadence_seconds": 30,
                "spatial_assignment": {
                    "point": "bounding_box_bottom_center",
                    "priority": ["drinking", "feeding", "resting", "open_movement"],
                    "out_of_polygon": "outside",
                },
                "flock": {
                    "visible_bird_count_mean": analytical["visible_bird_count_mean"],
                    "dominant_behavior": analytical["dominant_behavior"],
                },
                "zones": [
                    {"zone_id": zone, "bird_count_mean": analytical[f"{zone}_bird_count_mean"]}
                    for zone in [*ZONES, "outside"]
                ],
                "provenance": {
                    "source_video": clip.video_name,
                    "source_window_id": clip.window_id,
                    "producer": "Temporal Behavior Classification Model downstream canonicalizer",
                },
            }
        )

    validation = pd.DataFrame(rows)
    validation.to_csv(TABLES / "physical_to_analytical_validation_canonical.csv", index=False)
    pd.DataFrame(exclusions).to_csv(TABLES / "physical_to_analytical_canonical_exclusions.csv", index=False)
    counts = count_metrics(validation)
    counts.to_csv(TABLES / "physical_to_analytical_count_metrics.csv", index=False)
    behavior_classes, behavior_summary = behavior_metrics(validation)
    behavior_classes.to_csv(TABLES / "physical_to_analytical_behavior_metrics.csv", index=False)

    schema = canonical_schema()
    Draft202012Validator.check_schema(schema)
    schema_validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_path = SCHEMA_DIR / "clucktwin-shadow-1.0.0.schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    json_path = JSON_DIR / "clucktwin-shadow-1.0.0-validation-canonical.json"
    payload = {"schema_version": "clucktwin.shadow/1.0.0", "records": records}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    contract_table().to_csv(TABLES / "clucktwin_shadow_contract.csv", index=False)

    reopened = json.loads(json_path.read_text(encoding="utf-8"))
    schema_errors = {
        record["record_id"]: [error.message for error in schema_validator.iter_errors(record)]
        for record in reopened["records"]
    }
    schema_invalid = {key: value for key, value in schema_errors.items() if value}
    contract_errors = {record["record_id"]: validate_record(record) for record in reopened["records"]}
    contract_invalid = {key: value for key, value in contract_errors.items() if value}
    original = {record["record_id"]: record for record in records}
    roundtrip_errors = []
    missing = []
    for record in reopened["records"]:
        record_id = record["record_id"]
        if record_id not in original:
            missing.append(record_id)
            continue
        left = original[record_id]
        roundtrip_errors.append(abs(record["flock"]["visible_bird_count_mean"] - left["flock"]["visible_bird_count_mean"]))
        for candidate, source in zip(record["zones"], left["zones"]):
            roundtrip_errors.append(abs(candidate["bird_count_mean"] - source["bird_count_mean"]))
    analytical_json = pd.DataFrame(
        [
            {
                "records_expected": len(records),
                "records_serialized": len(reopened["records"]),
                "missing_records": len(set(original) - {record["record_id"] for record in reopened["records"]}),
                "extra_records": len(missing),
                "json_schema_draft_2020_12_invalid_records": len(schema_invalid),
                "custom_contract_invalid_records": len(contract_invalid),
                "schema_or_contract_invalid_records": len(set(schema_invalid) | set(contract_invalid)),
                "jsonschema_package_version": version("jsonschema"),
                "format_checker_enabled": True,
                "maximum_numeric_roundtrip_error": max(roundtrip_errors) if roundtrip_errors else np.nan,
                "timestamp_field_equality": all(original[item["record_id"]]["start_time"] == item["start_time"] and original[item["record_id"]]["end_time"] == item["end_time"] for item in reopened["records"]),
                "deprecated_welfare_or_state_fields": sum(("welfare" in item or "state" in item) for item in reopened["records"]),
            }
        ]
    )
    analytical_json.to_csv(TABLES / "analytical_to_json_fidelity.csv", index=False)
    full_json_audit = full_table_json_audit(schema_validator)
    full_json_audit.to_csv(TABLES / "analytical_to_json_fidelity_full_table.csv", index=False)
    make_figure(counts, behavior_classes, validation)

    summary = {
        "physical_to_analytical_windows": int(len(validation)),
        "manual_windows_available": int(len(manual)),
        "canonical_reprocessing_exclusions": exclusions,
        "sessions": int(validation["session_id"].nunique()),
        "date_start": str(validation["start_time"].min()),
        "date_end": str(validation["start_time"].max()),
        "canonical_spatial_operator": "bounding-box bottom-center with explicit outside category",
        "count_metrics": counts.to_dict("records"),
        "behavior": behavior_summary,
        "analytical_to_json": analytical_json.iloc[0].to_dict(),
        "analytical_to_json_full_table": full_json_audit.iloc[0].to_dict(),
        "json_to_unity": "not benchmarked for the versioned schema; the earlier prototype reader demonstrated replay but no latency, jitter, ordering, or drift metrics are claimed",
        "sample_limitation": "the existing manual set is restricted to August 16-September 1 and contains very little non-idle support; five of 50 windows lacked retained track-level CSV files for canonical reprocessing, and no additional manual annotation was available",
    }
    (ROOT / "analysis" / "contract_fidelity_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
