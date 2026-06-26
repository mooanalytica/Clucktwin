from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PreparationConfig
from .room_parser import parse_room_identifier_from_path
from .utils import write_table


@dataclass(frozen=True)
class EnvironmentLoadResult:
    env_df: pd.DataFrame
    output_path: Path
    report_path: Path


EXPECTED_COLUMNS = {
    "Date": "date",
    "Temp_AM_Min": "temp_am_min",
    "Temp_AM_Max": "temp_am_max",
    "Temp_PM": "temp_pm",
    "RH_AM": "rh_am",
    "RH_PM": "rh_pm",
}


def load_environment_tables(config: PreparationConfig) -> EnvironmentLoadResult:
    rows: list[pd.DataFrame] = []
    for path in sorted(config.raw_root.glob(config.env_glob), key=lambda value: value.as_posix().lower()):
        room_result = parse_room_identifier_from_path(path.name)
        raw_df = pd.read_excel(path, engine="openpyxl")
        normalized_df = normalize_environment_dataframe(raw_df, room_id=room_result.room_id)
        normalized_df["source_file"] = path.name
        rows.append(normalized_df)

    env_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["room_id", "date"])
    output_path = config.features_output_dir / "env_daily.csv"
    report_path = config.reports_output_dir / "env_report.md"
    write_table(env_df, output_path, write_csv=config.write_csv, write_parquet=config.write_parquet)
    report_path.write_text(_build_env_report(env_df), encoding="utf-8")
    return EnvironmentLoadResult(env_df=env_df, output_path=output_path, report_path=report_path)


def normalize_environment_dataframe(raw_df: pd.DataFrame, room_id: str) -> pd.DataFrame:
    normalized_df = raw_df.replace(
        {
            "NA": np.nan,
            "N/A": np.nan,
            "na": np.nan,
            "n/a": np.nan,
            "": np.nan,
        }
    ).copy()
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in normalized_df.columns]
    if missing_columns:
        raise ValueError("Environment workbook is missing expected columns: " + ", ".join(missing_columns))

    normalized_df = normalized_df[list(EXPECTED_COLUMNS)].rename(columns=EXPECTED_COLUMNS)
    normalized_df["date"] = _coerce_excel_date(normalized_df["date"])
    for column in ("temp_am_min", "temp_am_max", "temp_pm", "rh_am", "rh_pm"):
        normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce")

    normalized_df["temp_am_mean"] = normalized_df[["temp_am_min", "temp_am_max"]].mean(axis=1, skipna=True)
    normalized_df["temp_daily_mean"] = normalized_df[["temp_am_mean", "temp_pm"]].mean(axis=1, skipna=True)
    normalized_df["rh_daily_mean"] = normalized_df[["rh_am", "rh_pm"]].mean(axis=1, skipna=True)
    normalized_df["temp_daily_range"] = normalized_df["temp_am_max"] - normalized_df["temp_am_min"]
    normalized_df.loc[
        normalized_df["temp_am_max"].isna() | normalized_df["temp_am_min"].isna(),
        "temp_daily_range",
    ] = np.nan
    normalized_df["env_quality_flag"] = normalized_df.apply(_environment_quality_flag, axis=1)
    normalized_df["room_id"] = room_id
    normalized_df = normalized_df.sort_values("date", kind="stable").reset_index(drop=True)
    normalized_df["date"] = normalized_df["date"].dt.date
    return normalized_df[
        [
            "room_id",
            "date",
            "temp_am_min",
            "temp_am_max",
            "temp_pm",
            "rh_am",
            "rh_pm",
            "temp_am_mean",
            "temp_daily_mean",
            "rh_daily_mean",
            "temp_daily_range",
            "env_quality_flag",
        ]
    ]


def _coerce_excel_date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    numeric_mask = pd.to_numeric(series, errors="coerce").notna()
    result = pd.to_datetime(series, errors="coerce")
    if numeric_mask.any():
        numeric_dates = pd.to_datetime(
            pd.to_numeric(series[numeric_mask], errors="coerce"),
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
        result.loc[numeric_mask] = numeric_dates
    return result


def _environment_quality_flag(row: pd.Series) -> str:
    if pd.isna(row.get("date")):
        return "date_parse_failed"
    if pd.isna(row.get("temp_daily_mean")) and pd.isna(row.get("rh_daily_mean")):
        return "missing_temp_and_rh"
    if pd.isna(row.get("temp_daily_mean")) or pd.isna(row.get("rh_daily_mean")):
        return "partial"
    return "ok"


def _build_env_report(env_df: pd.DataFrame) -> str:
    if env_df.empty:
        return "# Environment Report\n\nNo environment rows were found.\n"
    room_counts = env_df.groupby("room_id").size().to_dict()
    lines = [
        "# Environment Report",
        "",
        "Daily environment data are treated as contextual covariates, not minute-level causal signals.",
        "",
        "## Rows Per Room",
        "",
    ]
    for room_id, count in room_counts.items():
        room_df = env_df[env_df["room_id"] == room_id]
        lines.append(
            f"- `{room_id}`: {int(count)} rows "
            f"(start `{room_df['date'].min()}`, end `{room_df['date'].max()}`)"
        )
    lines.extend(
        [
            "",
            "## Missing Values",
            "",
        ]
    )
    for column, value in env_df.isna().sum().to_dict().items():
        lines.append(f"- `{column}`: {int(value)}")
    lines.extend(
        [
            "",
            "## Summary Statistics",
            "",
            "```text",
            env_df[
                [
                    "temp_am_min",
                    "temp_am_max",
                    "temp_pm",
                    "temp_am_mean",
                    "temp_daily_mean",
                    "temp_daily_range",
                    "rh_am",
                    "rh_pm",
                    "rh_daily_mean",
                ]
            ].describe().to_string(),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"
