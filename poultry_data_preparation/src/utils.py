from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger(__name__)


def resolve_path(base_dir: Path, configured_path: str | Path | None) -> Path:
    path = Path(configured_path or "")
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown"


def combine_warnings(*values: object) -> str:
    combined: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            items = [str(item).strip() for item in value if str(item).strip()]
        else:
            text = str(value).strip()
            if not text or text.lower() == "nan":
                items = []
            else:
                items = [item.strip() for item in text.split(";") if item.strip()]
        for item in items:
            if item not in seen:
                seen.add(item)
                combined.append(item)
    return ";".join(combined)


def optional_text(value: object) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def safe_float(value: object, default: float | None = None) -> float | None:
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int | None = None) -> int | None:
    if value is None or pd.isna(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def write_table(
    dataframe: pd.DataFrame,
    csv_path: Path,
    write_csv: bool = True,
    write_parquet: bool = False,
) -> None:
    if write_csv:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(csv_path, index=False)
    if write_parquet:
        parquet_path = csv_path.with_suffix(".parquet")
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(parquet_path, index=False)


def read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def prepare_time_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    working_df = dataframe.copy()
    if "start_time" in working_df.columns:
        working_df["start_time_dt"] = pd.to_datetime(working_df["start_time"], errors="coerce", utc=False)
    else:
        working_df["start_time_dt"] = pd.NaT
    if "end_time" in working_df.columns:
        working_df["end_time_dt"] = pd.to_datetime(working_df["end_time"], errors="coerce", utc=False)
    else:
        working_df["end_time_dt"] = pd.NaT
    return working_df


def minmax_scale(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series([0.0] * len(series), index=series.index, dtype=float)
    minimum = float(valid.min())
    maximum = float(valid.max())
    if minimum == maximum:
        fill_value = 0.0 if minimum == 0.0 else 0.5
        return pd.Series([fill_value] * len(series), index=series.index, dtype=float)
    scaled = (numeric - minimum) / (maximum - minimum)
    return scaled.fillna(0.0).clip(0.0, 1.0)


def distribution_from_values(values) -> pd.Series:
    series = pd.Series(values, dtype=float).clip(lower=0.0)
    total = float(series.sum())
    if total <= 0:
        return pd.Series([1.0 / len(series)] * len(series), index=series.index, dtype=float)
    return series / total


def choose_output_video_path(
    absolute_path: Path,
    project_root: Path,
    path_style: str,
) -> str:
    if path_style == "absolute":
        return str(absolute_path.resolve())
    try:
        return absolute_path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return absolute_path.resolve().as_posix()


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def markdown_codeblock(text: str) -> str:
    return "```text\n" + text + "\n```"
