from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PreparationConfig
from .utils import combine_warnings, prepare_time_columns, safe_float, write_table


WINDOW_COLUMNS = [
    "window_id",
    "media_id",
    "room_id",
    "session_id",
    "start_time",
    "end_time",
    "duration_seconds",
    "video_path",
    "video_start_offset_sec",
    "has_audio",
    "quality_status",
    "warnings",
]


@dataclass(frozen=True)
class WindowIndexResult:
    window_df: pd.DataFrame
    output_path: Path
    report_path: Path


def index_windows(manifest_df: pd.DataFrame, config: PreparationConfig) -> WindowIndexResult:
    rows: list[dict] = []
    for _, media_row in manifest_df.iterrows():
        rows.extend(build_window_rows_for_media(media_row, config.window_seconds, config.stride_seconds))

    window_df = pd.DataFrame(rows, columns=WINDOW_COLUMNS)
    if not window_df.empty:
        window_df = prepare_time_columns(window_df)
        window_df = window_df.sort_values(["room_id", "session_id", "start_time_dt", "window_id"], kind="stable").reset_index(drop=True)
        window_df = _apply_window_limits(window_df, config)
        window_df = window_df.drop(columns=["start_time_dt", "end_time_dt"], errors="ignore")

    output_path = config.metadata_output_dir / "video_window_index.csv"
    report_path = config.reports_output_dir / "window_index_report.md"
    write_table(window_df, output_path, write_csv=config.write_csv, write_parquet=config.write_parquet)
    report_path.write_text(_build_window_report(window_df, config), encoding="utf-8")
    return WindowIndexResult(window_df=window_df, output_path=output_path, report_path=report_path)


def build_window_rows_for_media(media_row: pd.Series, window_seconds: int, stride_seconds: int) -> list[dict]:
    duration_seconds = safe_float(media_row.get("duration_seconds"), 0.0) or 0.0
    start_time = pd.to_datetime(media_row.get("start_time"), errors="coerce", utc=False)
    base_warning = media_row.get("warnings", "")
    quality_status = str(media_row.get("quality_status", "ok") or "ok")

    offsets: list[float] = []
    if duration_seconds <= 0:
        offsets = [0.0]
        base_warning = combine_warnings(base_warning, "missing_or_zero_duration")
        quality_status = "warning" if quality_status == "ok" else quality_status
    elif duration_seconds < window_seconds:
        offsets = [0.0]
        base_warning = combine_warnings(base_warning, "short_media_partial_window")
    else:
        offset = 0.0
        while offset + window_seconds <= duration_seconds + 1e-9:
            offsets.append(round(offset, 6))
            offset += stride_seconds
        if not offsets:
            offsets = [0.0]

    rows: list[dict] = []
    for window_index, offset in enumerate(offsets, start=1):
        effective_duration = min(window_seconds, max(duration_seconds - offset, 0.0)) if duration_seconds > 0 else float(window_seconds)
        if start_time is not pd.NaT and pd.notna(start_time):
            window_start = start_time + pd.to_timedelta(offset, unit="s")
            window_end = window_start + pd.to_timedelta(effective_duration, unit="s")
            start_text = window_start.isoformat()
            end_text = window_end.isoformat()
        else:
            start_text = ""
            end_text = ""

        rows.append(
            {
                "window_id": f"{media_row['media_id']}_w{window_index:04d}",
                "media_id": str(media_row.get("media_id", "")),
                "room_id": str(media_row.get("room_id", "")),
                "session_id": str(media_row.get("session_id", "")),
                "start_time": start_text,
                "end_time": end_text,
                "duration_seconds": float(effective_duration),
                "video_path": str(media_row.get("video_path", "")),
                "video_start_offset_sec": float(offset),
                "has_audio": bool(media_row.get("has_audio", False)),
                "quality_status": quality_status,
                "warnings": combine_warnings(base_warning),
            }
        )
    return rows


def _apply_window_limits(window_df: pd.DataFrame, config: PreparationConfig) -> pd.DataFrame:
    limited_df = window_df.copy()
    if config.max_windows_per_media is not None:
        limited_df = limited_df.groupby("media_id", sort=False, dropna=False).head(config.max_windows_per_media).reset_index(drop=True)
    if config.max_windows_per_room is not None:
        limited_df = limited_df.groupby("room_id", sort=False, dropna=False).head(config.max_windows_per_room).reset_index(drop=True)
    if config.max_windows is not None:
        limited_df = limited_df.head(config.max_windows).reset_index(drop=True)
    return limited_df


def _build_window_report(window_df: pd.DataFrame, config: PreparationConfig) -> str:
    if window_df.empty:
        return "# Window Index Report\n\nNo windows were generated.\n"

    per_room_counts = window_df.groupby("room_id", dropna=False).size().to_dict()
    per_media_counts = window_df.groupby("media_id", dropna=False).size().describe()
    lines = [
        "# Window Index Report",
        "",
        f"- Total windows: {len(window_df)}",
        f"- Window seconds: {config.window_seconds}",
        f"- Stride seconds: {config.stride_seconds}",
        "",
        "## Windows Per Room",
        "",
    ]
    for room_id, count in per_room_counts.items():
        room_df = window_df[window_df["room_id"] == room_id]
        lines.append(
            f"- `{room_id}`: {int(count)} windows "
            f"(start `{room_df['start_time'].min()}`, end `{room_df['start_time'].max()}`)"
        )
    lines.extend(
        [
            "",
            "## Windows Per Media Summary",
            "",
            "```text",
            per_media_counts.to_string(),
            "```",
            "",
            "## Expected Processing Load",
            "",
            f"- Media files represented: {window_df['media_id'].nunique()}",
            f"- Windows with audio enabled: {int(window_df['has_audio'].fillna(False).astype(bool).sum())}",
        ]
    )
    return "\n".join(lines) + "\n"
