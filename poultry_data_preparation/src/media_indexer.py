from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import PreparationConfig
from .room_parser import UNKNOWN_ROOM_ID, parse_room_identifier_from_path
from .utils import (
    choose_output_video_path,
    combine_warnings,
    optional_text,
    safe_float,
    safe_int,
    slugify,
    write_table,
)


LOGGER = logging.getLogger(__name__)


MEDIA_COLUMNS = [
    "media_id",
    "room_id",
    "room_display_name",
    "session_id",
    "file_name",
    "relative_path",
    "absolute_path",
    "video_path",
    "file_size_bytes",
    "timestamp_source",
    "start_time",
    "end_time",
    "duration_seconds",
    "width",
    "height",
    "frame_rate",
    "video_codec",
    "has_audio",
    "audio_codec",
    "audio_sample_rate",
    "audio_channels",
    "quality_status",
    "warnings",
]


@dataclass(frozen=True)
class MediaIndexResult:
    manifest_df: pd.DataFrame
    output_path: Path
    report_path: Path


def index_media(config: PreparationConfig, dry_run: bool = False) -> MediaIndexResult:
    video_files = _scan_video_files(config)
    rows: list[dict] = []
    exiftool_available = shutil.which(config.exiftool_path) is not None
    ffprobe_available = shutil.which(config.ffprobe_path) is not None

    for media_index, absolute_path in enumerate(video_files, start=1):
        LOGGER.info("Indexing media %s/%s: %s", media_index, len(video_files), absolute_path.name)
        rows.append(
            _build_media_row(
                config=config,
                media_id=f"media_{media_index:06d}",
                absolute_path=absolute_path,
                exiftool_available=exiftool_available,
                ffprobe_available=ffprobe_available,
            )
        )

    manifest_df = pd.DataFrame(rows, columns=MEDIA_COLUMNS)
    output_path = config.metadata_output_dir / "media_manifest.csv"
    report_path = config.reports_output_dir / "media_index_report.md"
    write_table(manifest_df, output_path, write_csv=config.write_csv, write_parquet=config.write_parquet)
    report_path.write_text(_build_media_report(manifest_df, ffprobe_available, exiftool_available, dry_run), encoding="utf-8")
    return MediaIndexResult(manifest_df=manifest_df, output_path=output_path, report_path=report_path)


def _scan_video_files(config: PreparationConfig) -> list[Path]:
    candidates: set[Path] = set()
    for pattern in config.video_globs:
        candidates.update(path.resolve() for path in config.raw_root.glob(pattern) if path.is_file())
    sorted_candidates = sorted(candidates, key=lambda path: path.as_posix().lower())
    filtered: list[Path] = []
    for absolute_path in sorted_candidates:
        relative_for_parsing = _relative_to_root(config.raw_root, absolute_path)
        room_result = parse_room_identifier_from_path(relative_for_parsing)
        if not _room_is_included(room_result.room_id, config):
            continue
        if room_result.room_id in set(config.rooms_exclude):
            continue
        filtered.append(absolute_path)
    if config.max_media is not None:
        filtered = filtered[: config.max_media]
    return filtered


def _room_is_included(room_id: str, config: PreparationConfig) -> bool:
    if config.rooms_include == "auto":
        return True
    if isinstance(config.rooms_include, str):
        return room_id == str(config.rooms_include)
    return room_id in {str(item) for item in config.rooms_include}


def _build_media_row(
    config: PreparationConfig,
    media_id: str,
    absolute_path: Path,
    exiftool_available: bool,
    ffprobe_available: bool,
) -> dict:
    relative_path = _relative_to_root(config.project_root, absolute_path)
    room_result = parse_room_identifier_from_path(relative_path)
    warnings: list[str] = []
    if room_result.warning:
        warnings.append(room_result.warning)

    ffprobe_metadata = _read_ffprobe_metadata(absolute_path, config.ffprobe_path) if ffprobe_available else {"warning": "ffprobe_missing"}
    if ffprobe_metadata.get("warning"):
        warnings.append(str(ffprobe_metadata["warning"]))

    timestamp_value, timestamp_source, timestamp_warning = _resolve_media_timestamp(
        absolute_path=absolute_path,
        exiftool_path=config.exiftool_path,
        exiftool_available=exiftool_available,
        ffprobe_metadata=ffprobe_metadata,
    )
    if timestamp_warning:
        warnings.append(timestamp_warning)

    duration_seconds = safe_float(ffprobe_metadata.get("duration_seconds"))
    end_time = None
    if timestamp_value is not None and duration_seconds is not None:
        end_time = timestamp_value + pd.to_timedelta(duration_seconds, unit="s")

    file_size_bytes = absolute_path.stat().st_size
    session_id = infer_session_id(relative_path, room_result.room_id)
    quality_status = "ok"
    if ffprobe_metadata.get("failed"):
        quality_status = "failed"
    elif warnings:
        quality_status = "warning"

    return {
        "media_id": media_id,
        "room_id": room_result.room_id,
        "room_display_name": room_result.display_name,
        "session_id": session_id,
        "file_name": absolute_path.name,
        "relative_path": relative_path.as_posix(),
        "absolute_path": str(absolute_path),
        "video_path": choose_output_video_path(absolute_path, config.project_root, config.paths_in_outputs),
        "file_size_bytes": file_size_bytes,
        "timestamp_source": timestamp_source,
        "start_time": timestamp_value.isoformat() if timestamp_value is not None else "",
        "end_time": end_time.isoformat() if end_time is not None else "",
        "duration_seconds": duration_seconds,
        "width": ffprobe_metadata.get("width"),
        "height": ffprobe_metadata.get("height"),
        "frame_rate": ffprobe_metadata.get("frame_rate"),
        "video_codec": ffprobe_metadata.get("video_codec"),
        "has_audio": bool(ffprobe_metadata.get("has_audio", False)),
        "audio_codec": ffprobe_metadata.get("audio_codec"),
        "audio_sample_rate": ffprobe_metadata.get("audio_sample_rate"),
        "audio_channels": ffprobe_metadata.get("audio_channels"),
        "quality_status": quality_status,
        "warnings": combine_warnings(warnings),
    }


def infer_session_id(relative_path: Path, room_id: str) -> str:
    parts = list(relative_path.parts)
    if "video" in parts:
        video_index = parts.index("video")
        room_index = video_index + 1
        session_parts = parts[room_index + 1 : -1] if len(parts) > room_index + 2 else parts[room_index:-1]
    else:
        session_parts = list(relative_path.parent.parts)
    session_text = "_".join(session_parts) if session_parts else relative_path.parent.name
    session_slug = slugify(session_text)
    if room_id != UNKNOWN_ROOM_ID and not session_slug.startswith(room_id):
        return f"{room_id}_{session_slug}"
    return session_slug


def _read_ffprobe_metadata(absolute_path: Path, ffprobe_path: str) -> dict[str, object]:
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        str(absolute_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {
            "failed": True,
            "warning": f"ffprobe_failed:{completed.stderr.strip()[:200]}",
        }
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams", [])
    format_info = payload.get("format", {})
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})

    frame_rate = _parse_fraction(video_stream.get("avg_frame_rate")) or _parse_fraction(video_stream.get("r_frame_rate"))
    return {
        "failed": False,
        "duration_seconds": safe_float(format_info.get("duration")),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "frame_rate": frame_rate,
        "video_codec": optional_text(video_stream.get("codec_name")),
        "has_audio": bool(audio_stream),
        "audio_codec": optional_text(audio_stream.get("codec_name")),
        "audio_sample_rate": safe_int(audio_stream.get("sample_rate")),
        "audio_channels": safe_int(audio_stream.get("channels")),
        "creation_time": _ffprobe_creation_time(format_info, video_stream, audio_stream),
    }


def _resolve_media_timestamp(
    absolute_path: Path,
    exiftool_path: str,
    exiftool_available: bool,
    ffprobe_metadata: dict[str, object],
) -> tuple[pd.Timestamp | None, str, str | None]:
    if exiftool_available:
        exif_timestamp = _read_exiftool_timestamp(absolute_path, exiftool_path)
        if exif_timestamp is not None:
            return exif_timestamp, "exiftool", None

    ffprobe_time = optional_text(ffprobe_metadata.get("creation_time"))
    if ffprobe_time:
        parsed = _parse_timestamp_text(ffprobe_time)
        if parsed is not None:
            return parsed, "ffprobe_creation_time", None

    mtime = datetime.fromtimestamp(absolute_path.stat().st_mtime).astimezone()
    return pd.Timestamp(mtime), "file_mtime_fallback", "timestamp_source=file_mtime_fallback"


def _read_exiftool_timestamp(absolute_path: Path, exiftool_path: str) -> pd.Timestamp | None:
    command = [
        exiftool_path,
        "-j",
        "-api",
        "QuickTimeUTC=1",
        "-CreateDate",
        "-CreationDate",
        "-MediaCreateDate",
        "-TrackCreateDate",
        "-DateTimeOriginal",
        str(absolute_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None
    payload = json.loads(completed.stdout or "[]")
    if not payload:
        return None
    item = payload[0]
    for key in ("CreationDate", "CreateDate", "MediaCreateDate", "TrackCreateDate", "DateTimeOriginal"):
        parsed = _parse_timestamp_text(optional_text(item.get(key)))
        if parsed is not None:
            return parsed
    return None


def _parse_timestamp_text(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    if len(text) >= 10 and text[4] == ":" and text[7] == ":":
        text = text[:4] + "-" + text[5:7] + "-" + text[8:]
    try:
        parsed = pd.to_datetime(text, errors="coerce", utc=False)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _ffprobe_creation_time(format_info: dict, video_stream: dict, audio_stream: dict) -> str | None:
    for source in (format_info, video_stream, audio_stream):
        tags = source.get("tags", {}) if isinstance(source, dict) else {}
        for key in ("creation_time", "CreationTime"):
            value = optional_text(tags.get(key))
            if value:
                return value
    return None


def _parse_fraction(value: object) -> float | None:
    text = optional_text(value)
    if not text or text in {"0/0", "N/A"}:
        return None
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        numerator_value = safe_float(numerator)
        denominator_value = safe_float(denominator)
        if numerator_value is None or denominator_value in (None, 0):
            return None
        return float(numerator_value / denominator_value)
    return safe_float(text)


def _relative_to_root(root: Path, absolute_path: Path) -> Path:
    try:
        return absolute_path.resolve().relative_to(root.resolve())
    except ValueError:
        return absolute_path.resolve()


def _build_media_report(
    manifest_df: pd.DataFrame,
    ffprobe_available: bool,
    exiftool_available: bool,
    dry_run: bool,
) -> str:
    room_counts = manifest_df["room_id"].value_counts(dropna=False).to_dict() if not manifest_df.empty else {}
    codec_summary = manifest_df.groupby(["video_codec", "audio_codec"], dropna=False).size().reset_index(name="count") if not manifest_df.empty else pd.DataFrame()
    timestamp_fallback_count = int(manifest_df["timestamp_source"].eq("file_mtime_fallback").sum()) if not manifest_df.empty else 0
    missing_audio_count = int((~manifest_df["has_audio"].fillna(False).astype(bool)).sum()) if not manifest_df.empty else 0
    lines = [
        "# Media Index Report",
        "",
        f"- Dry run: `{dry_run}`",
        f"- ffprobe available: `{ffprobe_available}`",
        f"- exiftool available: `{exiftool_available}`",
        f"- Number of videos found: {len(manifest_df)}",
        f"- Files missing timestamps and using file mtime fallback: {timestamp_fallback_count}",
        f"- Files missing audio: {missing_audio_count}",
        "",
        "## Videos Per Room",
        "",
    ]
    for room_id, count in room_counts.items():
        room_df = manifest_df[manifest_df["room_id"] == room_id]
        lines.append(
            f"- `{room_id}`: {int(count)} videos "
            f"(start `{room_df['start_time'].min() if not room_df.empty else 'n/a'}`, "
            f"end `{room_df['start_time'].max() if not room_df.empty else 'n/a'}`)"
        )
    lines.extend(
        [
            "",
            "## Codec Summary",
            "",
            "```text",
            codec_summary.to_string(index=False) if not codec_summary.empty else "No media rows available.",
            "```",
            "",
            "## Warning Summary",
            "",
        ]
    )
    warning_rows = manifest_df[manifest_df["warnings"].astype(str).str.len() > 0] if not manifest_df.empty else pd.DataFrame()
    if warning_rows.empty:
        lines.append("- none")
    else:
        for _, row in warning_rows.head(25).iterrows():
            lines.append(f"- `{row['file_name']}`: `{row['warnings']}`")
        if len(warning_rows) > 25:
            lines.append(f"- plus {len(warning_rows) - 25} more warning rows")
    return "\n".join(lines) + "\n"
