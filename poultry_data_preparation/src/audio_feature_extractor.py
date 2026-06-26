from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import PreparationConfig
from .utils import combine_warnings, prepare_time_columns, write_table


LOGGER = logging.getLogger(__name__)

AUDIO_CACHE_SCHEMA_VERSION = 2

REGIME_AXIS_COMPONENTS = {
    "audio_dense_regime_axis": [
        "audio_energy_entropy",
        "audio_band_energy_ratio_2_5khz",
        "audio_chirp_like_occupancy",
    ],
    "audio_sparse_regime_axis": [
        "audio_band_energy_ratio_300_1500hz",
        "audio_cluck_like_occupancy",
        "audio_cluck_like_mean_duration_sec",
    ],
}


AUDIO_COLUMNS = [
    "window_id",
    "media_id",
    "room_id",
    "session_id",
    "start_time",
    "end_time",
    "audio_available",
    "audio_duration_sec",
    "audio_sample_rate",
    "audio_analysis_frame_seconds",
    "audio_rms",
    "audio_short_time_energy",
    "audio_zero_crossing_rate",
    "audio_spectral_centroid",
    "audio_spectral_bandwidth",
    "audio_spectral_rolloff",
    "audio_spectral_flatness",
    "audio_energy_entropy",
    "audio_band_energy_ratio_2_5khz",
    "audio_band_energy_ratio_3_6khz",
    "audio_band_energy_ratio_300_1500hz",
    "audio_spectral_flux",
    "audio_frame_spectral_flatness_mean",
    "audio_dominant_frequency",
    "audio_call_like_event_rate",
    "audio_call_like_occupancy",
    "audio_chirp_like_event_rate",
    "audio_chirp_like_occupancy",
    "audio_chirp_like_mean_duration_sec",
    "audio_cluck_like_event_rate",
    "audio_cluck_like_occupancy",
    "audio_cluck_like_mean_duration_sec",
    "audio_bird_event_rate",
    "audio_bird_event_occupancy",
    "audio_bird_event_mean_duration_sec",
    "audio_dense_bird_event_rate",
    "audio_dense_bird_event_occupancy",
    "audio_dense_bird_event_mean_duration_sec",
    "audio_sparse_bird_event_rate",
    "audio_sparse_bird_event_occupancy",
    "audio_sparse_bird_event_mean_duration_sec",
    "audio_nonbird_event_rate",
    "audio_nonbird_event_occupancy",
    "audio_nonbird_event_mean_duration_sec",
    "audio_disturbance_event_rate",
    "audio_disturbance_event_occupancy",
    "audio_disturbance_event_mean_duration_sec",
    "audio_impact_like_event_rate",
    "audio_impact_like_occupancy",
    "audio_lowfreq_uncertain_event_rate",
    "audio_lowfreq_uncertain_occupancy",
    "audio_lowfreq_uncertain_mean_duration_sec",
    "audio_active_event_occupancy",
    "audio_bird_minus_nonbird_occupancy",
    "audio_bird_fraction_within_active",
    "audio_event_regime_hint",
    "audio_dense_regime_axis",
    "audio_sparse_regime_axis",
    "audio_regime_dense_score",
    "audio_regime_background_score",
    "audio_regime_sparse_score",
    "audio_regime_dense_score_smoothed_2min",
    "audio_regime_background_score_smoothed_2min",
    "audio_regime_sparse_score_smoothed_2min",
    "audio_regime_hard_label",
    "audio_regime_margin",
    "audio_quality_flag",
    "warnings",
]


@dataclass(frozen=True)
class AudioFeatureResult:
    audio_df: pd.DataFrame
    output_path: Path
    report_path: Path
    failed_df: pd.DataFrame


def extract_audio_features(
    config: PreparationConfig,
    media_df: pd.DataFrame,
    window_df: pd.DataFrame,
    dry_run: bool = False,
) -> AudioFeatureResult:
    output_path = config.features_output_dir / "audio_window_features.csv"
    report_path = config.reports_output_dir / "audio_feature_report.md"

    if dry_run or not config.audio_features_enabled:
        audio_df = pd.DataFrame(columns=AUDIO_COLUMNS)
        report_path.write_text(_build_dry_run_report(window_df, config.audio_features_enabled), encoding="utf-8")
        return AudioFeatureResult(audio_df, output_path, report_path, pd.DataFrame())

    if shutil.which(config.ffmpeg_path) is None:
        audio_df = pd.DataFrame(columns=AUDIO_COLUMNS)
        report_path.write_text("# Audio Feature Report\n\nffmpeg was not available, so audio extraction was skipped.\n", encoding="utf-8")
        return AudioFeatureResult(audio_df, output_path, report_path, pd.DataFrame())

    media_lookup = media_df.set_index("media_id", drop=False).to_dict(orient="index")
    all_rows: list[dict] = []
    failed_rows: list[dict] = []
    grouped = list(window_df.groupby("media_id", sort=False, dropna=False))
    processed_windows = 0

    for group_index, (media_id, media_windows) in enumerate(grouped, start=1):
        media_windows = media_windows.sort_values("video_start_offset_sec", kind="stable").reset_index(drop=True)
        LOGGER.info("Audio features media %s/%s: %s (%s windows)", group_index, len(grouped), media_id, len(media_windows))
        media_metadata = media_lookup.get(str(media_id))
        if media_metadata is None:
            for _, row in media_windows.iterrows():
                failed_rows.append(_failed_row(row, "missing_media_manifest_row"))
            continue

        if not bool(media_metadata.get("has_audio", False)):
            media_rows = _unavailable_audio_rows(
                media_windows,
                "manifest_has_audio_false",
                config.audio_sample_rate,
                config.audio_analysis_frame_seconds,
            )
        else:
            decode_duration_seconds = _media_decode_duration_seconds(media_windows)
            cache_data = _load_or_compute_audio_cache(
                config,
                Path(str(media_metadata["absolute_path"])),
                str(media_metadata["media_id"]),
                force_recompute=config.audio_force_recompute,
                decode_duration_seconds=decode_duration_seconds,
            )
            media_rows = _aggregate_audio_windows(
                media_windows,
                cache_data,
                config.audio_sample_rate,
                config.audio_analysis_frame_seconds,
            )
        all_rows.extend(media_rows.to_dict(orient="records"))
        failures = media_rows.loc[~media_rows["audio_available"].fillna(False).astype(bool), ["window_id", "media_id", "room_id"]]
        for _, failure_row in failures.iterrows():
            failed_rows.append(
                {
                    "stage": "audio_features",
                    "window_id": failure_row["window_id"],
                    "media_id": failure_row["media_id"],
                    "room_id": failure_row["room_id"],
                    "issue": "audio_unavailable",
                }
            )
        partial_df = _sort_audio_df(pd.DataFrame(all_rows, columns=AUDIO_COLUMNS))
        write_table(partial_df, output_path, write_csv=config.write_csv, write_parquet=False)
        processed_windows += len(media_windows)
        LOGGER.info("Processed %s/%s audio windows", processed_windows, len(window_df))

    audio_df = _sort_audio_df(pd.DataFrame(all_rows, columns=AUDIO_COLUMNS))
    audio_df = _finalize_audio_regime_features(audio_df, config.audio_regime_score_smoothing_windows)
    write_table(audio_df, output_path, write_csv=config.write_csv, write_parquet=config.write_parquet)
    failed_df = pd.DataFrame(failed_rows)
    report_path.write_text(_build_audio_report(window_df, audio_df, failed_df), encoding="utf-8")
    return AudioFeatureResult(audio_df, output_path, report_path, failed_df)


def _load_or_compute_audio_cache(
    config: PreparationConfig,
    absolute_path: Path,
    media_id: str,
    force_recompute: bool,
    decode_duration_seconds: float | None = None,
) -> dict[str, object]:
    cache_path = config.audio_feature_cache_dir / f"{media_id}.joblib"
    expected_metadata = _build_audio_cache_metadata(config, absolute_path, decode_duration_seconds=decode_duration_seconds)
    if cache_path.exists() and not force_recompute:
        cached = joblib.load(cache_path)
        if _audio_cache_metadata_matches(cached, expected_metadata):
            LOGGER.info("Reusing cached audio features for %s", media_id)
            return cached
        LOGGER.info("Discarding stale audio feature cache for %s", media_id)
    LOGGER.info("Computing audio features for %s", media_id)
    cache_data = _decode_media_audio_to_frames(
        absolute_path,
        config.ffmpeg_path,
        config.audio_sample_rate,
        config.audio_frame_seconds,
        analysis_frame_seconds=config.audio_analysis_frame_seconds,
        entropy_bands=config.audio_entropy_bands,
        decode_duration_seconds=decode_duration_seconds,
    )
    if config.audio_cache_per_media:
        cache_data["cache_metadata"] = expected_metadata
        joblib.dump(cache_data, cache_path)
    return cache_data


def _build_audio_cache_metadata(
    config: PreparationConfig,
    absolute_path: Path,
    decode_duration_seconds: float | None = None,
) -> dict[str, object]:
    resolved_path = absolute_path.resolve()
    file_stat = resolved_path.stat()
    return {
        "schema_version": AUDIO_CACHE_SCHEMA_VERSION,
        "audio_sample_rate": int(config.audio_sample_rate),
        "audio_frame_seconds": float(config.audio_frame_seconds),
        "audio_analysis_frame_seconds": float(config.audio_analysis_frame_seconds),
        "audio_entropy_bands": int(config.audio_entropy_bands),
        "decode_duration_seconds": _normalize_decode_duration(decode_duration_seconds),
        "source_media_path": str(resolved_path),
        "source_media_size": int(file_stat.st_size),
        "source_media_mtime_ns": int(file_stat.st_mtime_ns),
    }


def _audio_cache_metadata_matches(cache_data: dict[str, object], expected_metadata: dict[str, object]) -> bool:
    cached_metadata = cache_data.get("cache_metadata")
    if not isinstance(cached_metadata, dict):
        return False
    return all(cached_metadata.get(key) == value for key, value in expected_metadata.items())


def _decode_media_audio_to_frames(
    absolute_path: Path,
    ffmpeg_path: str,
    sample_rate: int,
    frame_seconds: float,
    analysis_frame_seconds: float = 0.1,
    entropy_bands: int = 10,
    decode_duration_seconds: float | None = None,
) -> dict[str, object]:
    command = [
        ffmpeg_path,
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(absolute_path),
        "-vn",
    ]
    normalized_duration = _normalize_decode_duration(decode_duration_seconds)
    if normalized_duration is not None:
        command.extend(["-t", f"{normalized_duration:.6f}"])
    command.extend(
        [
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "-",
        ]
    )
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        return {
            "frame_df": pd.DataFrame(),
            "extended_frame_df": pd.DataFrame(),
            "quality_flag": "ffmpeg_decode_failed",
            "warning": completed.stderr.decode("utf-8", errors="ignore")[:300],
        }
    waveform = np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if waveform.size == 0:
        return {
            "frame_df": pd.DataFrame(),
            "extended_frame_df": pd.DataFrame(),
            "quality_flag": "no_audio_samples",
            "warning": "No audio samples decoded.",
        }

    extended_frame_df = _compute_extended_audio_frame_features(
        waveform,
        sample_rate,
        analysis_frame_seconds,
        entropy_bands,
    )
    if not extended_frame_df.empty:
        extended_frame_df["frame_call_like"] = _build_call_like_mask(extended_frame_df).astype(bool)
        extended_frame_df["frame_chirp_like"] = _build_chirp_like_mask(extended_frame_df).astype(bool)
        extended_frame_df["frame_cluck_like"] = _build_cluck_like_mask(extended_frame_df).astype(bool)
    return {
        "frame_df": _compute_audio_frame_features(waveform, sample_rate, frame_seconds),
        "extended_frame_df": extended_frame_df,
        "quality_flag": "ok",
        "warning": "",
    }


def _media_decode_duration_seconds(media_windows: pd.DataFrame) -> float | None:
    if media_windows.empty:
        return None
    starts = pd.to_numeric(media_windows.get("video_start_offset_sec"), errors="coerce")
    durations = pd.to_numeric(media_windows.get("duration_seconds"), errors="coerce")
    max_end = (starts + durations).dropna().max()
    if pd.isna(max_end):
        return None
    return float(max(float(max_end), 0.0))


def _normalize_decode_duration(decode_duration_seconds: float | None) -> float | None:
    if decode_duration_seconds is None:
        return None
    value = float(decode_duration_seconds)
    if not np.isfinite(value) or value <= 0:
        return None
    return round(value, 6)


def _compute_audio_frame_features(waveform: np.ndarray, sample_rate: int, frame_seconds: float) -> pd.DataFrame:
    frame_size = max(1, int(round(sample_rate * frame_seconds)))
    frame_count = int(np.ceil(len(waveform) / frame_size))
    padded = np.pad(waveform, (0, max(0, frame_count * frame_size - len(waveform))), mode="constant")
    frames = padded.reshape(frame_count, frame_size)
    frame_starts = np.arange(frame_count, dtype=float) * frame_seconds
    frame_centers = frame_starts + (frame_seconds / 2.0)

    window = np.hanning(frame_size).astype(np.float32)
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    epsilon = 1e-12
    rms_values = np.sqrt(np.mean(frames * frames, axis=1))
    energy_values = np.mean(frames * frames, axis=1)
    zcr_values = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    magnitude = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
    magnitude_sum = magnitude.sum(axis=1) + epsilon
    centroid = (magnitude * frequencies).sum(axis=1) / magnitude_sum
    bandwidth = np.sqrt(((frequencies[None, :] - centroid[:, None]) ** 2 * magnitude).sum(axis=1) / magnitude_sum)
    cumulative_mag = np.cumsum(magnitude, axis=1)
    rolloff_threshold = 0.85 * magnitude_sum
    rolloff_indices = (cumulative_mag >= rolloff_threshold[:, None]).argmax(axis=1)
    flatness = np.exp(np.mean(np.log(magnitude + epsilon), axis=1)) / (np.mean(magnitude + epsilon, axis=1))
    return pd.DataFrame(
        {
            "frame_start_sec": frame_starts,
            "frame_center_sec": frame_centers,
            "audio_rms": rms_values,
            "audio_short_time_energy": energy_values,
            "audio_zero_crossing_rate": zcr_values,
            "audio_spectral_centroid": centroid,
            "audio_spectral_bandwidth": bandwidth,
            "audio_spectral_rolloff": frequencies[rolloff_indices],
            "audio_spectral_flatness": flatness,
        }
    )


def _compute_extended_audio_frame_features(
    waveform: np.ndarray,
    sample_rate: int,
    analysis_frame_seconds: float,
    entropy_bands: int,
) -> pd.DataFrame:
    frame_size = max(1, int(round(sample_rate * analysis_frame_seconds)))
    frame_count = int(np.ceil(len(waveform) / frame_size))
    padded = np.pad(waveform, (0, max(0, frame_count * frame_size - len(waveform))), mode="constant")
    frames = padded.reshape(frame_count, frame_size)
    frame_starts = np.arange(frame_count, dtype=float) * analysis_frame_seconds
    frame_centers = frame_starts + (analysis_frame_seconds / 2.0)

    window = np.hanning(frame_size).astype(np.float32)
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    epsilon = 1e-12
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    magnitude = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
    power = np.square(magnitude, dtype=np.float32)
    total_power = power.sum(axis=1) + epsilon
    magnitude_sum = magnitude.sum(axis=1) + epsilon

    entropy_band_indices = np.array_split(np.arange(power.shape[1]), max(2, int(entropy_bands)))
    band_energy = np.stack([power[:, index].sum(axis=1) for index in entropy_band_indices], axis=1)
    band_energy /= band_energy.sum(axis=1, keepdims=True) + epsilon
    energy_entropy = -np.sum(band_energy * np.log(band_energy + epsilon), axis=1) / np.log(band_energy.shape[1])

    normalized_magnitude = magnitude / magnitude_sum[:, None]
    spectral_flux = np.zeros(frame_count, dtype=np.float32)
    if frame_count > 1:
        spectral_flux[1:] = np.sqrt(np.square(normalized_magnitude[1:] - normalized_magnitude[:-1]).sum(axis=1))

    spectral_flatness = np.exp(np.mean(np.log(power + epsilon), axis=1)) / (np.mean(power, axis=1) + epsilon)
    dominant_frequency = frequencies[np.argmax(magnitude, axis=1)]
    return pd.DataFrame(
        {
            "frame_start_sec": frame_starts,
            "frame_center_sec": frame_centers,
            "frame_rms": rms,
            "frame_energy_entropy": energy_entropy,
            "frame_band_energy_ratio_2_5khz": _compute_band_energy_ratio(power, frequencies, total_power, 2000.0, 5000.0),
            "frame_band_energy_ratio_3_6khz": _compute_band_energy_ratio(power, frequencies, total_power, 3000.0, 6000.0),
            "frame_band_energy_ratio_300_1500hz": _compute_band_energy_ratio(power, frequencies, total_power, 300.0, 1500.0),
            "frame_spectral_flux": spectral_flux,
            "frame_spectral_flatness": spectral_flatness,
            "frame_dominant_frequency": dominant_frequency,
        }
    )


def _compute_band_energy_ratio(
    power: np.ndarray,
    frequencies: np.ndarray,
    total_power: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    band_mask = (frequencies >= float(low_hz)) & (frequencies <= float(high_hz))
    if not np.any(band_mask):
        return np.zeros(power.shape[0], dtype=np.float32)
    return power[:, band_mask].sum(axis=1) / total_power


def _aggregate_audio_windows(
    media_windows: pd.DataFrame,
    cache_data: dict[str, object],
    sample_rate: int,
    analysis_frame_seconds: float,
) -> pd.DataFrame:
    frame_df = cache_data.get("frame_df", pd.DataFrame())
    extended_frame_df = cache_data.get("extended_frame_df", pd.DataFrame())
    quality_flag = str(cache_data.get("quality_flag", "unknown"))
    warning = str(cache_data.get("warning", ""))
    if frame_df is None or not isinstance(frame_df, pd.DataFrame) or frame_df.empty:
        return _unavailable_audio_rows(media_windows, quality_flag, sample_rate, analysis_frame_seconds, warning)
    if extended_frame_df is None or not isinstance(extended_frame_df, pd.DataFrame):
        extended_frame_df = pd.DataFrame()

    rows: list[dict] = []
    for _, row in media_windows.iterrows():
        base_row = _aggregate_basic_audio_window(row, frame_df, sample_rate, analysis_frame_seconds, quality_flag, warning)
        _add_extended_audio_window_features(base_row, row, extended_frame_df, analysis_frame_seconds)
        rows.append(base_row)
    return pd.DataFrame(rows, columns=AUDIO_COLUMNS)


def _aggregate_basic_audio_window(
    row: pd.Series,
    frame_df: pd.DataFrame,
    sample_rate: int,
    analysis_frame_seconds: float,
    quality_flag: str,
    warning: str,
) -> dict:
    duration_seconds = float(pd.to_numeric(row["duration_seconds"], errors="coerce"))
    start_offset = float(pd.to_numeric(row["video_start_offset_sec"], errors="coerce"))
    end_offset = start_offset + duration_seconds
    mask = (frame_df["frame_center_sec"] >= start_offset) & (frame_df["frame_center_sec"] < end_offset)
    subset_df = frame_df.loc[mask]
    base_row = _base_audio_row(row, sample_rate, analysis_frame_seconds, quality_flag, warning)
    if subset_df.empty:
        base_row["audio_quality_flag"] = "insufficient_audio_frames"
        base_row["warnings"] = combine_warnings(base_row["warnings"], "No audio frames overlapped the window.")
        return base_row

    base_row.update(
        {
            "audio_available": True,
            "audio_rms": float(subset_df["audio_rms"].mean()),
            "audio_short_time_energy": float(subset_df["audio_short_time_energy"].mean()),
            "audio_zero_crossing_rate": float(subset_df["audio_zero_crossing_rate"].mean()),
            "audio_spectral_centroid": float(subset_df["audio_spectral_centroid"].mean()),
            "audio_spectral_bandwidth": float(subset_df["audio_spectral_bandwidth"].mean()),
            "audio_spectral_rolloff": float(subset_df["audio_spectral_rolloff"].mean()),
            "audio_spectral_flatness": float(subset_df["audio_spectral_flatness"].mean()),
            "audio_quality_flag": quality_flag,
        }
    )
    return base_row


def _add_extended_audio_window_features(
    base_row: dict,
    row: pd.Series,
    extended_frame_df: pd.DataFrame,
    analysis_frame_seconds: float,
) -> None:
    if extended_frame_df.empty:
        return
    start_offset = float(pd.to_numeric(row["video_start_offset_sec"], errors="coerce"))
    duration_seconds = float(pd.to_numeric(row["duration_seconds"], errors="coerce"))
    end_offset = start_offset + duration_seconds
    subset_df = extended_frame_df[
        (extended_frame_df["frame_center_sec"] >= start_offset)
        & (extended_frame_df["frame_center_sec"] < end_offset)
    ].copy()
    if subset_df.empty:
        return

    call_like_stats = _compute_mask_event_statistics(subset_df["frame_call_like"].fillna(False).astype(bool), analysis_frame_seconds)
    chirp_like_stats = _compute_mask_event_statistics(subset_df["frame_chirp_like"].fillna(False).astype(bool), analysis_frame_seconds)
    cluck_like_stats = _compute_mask_event_statistics(subset_df["frame_cluck_like"].fillna(False).astype(bool), analysis_frame_seconds)
    minutes = max(duration_seconds / 60.0, 1e-9)
    base_row.update(
        {
            "audio_energy_entropy": float(pd.to_numeric(subset_df["frame_energy_entropy"], errors="coerce").mean()),
            "audio_band_energy_ratio_2_5khz": float(pd.to_numeric(subset_df["frame_band_energy_ratio_2_5khz"], errors="coerce").mean()),
            "audio_band_energy_ratio_3_6khz": float(pd.to_numeric(subset_df["frame_band_energy_ratio_3_6khz"], errors="coerce").mean()),
            "audio_band_energy_ratio_300_1500hz": float(pd.to_numeric(subset_df["frame_band_energy_ratio_300_1500hz"], errors="coerce").mean()),
            "audio_spectral_flux": float(pd.to_numeric(subset_df["frame_spectral_flux"], errors="coerce").mean()),
            "audio_frame_spectral_flatness_mean": float(pd.to_numeric(subset_df["frame_spectral_flatness"], errors="coerce").mean()),
            "audio_dominant_frequency": float(pd.to_numeric(subset_df["frame_dominant_frequency"], errors="coerce").median()),
            "audio_call_like_event_rate": float(call_like_stats["event_count"] / minutes),
            "audio_call_like_occupancy": float(call_like_stats["occupancy"]),
            "audio_chirp_like_event_rate": float(chirp_like_stats["event_count"] / minutes),
            "audio_chirp_like_occupancy": float(chirp_like_stats["occupancy"]),
            "audio_chirp_like_mean_duration_sec": float(chirp_like_stats["mean_duration_sec"]),
            "audio_cluck_like_event_rate": float(cluck_like_stats["event_count"] / minutes),
            "audio_cluck_like_occupancy": float(cluck_like_stats["occupancy"]),
            "audio_cluck_like_mean_duration_sec": float(cluck_like_stats["mean_duration_sec"]),
        }
    )
    _add_event_audio_window_features(base_row, subset_df, analysis_frame_seconds, duration_seconds)


def _add_event_audio_window_features(
    base_row: dict,
    subset_df: pd.DataFrame,
    analysis_frame_seconds: float,
    duration_seconds: float,
) -> None:
    dense_mask, sparse_mask, disturbance_mask, lowfreq_uncertain_mask, active_mask = _build_event_frame_masks(subset_df)
    bird_mask = dense_mask | sparse_mask
    nonbird_mask = disturbance_mask | lowfreq_uncertain_mask
    dense_stats = _summarize_mask(dense_mask, analysis_frame_seconds, duration_seconds)
    sparse_stats = _summarize_mask(sparse_mask, analysis_frame_seconds, duration_seconds)
    disturbance_stats = _summarize_mask(disturbance_mask, analysis_frame_seconds, duration_seconds)
    lowfreq_uncertain_stats = _summarize_mask(lowfreq_uncertain_mask, analysis_frame_seconds, duration_seconds)
    bird_stats = _summarize_mask(bird_mask, analysis_frame_seconds, duration_seconds)
    nonbird_stats = _summarize_mask(nonbird_mask, analysis_frame_seconds, duration_seconds)
    active_stats = _summarize_mask(active_mask, analysis_frame_seconds, duration_seconds)
    impact_stats = _summarize_mask(disturbance_mask, analysis_frame_seconds, duration_seconds, max_duration_sec=0.5)
    bird_occ = float(bird_stats["occupancy"])
    nonbird_occ = float(nonbird_stats["occupancy"])
    active_occ = float(active_stats["occupancy"])
    base_row.update(
        {
            "audio_bird_event_rate": float(bird_stats["event_rate"]),
            "audio_bird_event_occupancy": bird_occ,
            "audio_bird_event_mean_duration_sec": float(bird_stats["mean_duration_sec"]),
            "audio_dense_bird_event_rate": float(dense_stats["event_rate"]),
            "audio_dense_bird_event_occupancy": float(dense_stats["occupancy"]),
            "audio_dense_bird_event_mean_duration_sec": float(dense_stats["mean_duration_sec"]),
            "audio_sparse_bird_event_rate": float(sparse_stats["event_rate"]),
            "audio_sparse_bird_event_occupancy": float(sparse_stats["occupancy"]),
            "audio_sparse_bird_event_mean_duration_sec": float(sparse_stats["mean_duration_sec"]),
            "audio_nonbird_event_rate": float(nonbird_stats["event_rate"]),
            "audio_nonbird_event_occupancy": nonbird_occ,
            "audio_nonbird_event_mean_duration_sec": float(nonbird_stats["mean_duration_sec"]),
            "audio_disturbance_event_rate": float(disturbance_stats["event_rate"]),
            "audio_disturbance_event_occupancy": float(disturbance_stats["occupancy"]),
            "audio_disturbance_event_mean_duration_sec": float(disturbance_stats["mean_duration_sec"]),
            "audio_impact_like_event_rate": float(impact_stats["event_rate"]),
            "audio_impact_like_occupancy": float(impact_stats["occupancy"]),
            "audio_lowfreq_uncertain_event_rate": float(lowfreq_uncertain_stats["event_rate"]),
            "audio_lowfreq_uncertain_occupancy": float(lowfreq_uncertain_stats["occupancy"]),
            "audio_lowfreq_uncertain_mean_duration_sec": float(lowfreq_uncertain_stats["mean_duration_sec"]),
            "audio_active_event_occupancy": active_occ,
            "audio_bird_minus_nonbird_occupancy": float(bird_occ - nonbird_occ),
            "audio_bird_fraction_within_active": float(bird_occ / active_occ) if active_occ > 0 else 0.0,
            "audio_event_regime_hint": _infer_event_regime_hint(
                bird_occ=bird_occ,
                dense_occ=float(dense_stats["occupancy"]),
                sparse_occ=float(sparse_stats["occupancy"]),
                nonbird_occ=nonbird_occ,
                disturbance_occ=float(disturbance_stats["occupancy"]),
                lowfreq_uncertain_occ=float(lowfreq_uncertain_stats["occupancy"]),
                active_occ=active_occ,
            ),
        }
    )


def _base_audio_row(
    row: pd.Series,
    sample_rate: int,
    analysis_frame_seconds: float,
    quality_flag: str,
    warning: str = "",
) -> dict:
    duration_seconds = float(pd.to_numeric(row["duration_seconds"], errors="coerce"))
    values = {
        "window_id": str(row["window_id"]),
        "media_id": str(row["media_id"]),
        "room_id": str(row["room_id"]),
        "session_id": str(row["session_id"]),
        "start_time": str(row["start_time"]),
        "end_time": str(row["end_time"]),
        "audio_available": False,
        "audio_duration_sec": duration_seconds,
        "audio_sample_rate": sample_rate,
        "audio_analysis_frame_seconds": analysis_frame_seconds,
        "audio_quality_flag": quality_flag,
        "warnings": warning,
    }
    for column in AUDIO_COLUMNS:
        if column not in values:
            values[column] = _default_audio_value(column)
    return values


def _default_audio_value(column: str) -> object:
    if column == "audio_event_regime_hint":
        return "unavailable"
    if column == "audio_regime_hard_label":
        return ""
    if column in {"window_id", "media_id", "room_id", "session_id", "start_time", "end_time", "warnings", "audio_quality_flag"}:
        return ""
    if column == "audio_available":
        return False
    return np.nan


def _unavailable_audio_rows(
    media_windows: pd.DataFrame,
    quality_flag: str,
    sample_rate: int,
    analysis_frame_seconds: float,
    warning: str = "",
) -> pd.DataFrame:
    rows = [_unavailable_audio_row(row, quality_flag, sample_rate, analysis_frame_seconds, warning) for _, row in media_windows.iterrows()]
    return pd.DataFrame(rows, columns=AUDIO_COLUMNS)


def _unavailable_audio_row(
    row: pd.Series,
    quality_flag: str,
    sample_rate: int,
    analysis_frame_seconds: float = 0.1,
    warning: str = "",
) -> dict:
    return _base_audio_row(row, sample_rate, analysis_frame_seconds, quality_flag, warning)


def _build_call_like_mask(frame_df: pd.DataFrame) -> pd.Series:
    if frame_df.empty:
        return pd.Series(dtype=bool)
    rms = pd.to_numeric(frame_df["frame_rms"], errors="coerce")
    band_ratio = pd.to_numeric(frame_df["frame_band_energy_ratio_2_5khz"], errors="coerce")
    spectral_flux = pd.to_numeric(frame_df["frame_spectral_flux"], errors="coerce")
    dominant_frequency = pd.to_numeric(frame_df["frame_dominant_frequency"], errors="coerce")
    rms_threshold = _robust_center(rms)
    band_threshold = float(band_ratio.quantile(0.75))
    flux_threshold = float(spectral_flux.quantile(0.70))
    mask = (
        (band_ratio >= band_threshold)
        & dominant_frequency.between(1000.0, 5000.0, inclusive="both")
        & ((spectral_flux >= flux_threshold) | (rms >= rms_threshold))
    )
    return mask.fillna(False)


def _build_chirp_like_mask(frame_df: pd.DataFrame) -> pd.Series:
    if frame_df.empty:
        return pd.Series(dtype=bool)
    rms = pd.to_numeric(frame_df["frame_rms"], errors="coerce")
    high_band_ratio = pd.to_numeric(frame_df["frame_band_energy_ratio_2_5khz"], errors="coerce")
    chirp_band_ratio = pd.to_numeric(frame_df["frame_band_energy_ratio_3_6khz"], errors="coerce")
    dominant_frequency = pd.to_numeric(frame_df["frame_dominant_frequency"], errors="coerce")
    rms_threshold = _robust_center(rms)
    high_band_threshold = float(high_band_ratio.quantile(0.75))
    chirp_band_threshold = float(chirp_band_ratio.quantile(0.75))
    mask = (
        (high_band_ratio >= high_band_threshold)
        & (chirp_band_ratio >= chirp_band_threshold)
        & dominant_frequency.between(1800.0, 7000.0, inclusive="both")
        & (rms >= rms_threshold)
    )
    return mask.fillna(False)


def _build_cluck_like_mask(frame_df: pd.DataFrame) -> pd.Series:
    if frame_df.empty:
        return pd.Series(dtype=bool)
    rms = pd.to_numeric(frame_df["frame_rms"], errors="coerce")
    chirp_band_ratio = pd.to_numeric(frame_df["frame_band_energy_ratio_3_6khz"], errors="coerce")
    cluck_band_ratio = pd.to_numeric(frame_df["frame_band_energy_ratio_300_1500hz"], errors="coerce")
    dominant_frequency = pd.to_numeric(frame_df["frame_dominant_frequency"], errors="coerce")
    rms_threshold = float(rms.quantile(0.65))
    cluck_band_threshold = float(cluck_band_ratio.quantile(0.80))
    mask = (
        (cluck_band_ratio >= cluck_band_threshold)
        & (cluck_band_ratio > chirp_band_ratio)
        & dominant_frequency.between(250.0, 2000.0, inclusive="both")
        & (rms >= rms_threshold)
    )
    return mask.fillna(False)


def _build_event_frame_masks(subset_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_rms = pd.to_numeric(subset_df["frame_rms"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    frame_flux = pd.to_numeric(subset_df["frame_spectral_flux"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    frame_flatness = pd.to_numeric(subset_df["frame_spectral_flatness"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    band_2_5 = pd.to_numeric(subset_df["frame_band_energy_ratio_2_5khz"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    band_3_6 = pd.to_numeric(subset_df["frame_band_energy_ratio_3_6khz"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    band_300_1500 = pd.to_numeric(subset_df["frame_band_energy_ratio_300_1500hz"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    flux_q90 = float(np.nanquantile(frame_flux, 0.90))
    rms_q60 = float(np.nanquantile(frame_rms, 0.60))
    flux_q60 = float(np.nanquantile(frame_flux, 0.60))
    band_2_5_q70 = float(np.nanquantile(band_2_5, 0.70))
    band_300_q70 = float(np.nanquantile(band_300_1500, 0.70))
    active_mask = (
        (frame_rms >= rms_q60)
        | (frame_flux >= flux_q60)
        | (band_2_5 >= band_2_5_q70)
        | (band_300_1500 >= band_300_q70)
    )
    active_mask = _close_single_frame_gaps(active_mask)
    dense_bird_mask = (
        (((band_2_5 >= 0.022) | (band_3_6 >= 0.0025)) & (frame_flatness <= 0.0020))
        | (
            (band_300_1500 >= 0.11)
            & (band_3_6 >= 0.0020)
            & (frame_flatness <= 0.0080)
            & (frame_flux < flux_q90)
        )
    )
    sparse_bird_mask = (
        (band_300_1500 >= 0.11)
        & (band_300_1500 > band_2_5)
        & (frame_flatness <= 0.0012)
        & (frame_flux < flux_q90)
    )
    lowfreq_uncertain_mask = (
        (band_300_1500 >= 0.11)
        & (band_2_5 < 0.020)
        & (((frame_flatness >= 0.0009) & (frame_flatness <= 0.0080)) | (frame_flux >= flux_q90))
        & (~dense_bird_mask)
        & (~sparse_bird_mask)
        & active_mask
    )
    disturbance_mask = (
        (frame_flux >= max(flux_q90, 0.082))
        & (frame_flatness >= 0.0010)
        & (~dense_bird_mask)
        & (~sparse_bird_mask)
        & active_mask
    )
    dense_bird_mask &= active_mask
    sparse_bird_mask &= active_mask
    return dense_bird_mask, sparse_bird_mask, disturbance_mask, lowfreq_uncertain_mask, active_mask


def _compute_mask_event_statistics(mask: pd.Series, frame_seconds: float) -> dict[str, float]:
    bool_values = mask.fillna(False).astype(bool).to_numpy()
    occupancy = float(bool_values.mean()) if bool_values.size else 0.0
    if not bool_values.size or not bool_values.any():
        return {"event_count": 0.0, "occupancy": occupancy, "mean_duration_sec": 0.0}
    durations_sec = _compute_event_durations(bool_values, frame_seconds)
    return {
        "event_count": float(len(durations_sec)),
        "occupancy": occupancy,
        "mean_duration_sec": float(np.mean(durations_sec)) if durations_sec else 0.0,
    }


def _summarize_mask(
    mask: np.ndarray,
    frame_seconds: float,
    window_seconds: float,
    max_duration_sec: float | None = None,
) -> dict[str, float]:
    durations = _compute_event_durations(mask, frame_seconds)
    if max_duration_sec is not None:
        durations = [value for value in durations if value <= max_duration_sec]
    occupancy = float(sum(durations) / window_seconds) if window_seconds > 0 else 0.0
    event_rate = float(len(durations) / (window_seconds / 60.0)) if window_seconds > 0 else 0.0
    return {
        "occupancy": occupancy,
        "event_rate": event_rate,
        "mean_duration_sec": float(np.mean(durations)) if durations else 0.0,
    }


def _compute_event_durations(mask: np.ndarray, frame_seconds: float) -> list[float]:
    mask_array = np.asarray(mask, dtype=bool)
    durations: list[float] = []
    current_length = 0
    for value in mask_array.tolist():
        if value:
            current_length += 1
        elif current_length > 0:
            durations.append(float(current_length * frame_seconds))
            current_length = 0
    if current_length > 0:
        durations.append(float(current_length * frame_seconds))
    return durations


def _close_single_frame_gaps(mask: np.ndarray) -> np.ndarray:
    output = np.asarray(mask, dtype=bool).copy()
    if output.size < 3:
        return output
    for index in range(1, output.size - 1):
        if not output[index] and output[index - 1] and output[index + 1]:
            output[index] = True
    return output


def _infer_event_regime_hint(
    bird_occ: float,
    dense_occ: float,
    sparse_occ: float,
    nonbird_occ: float,
    disturbance_occ: float,
    lowfreq_uncertain_occ: float,
    active_occ: float,
) -> str:
    if active_occ < 0.05:
        return "background_quiet"
    if bird_occ >= 0.20 and bird_occ > nonbird_occ * 1.5:
        if sparse_occ > dense_occ * 0.8:
            return "bird_dominant_with_sparse_component"
        return "bird_dominant_dense"
    if sparse_occ >= 0.08 and sparse_occ > dense_occ and sparse_occ > nonbird_occ:
        return "bird_dominant_sparse"
    if nonbird_occ >= 0.10 and nonbird_occ > bird_occ * 1.25:
        if lowfreq_uncertain_occ >= disturbance_occ:
            return "background_with_lowfreq_uncertain_events"
        return "background_with_transient_disturbance"
    if bird_occ >= 0.05 and nonbird_occ >= 0.05:
        return "mixed_bird_and_disturbance"
    if bird_occ >= 0.05:
        return "background_with_bird_events"
    if nonbird_occ >= 0.05:
        return "background_with_nonbird_events"
    return "background_low_level_activity"


def _finalize_audio_regime_features(audio_df: pd.DataFrame, smoothing_windows: int) -> pd.DataFrame:
    if audio_df.empty:
        return pd.DataFrame(columns=AUDIO_COLUMNS)
    working_df = audio_df.copy()
    available = working_df["audio_available"].fillna(False).astype(bool)
    for axis_name, component_columns in REGIME_AXIS_COMPONENTS.items():
        component_values = []
        for column in component_columns:
            z_column = f"_{column}_global_robust_z"
            center, scale = _compute_robust_center_scale(working_df.loc[available, column])
            values = pd.to_numeric(working_df[column], errors="coerce")
            working_df[z_column] = (values - center) / scale if np.isfinite(center) and np.isfinite(scale) and scale > 0 else np.nan
            component_values.append(z_column)
        working_df[axis_name] = working_df[component_values].mean(axis=1, skipna=True)
        working_df = working_df.drop(columns=component_values, errors="ignore")

    _assign_regime_scores(working_df, available)
    _add_smoothed_regime_scores(working_df, max(1, int(smoothing_windows)))
    working_df = working_df.drop(columns=["start_time_dt", "end_time_dt"], errors="ignore")
    return working_df.reindex(columns=AUDIO_COLUMNS)


def _assign_regime_scores(working_df: pd.DataFrame, available: pd.Series) -> None:
    for axis_name, score_column in [
        ("audio_dense_regime_axis", "audio_regime_dense_score"),
        ("audio_sparse_regime_axis", "audio_regime_sparse_score"),
    ]:
        series = pd.to_numeric(working_df.loc[available, axis_name], errors="coerce").dropna()
        if series.empty:
            working_df[score_column] = np.nan
            continue
        threshold = float(series.quantile(0.95))
        upper = float(series.quantile(0.99))
        scale = max(upper - threshold, 0.1)
        logits = (pd.to_numeric(working_df[axis_name], errors="coerce") - threshold) / max(scale, 1e-6)
        working_df[score_column] = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    working_df["audio_regime_background_score"] = (
        1.0 - pd.to_numeric(working_df["audio_regime_dense_score"], errors="coerce")
    ) * (
        1.0 - pd.to_numeric(working_df["audio_regime_sparse_score"], errors="coerce")
    )
    score_columns = [
        "audio_regime_dense_score",
        "audio_regime_background_score",
        "audio_regime_sparse_score",
    ]
    labels = ["dense_flock_vocalization", "environmental_noise_only", "sparse_salient_calls"]
    score_df = working_df[score_columns].apply(pd.to_numeric, errors="coerce")
    hard_labels: list[str] = []
    margins: list[float] = []
    for _, row in score_df.iterrows():
        values = row.to_numpy(dtype=float)
        if not np.isfinite(values).any():
            hard_labels.append("")
            margins.append(np.nan)
            continue
        order = np.argsort(-np.nan_to_num(values, nan=-np.inf))
        hard_labels.append(labels[int(order[0])])
        margins.append(float(values[int(order[0])] - values[int(order[1])]) if len(order) > 1 else np.nan)
    working_df["audio_regime_hard_label"] = hard_labels
    working_df["audio_regime_margin"] = margins


def _add_smoothed_regime_scores(working_df: pd.DataFrame, smoothing_windows: int) -> None:
    ordered = prepare_time_columns(working_df).sort_values(["room_id", "session_id", "media_id", "start_time_dt", "window_id"], kind="stable")
    same_media = ordered["media_id"].astype("string").eq(ordered["media_id"].shift().astype("string"))
    gap_seconds = (ordered["start_time_dt"] - ordered["start_time_dt"].shift()).dt.total_seconds()
    segment_break = (~same_media) | gap_seconds.isna() | gap_seconds.gt(15.0)
    segment_id = pd.Series(np.cumsum(segment_break.to_numpy(dtype=bool)), index=ordered.index)
    min_periods = min(3, max(smoothing_windows, 1))
    for score_column in [
        "audio_regime_dense_score",
        "audio_regime_background_score",
        "audio_regime_sparse_score",
    ]:
        smoothed_column = score_column.replace("_score", "_score_smoothed_2min")
        ordered[smoothed_column] = (
            ordered.groupby(segment_id, dropna=False)[score_column]
            .transform(lambda series: series.rolling(window=smoothing_windows, min_periods=min_periods).mean())
        )
    for column in [
        "audio_regime_dense_score_smoothed_2min",
        "audio_regime_background_score_smoothed_2min",
        "audio_regime_sparse_score_smoothed_2min",
    ]:
        working_df[column] = ordered.sort_index()[column]


def _compute_robust_center_scale(series: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan"), float("nan")
    values = numeric.to_numpy(dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    std = float(np.std(values, ddof=0))
    scale = max(mad * 1.4826, iqr / 1.349 if iqr > 0 else 0.0, std, 1e-6)
    return median, scale


def _robust_center(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.median())


def _sort_audio_df(audio_df: pd.DataFrame) -> pd.DataFrame:
    if audio_df.empty:
        return pd.DataFrame(columns=AUDIO_COLUMNS)
    ordered = prepare_time_columns(audio_df).sort_values(["room_id", "session_id", "start_time_dt", "window_id"], kind="stable")
    return ordered.drop(columns=["start_time_dt", "end_time_dt"], errors="ignore").reset_index(drop=True).reindex(columns=AUDIO_COLUMNS)


def _failed_row(row: pd.Series, issue: str) -> dict:
    return {
        "stage": "audio_features",
        "window_id": str(row.get("window_id", "")),
        "media_id": str(row.get("media_id", "")),
        "room_id": str(row.get("room_id", "")),
        "issue": issue,
    }


def _build_dry_run_report(window_df: pd.DataFrame, enabled: bool) -> str:
    return "\n".join(
        [
            "# Audio Feature Report",
            "",
            f"- Audio extraction enabled: `{enabled}`",
            "- Dry run mode skipped heavy embedded-audio extraction.",
            f"- Planned windows: {len(window_df)}",
            "- Extended audio frame seconds: 0.1 by default.",
        ]
    ) + "\n"


def _build_audio_report(window_df: pd.DataFrame, audio_df: pd.DataFrame, failed_df: pd.DataFrame) -> str:
    available_df = audio_df[audio_df["audio_available"].fillna(False).astype(bool)].copy() if not audio_df.empty else pd.DataFrame()
    summary_columns = [
        "audio_rms",
        "audio_energy_entropy",
        "audio_band_energy_ratio_2_5khz",
        "audio_spectral_flux",
        "audio_chirp_like_occupancy",
        "audio_cluck_like_occupancy",
        "audio_bird_event_occupancy",
        "audio_nonbird_event_occupancy",
        "audio_regime_dense_score",
        "audio_regime_background_score",
        "audio_regime_sparse_score",
    ]
    lines = [
        "# Audio Feature Report",
        "",
        "- Audio comes from embedded MP4 audio only. No external WAV files are used.",
        "- Basic features use 1.0 s frames; extended acoustic/event proxies use 0.1 s frames.",
        f"- Windows attempted: {len(window_df)}",
        f"- Windows with audio available: {int(audio_df['audio_available'].fillna(False).astype(bool).sum()) if not audio_df.empty else 0}",
        f"- Failed or unavailable windows: {len(failed_df)}",
        "",
        "## Feature Summary",
        "",
        "```text",
        available_df[[column for column in summary_columns if column in available_df.columns]].describe().to_string()
        if not available_df.empty
        else "No successful audio rows available.",
        "```",
    ]
    return "\n".join(lines) + "\n"
