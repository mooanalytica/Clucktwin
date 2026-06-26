from __future__ import annotations

from pathlib import Path
import sys
import uuid

import joblib
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_feature_extractor import (
    _add_smoothed_regime_scores,
    _build_audio_cache_metadata,
    _load_or_compute_audio_cache,
)
from src.config import ensure_output_dirs, load_config


def _build_test_config(base_path: Path, sample_rate: int = 22050, frame_seconds: float = 1.0):
    config_dir = base_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "test.yaml"
    config_payload = {
        "paths": {
            "project_root": "..",
            "raw_root": "data/raw",
            "metadata_root": "data/metadata",
            "output_root": "outputs",
            "cache_root": "outputs/cache",
        },
        "inputs": {
            "video_globs": ["video/Room */**/*.MP4"],
            "env_glob": "env/Combined Room *.xlsx",
            "semantic_zone_ref_dir": "semantic_zone_refs",
            "caretaker_event_glob": [],
        },
        "rooms": {"include": "auto", "exclude": []},
        "windowing": {"window_seconds": 30, "stride_seconds": 10},
        "video_features": {
            "enabled": False,
            "method": "frame_difference",
            "resize_width": 320,
            "max_windows": None,
            "max_windows_per_room": None,
            "max_windows_per_media": None,
            "max_media": None,
            "force_recompute": False,
        },
        "audio_features": {
            "enabled": True,
            "sample_rate": sample_rate,
            "frame_seconds": frame_seconds,
            "cache_per_media_audio": True,
            "force_recompute": False,
        },
        "semantic_zones": {
            "enabled": False,
            "auto_detect_from_annotated_image": True,
            "allow_manual_fallback": True,
        },
        "output": {
            "write_csv": True,
            "write_parquet": False,
            "copy_manifest_for_mvp": True,
            "paths_in_outputs": "relative",
        },
    }
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    config = load_config(config_path)
    ensure_output_dirs(config)
    return config


def _fresh_audio_cache(feature_value: float) -> dict[str, object]:
    return {
        "frame_df": pd.DataFrame(
            [
                {
                    "frame_start_sec": 0.0,
                    "frame_center_sec": 0.5,
                    "audio_rms": feature_value,
                    "audio_short_time_energy": feature_value + 1.0,
                    "audio_zero_crossing_rate": feature_value + 2.0,
                    "audio_spectral_centroid": feature_value + 3.0,
                    "audio_spectral_bandwidth": feature_value + 4.0,
                    "audio_spectral_rolloff": feature_value + 5.0,
                    "audio_spectral_flatness": feature_value + 6.0,
                }
            ]
        ),
        "quality_flag": "ok",
        "warning": "",
    }


def test_audio_feature_cache_recomputes_legacy_cache(monkeypatch) -> None:
    scratch_root = Path("tmp_tests")
    scratch_root.mkdir(parents=True, exist_ok=True)
    tmp_path = scratch_root / f"audio_cache_test_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = _build_test_config(tmp_path)

    absolute_path = tmp_path / "data" / "raw" / "video" / "Room 1" / "test.mp4"
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_bytes(b"fake-video-with-audio")

    cache_path = config.audio_feature_cache_dir / "media_000001.joblib"
    joblib.dump(_fresh_audio_cache(feature_value=1.0), cache_path)

    replacement_cache = _fresh_audio_cache(feature_value=9.0)

    def fake_decode(*args, **kwargs):
        return replacement_cache.copy()

    monkeypatch.setattr("src.audio_feature_extractor._decode_media_audio_to_frames", fake_decode)

    result = _load_or_compute_audio_cache(config, absolute_path, "media_000001", force_recompute=False)

    assert result["frame_df"]["audio_rms"].tolist() == [9.0]
    reloaded = joblib.load(cache_path)
    assert reloaded["frame_df"]["audio_rms"].tolist() == [9.0]
    assert "cache_metadata" in reloaded


def test_audio_feature_cache_recomputes_when_audio_parameters_change(monkeypatch) -> None:
    scratch_root = Path("tmp_tests")
    scratch_root.mkdir(parents=True, exist_ok=True)
    tmp_path = scratch_root / f"audio_cache_test_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = _build_test_config(tmp_path, sample_rate=22050, frame_seconds=1.0)

    absolute_path = tmp_path / "data" / "raw" / "video" / "Room 1" / "test.mp4"
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_bytes(b"fake-video-with-audio")

    cached = _fresh_audio_cache(feature_value=1.0)
    cached["cache_metadata"] = _build_audio_cache_metadata(config, absolute_path)
    cache_path = config.audio_feature_cache_dir / "media_000001.joblib"
    joblib.dump(cached, cache_path)

    replacement_cache = _fresh_audio_cache(feature_value=7.0)

    def fake_decode(*args, **kwargs):
        return replacement_cache.copy()

    monkeypatch.setattr("src.audio_feature_extractor._decode_media_audio_to_frames", fake_decode)

    config.audio_sample_rate = 44100
    result = _load_or_compute_audio_cache(config, absolute_path, "media_000001", force_recompute=False)

    assert result["frame_df"]["audio_rms"].tolist() == [7.0]
    reloaded = joblib.load(cache_path)
    assert reloaded["frame_df"]["audio_rms"].tolist() == [7.0]
    assert reloaded["cache_metadata"]["audio_sample_rate"] == 44100


def test_audio_regime_smoothing_handles_pyarrow_string_storage() -> None:
    pytest.importorskip("pyarrow")
    previous_storage = pd.options.mode.string_storage
    pd.options.mode.string_storage = "pyarrow"
    try:
        audio_df = pd.DataFrame(
            {
                "window_id": ["media_000001_w0001", "media_000001_w0002"],
                "media_id": ["media_000001", "media_000001"],
                "room_id": ["room_1", "room_1"],
                "session_id": ["room_1_test", "room_1_test"],
                "start_time": ["2025-08-16T06:00:00-03:00", "2025-08-16T06:00:01-03:00"],
                "end_time": ["2025-08-16T06:00:01-03:00", "2025-08-16T06:00:02-03:00"],
                "audio_regime_dense_score": [0.2, 0.4],
                "audio_regime_background_score": [0.7, 0.5],
                "audio_regime_sparse_score": [0.1, 0.1],
            }
        )

        _add_smoothed_regime_scores(audio_df, smoothing_windows=2)
    finally:
        pd.options.mode.string_storage = previous_storage

    assert "audio_regime_dense_score_smoothed_2min" in audio_df.columns
