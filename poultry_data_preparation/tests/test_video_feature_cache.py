from __future__ import annotations

from pathlib import Path
import sys
import uuid

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ensure_output_dirs, load_config
from src.video_feature_extractor import (
    _build_video_cache_metadata,
    _load_or_compute_media_cache,
)


def _build_test_config(base_path: Path):
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
            "enabled": True,
            "method": "frame_difference",
            "resize_width": 320,
            "max_windows": None,
            "max_windows_per_room": None,
            "max_windows_per_media": None,
            "max_media": None,
            "force_recompute": False,
        },
        "audio_features": {
            "enabled": False,
            "sample_rate": 22050,
            "frame_seconds": 1.0,
            "cache_per_media_audio": True,
            "force_recompute": False,
        },
        "semantic_zones": {
            "enabled": True,
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


def _build_zone_config(feeding_right_edge: int) -> dict:
    return {
        "room_id": "room_1",
        "zone_config_id": "room_1_semantic_v2",
        "image_width": 1920,
        "image_height": 1080,
        "zones": [
            {
                "zone_id": "drinking_zone",
                "display_name": "Drinking Area",
                "semantic_type": "drinking",
                "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
            },
            {
                "zone_id": "feeding_zone",
                "display_name": "Feeding Area",
                "semantic_type": "feeding",
                "polygon": [[300, 300], [feeding_right_edge, 300], [feeding_right_edge, 400], [300, 400]],
            },
            {
                "zone_id": "open_movement_zone",
                "display_name": "Open Movement",
                "semantic_type": "open_movement",
                "polygon": [[500, 500], [700, 500], [700, 700], [500, 700]],
            },
            {
                "zone_id": "resting_zone",
                "display_name": "Resting Area",
                "semantic_type": "resting",
                "polygon": [[900, 150], [1100, 150], [1100, 350], [900, 350]],
            },
        ],
    }


def _build_media_windows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_id": "media_000001_w0001",
                "media_id": "media_000001",
                "room_id": "room_1",
                "session_id": "room_1_test",
                "start_time": "2025-08-16T06:00:00",
                "end_time": "2025-08-16T06:00:30",
                "video_path": "data/raw/video/Room 1/test.mp4",
                "video_start_offset_sec": 0.0,
                "duration_seconds": 30.0,
                "quality_status": "ok",
                "warnings": "",
            }
        ]
    )


def _fresh_cache(interval_value: float) -> dict[str, object]:
    return {
        "interval_center_sec": np.array([interval_value], dtype=float),
        "zone_metrics": {
            "drinking_zone": {
                "activity_mean": np.array([0.1], dtype=float),
                "activity_std": np.array([0.01], dtype=float),
                "activity_sum": np.array([1.0], dtype=float),
            },
            "feeding_zone": {
                "activity_mean": np.array([0.2], dtype=float),
                "activity_std": np.array([0.02], dtype=float),
                "activity_sum": np.array([2.0], dtype=float),
            },
            "open_movement_zone": {
                "activity_mean": np.array([0.3], dtype=float),
                "activity_std": np.array([0.03], dtype=float),
                "activity_sum": np.array([3.0], dtype=float),
            },
            "resting_zone": {
                "activity_mean": np.array([0.4], dtype=float),
                "activity_std": np.array([0.04], dtype=float),
                "activity_sum": np.array([4.0], dtype=float),
            },
        },
        "warnings": [],
    }


def test_video_feature_cache_recomputes_legacy_cache(monkeypatch) -> None:
    scratch_root = Path("tmp_tests")
    scratch_root.mkdir(parents=True, exist_ok=True)
    tmp_path = scratch_root / f"video_cache_test_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = _build_test_config(tmp_path)

    absolute_path = tmp_path / "data" / "raw" / "video" / "Room 1" / "test.mp4"
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_bytes(b"fake-video")

    media_windows = _build_media_windows()
    zone_config = _build_zone_config(feeding_right_edge=420)
    cache_path = config.video_feature_cache_dir / "media_000001.joblib"
    joblib.dump(_fresh_cache(interval_value=1.0), cache_path)

    replacement_cache = _fresh_cache(interval_value=9.0)

    def fake_compute(*args, **kwargs):
        return replacement_cache.copy()

    monkeypatch.setattr("src.video_feature_extractor._compute_media_interval_metrics", fake_compute)

    result = _load_or_compute_media_cache(config, zone_config, absolute_path, media_windows, force_recompute=False)

    assert np.array_equal(result["interval_center_sec"], np.array([9.0], dtype=float))
    reloaded = joblib.load(cache_path)
    assert np.array_equal(reloaded["interval_center_sec"], np.array([9.0], dtype=float))
    assert "cache_metadata" in reloaded


def test_video_feature_cache_recomputes_when_zone_signature_changes(monkeypatch) -> None:
    scratch_root = Path("tmp_tests")
    scratch_root.mkdir(parents=True, exist_ok=True)
    tmp_path = scratch_root / f"video_cache_test_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = _build_test_config(tmp_path)

    absolute_path = tmp_path / "data" / "raw" / "video" / "Room 1" / "test.mp4"
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_bytes(b"fake-video")

    media_windows = _build_media_windows()
    original_zone = _build_zone_config(feeding_right_edge=420)
    updated_zone = _build_zone_config(feeding_right_edge=520)
    cached = _fresh_cache(interval_value=1.0)
    cached["cache_metadata"] = _build_video_cache_metadata(config, original_zone, absolute_path)
    cache_path = config.video_feature_cache_dir / "media_000001.joblib"
    joblib.dump(cached, cache_path)

    replacement_cache = _fresh_cache(interval_value=7.0)

    def fake_compute(*args, **kwargs):
        return replacement_cache.copy()

    monkeypatch.setattr("src.video_feature_extractor._compute_media_interval_metrics", fake_compute)

    result = _load_or_compute_media_cache(config, updated_zone, absolute_path, media_windows, force_recompute=False)

    assert np.array_equal(result["interval_center_sec"], np.array([7.0], dtype=float))
    reloaded = joblib.load(cache_path)
    assert np.array_equal(reloaded["interval_center_sec"], np.array([7.0], dtype=float))
    assert reloaded["cache_metadata"]["zone_signature"] == _build_video_cache_metadata(config, updated_zone, absolute_path)["zone_signature"]
