from __future__ import annotations

from pathlib import Path
import sys
import uuid

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ensure_output_dirs, load_config
from src.merge_outputs import REQUIRED_PROCESSED_COLUMNS, merge_processed_outputs


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
            "enabled": True,
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


def test_merge_outputs_required_columns_and_no_duplicate_window_ids() -> None:
    scratch_root = Path("tmp_tests")
    scratch_root.mkdir(parents=True, exist_ok=True)
    tmp_path = scratch_root / f"merge_test_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = _build_test_config(tmp_path)

    media_df = pd.DataFrame(
        [
            {
                "media_id": "media_000001",
                "room_id": "room_1",
                "session_id": "room_1_room_1_16_17_aug",
                "file_name": "test.mp4",
                "relative_path": "data/raw/video/Room 1/Room 1 (16, 17 Aug)/test.mp4",
                "absolute_path": str((tmp_path / "data" / "raw" / "video" / "Room 1" / "Room 1 (16, 17 Aug)" / "test.mp4").resolve()),
                "video_path": "data/raw/video/Room 1/Room 1 (16, 17 Aug)/test.mp4",
                "start_time": "2025-08-16T06:00:00-03:00",
                "end_time": "2025-08-16T06:01:00-03:00",
                "has_audio": True,
            }
        ]
    )
    window_df = pd.DataFrame(
        [
            {
                "window_id": "media_000001_w0001",
                "media_id": "media_000001",
                "room_id": "room_1",
                "session_id": "room_1_room_1_16_17_aug",
                "start_time": "2025-08-16T06:00:00-03:00",
                "end_time": "2025-08-16T06:00:30-03:00",
                "duration_seconds": 30.0,
                "video_path": "data/raw/video/Room 1/Room 1 (16, 17 Aug)/test.mp4",
                "video_start_offset_sec": 0.0,
                "has_audio": True,
                "quality_status": "ok",
                "warnings": "",
            }
        ]
    )
    biomarker_df = pd.DataFrame(
        [
            {
                "window_id": "media_000001_w0001",
                "media_id": "media_000001",
                "room_id": "room_1",
                "session_id": "room_1_room_1_16_17_aug",
                "start_time": "2025-08-16T06:00:00-03:00",
                "end_time": "2025-08-16T06:00:30-03:00",
                "activity_mean": 0.5,
                "normalized_activity": 0.5,
                "mobility_index": 0.5,
                "spatial_freedom_index": 0.8,
                "occupancy_imbalance_index": 0.2,
                "semantic_transition_proxy": 0.1,
                "drinking_activity_fraction": 0.2,
                "feeding_activity_fraction": 0.3,
                "open_movement_activity_fraction": 0.35,
                "resting_activity_fraction": 0.15,
                "general_activity_fraction": 0.5,
                "feeding_plus_drinking_activity_fraction": 0.5,
                "drinking_to_feeding_activity_ratio": 0.666,
                "quality_status": "ok",
                "warnings": "",
            }
        ]
    )
    audio_df = pd.DataFrame(
        [
            {
                "window_id": "media_000001_w0001",
                "media_id": "media_000001",
                "room_id": "room_1",
                "session_id": "room_1_room_1_16_17_aug",
                "start_time": "2025-08-16T06:00:00-03:00",
                "end_time": "2025-08-16T06:00:30-03:00",
                "audio_available": True,
                "audio_duration_sec": 30.0,
                "audio_sample_rate": 22050,
                "audio_rms": 0.1,
                "audio_short_time_energy": 0.01,
                "audio_zero_crossing_rate": 0.02,
                "audio_spectral_centroid": 1500.0,
                "audio_spectral_bandwidth": 1000.0,
                "audio_spectral_rolloff": 2000.0,
                "audio_spectral_flatness": 0.2,
                "audio_quality_flag": "ok",
                "warnings": "",
            }
        ]
    )
    env_df = pd.DataFrame(
        [
            {
                "room_id": "room_1",
                "date": "2025-08-16",
                "temp_am_min": 20.0,
                "temp_am_max": 24.0,
                "temp_pm": 25.0,
                "rh_am": 50.0,
                "rh_pm": 55.0,
                "temp_am_mean": 22.0,
                "temp_daily_mean": 23.5,
                "rh_daily_mean": 52.5,
                "temp_daily_range": 4.0,
                "env_quality_flag": "ok",
            }
        ]
    )
    zone_feature_df = pd.DataFrame(
        [
            {
                "window_id": "media_000001_w0001",
                "media_id": "media_000001",
                "room_id": "room_1",
                "session_id": "room_1_room_1_16_17_aug",
                "zone_id": "feeding_zone",
                "semantic_type": "feeding",
                "zone_activity_fraction": 0.3,
            }
        ]
    )

    media_df.to_csv(config.metadata_output_dir / "media_manifest.csv", index=False)
    window_df.to_csv(config.metadata_output_dir / "video_window_index.csv", index=False)
    biomarker_df.to_csv(config.features_output_dir / "semantic_biomarker_window_table.csv", index=False)
    audio_df.to_csv(config.features_output_dir / "audio_window_features.csv", index=False)
    env_df.to_csv(config.features_output_dir / "env_daily.csv", index=False)
    zone_feature_df.to_csv(config.features_output_dir / "semantic_zone_video_features.csv", index=False)
    (config.zones_output_dir / "semantic_zone_configs.json").write_text("[]", encoding="utf-8")

    merge_result = merge_processed_outputs(
        config,
        media_df,
        window_df,
        biomarker_df,
        audio_df,
        env_df,
        failed_frames=None,
        zone_feature_df=zone_feature_df,
    )
    processed_df = merge_result.processed_df
    assert processed_df["window_id"].nunique() == len(processed_df)
    for column in REQUIRED_PROCESSED_COLUMNS:
        assert column in processed_df.columns

    session_dir = config.output_root / "Room 1" / "Room 1 (16, 17 Aug)"
    assert session_dir.exists()
    partition_processed_df = pd.read_csv(session_dir / "processed_multimodal_window_table.csv")
    partition_audio_df = pd.read_csv(session_dir / "audio_window_features.csv")
    partition_zone_df = pd.read_csv(session_dir / "semantic_zone_video_features.csv")
    assert partition_processed_df["window_id"].tolist() == ["media_000001_w0001"]
    assert partition_audio_df["audio_rms"].tolist() == [0.1]
    assert partition_zone_df["zone_id"].tolist() == ["feeding_zone"]
