from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import main as prep_main
from src.config import load_config


def _write_config(base_path: Path) -> Path:
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
    return config_path


def test_parse_args_accepts_force_recompute_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prep",
            "--force-recompute-video",
            "--force-recompute-audio",
            "--force-recompute-all",
        ],
    )

    args = prep_main.parse_args()

    assert args.force_recompute_video is True
    assert args.force_recompute_audio is True
    assert args.force_recompute_all is True


def test_run_pipeline_applies_force_recompute_overrides(monkeypatch) -> None:
    scratch_root = Path("tmp_tests")
    scratch_root.mkdir(parents=True, exist_ok=True)
    tmp_path = scratch_root / f"main_test_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = _write_config(tmp_path)
    seen = {"video": None, "audio": None}

    def fake_index_media(config, dry_run=False):
        return SimpleNamespace(manifest_df=pd.DataFrame([{"media_id": "media_000001"}]))

    def fake_index_windows(media_df, config):
        return SimpleNamespace(window_df=pd.DataFrame([{"window_id": "media_000001_w0001", "media_id": "media_000001"}]))

    def fake_load_semantic_zone_configs(config, dry_run=False):
        return SimpleNamespace(zone_configs=[])

    def fake_extract_video_features(config, media_df, window_df, zone_configs, dry_run=False):
        seen["video"] = config.video_force_recompute
        return SimpleNamespace(biomarker_df=pd.DataFrame(), failed_df=pd.DataFrame())

    def fake_extract_audio_features(config, media_df, window_df, dry_run=False):
        seen["audio"] = config.audio_force_recompute
        return SimpleNamespace(audio_df=pd.DataFrame(), failed_df=pd.DataFrame())

    def fake_load_environment_tables(config):
        return SimpleNamespace(env_df=pd.DataFrame())

    def fake_merge_processed_outputs(
        config,
        media_df,
        window_df,
        biomarker_df,
        audio_df,
        env_df,
        failed_frames,
        zone_feature_df=None,
    ):
        return SimpleNamespace(handoff_readme_path=config.handoff_output_dir / "README_HANDOFF.md")

    monkeypatch.setattr(prep_main, "index_media", fake_index_media)
    monkeypatch.setattr(prep_main, "index_windows", fake_index_windows)
    monkeypatch.setattr(prep_main, "load_semantic_zone_configs", fake_load_semantic_zone_configs)
    monkeypatch.setattr(prep_main, "extract_video_features", fake_extract_video_features)
    monkeypatch.setattr(prep_main, "extract_audio_features", fake_extract_audio_features)
    monkeypatch.setattr(prep_main, "load_environment_tables", fake_load_environment_tables)
    monkeypatch.setattr(prep_main, "merge_processed_outputs", fake_merge_processed_outputs)

    prep_main.run_pipeline(
        config_path=config_path,
        stage="all",
        force_recompute_all=True,
    )

    assert seen["video"] is True
    assert seen["audio"] is True


def test_load_config_can_timestamp_output_root() -> None:
    scratch_root = Path("tmp_tests")
    scratch_root.mkdir(parents=True, exist_ok=True)
    tmp_path = scratch_root / f"timestamp_config_test_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = _write_config(tmp_path)
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["output"]["timestamp_output_root"] = True
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

    config = load_config(config_path)

    assert config.timestamp_output_root is True
    assert config.run_timestamp is not None
    assert len(config.run_timestamp) == len("20260625_113000")
    assert config.output_root.name == f"outputs_{config.run_timestamp}"
