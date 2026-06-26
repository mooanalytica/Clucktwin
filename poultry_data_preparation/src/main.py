from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .audio_feature_extractor import extract_audio_features
from .config import PreparationConfig, ensure_output_dirs, load_config
from .env_loader import load_environment_tables
from .media_indexer import index_media
from .merge_outputs import merge_processed_outputs
from .semantic_zone_loader import load_semantic_zone_configs
from .utils import configure_logging, read_csv_if_exists
from .video_feature_extractor import build_semantic_biomarker_table, extract_video_features
from .window_indexer import index_windows


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable poultry data-preparation pipeline.")
    parser.add_argument("--config", default="poultry_data_preparation/config/default.yaml", help="Path to YAML config.")
    parser.add_argument("--stage", default="all", choices=["index", "zones", "video_features", "audio_features", "env", "merge", "all"], help="Stage to run.")
    parser.add_argument("--dry-run", action="store_true", help="Scan files and write reports, but skip heavy video and audio extraction.")
    parser.add_argument("--max-windows", type=int, default=None, help="Override max_windows for small test runs.")
    parser.add_argument("--max-media", type=int, default=None, help="Override max_media for small test runs.")
    parser.add_argument("--force-recompute-video", action="store_true", help="Ignore cached video features and recompute them for this run.")
    parser.add_argument("--force-recompute-audio", action="store_true", help="Ignore cached audio features and recompute them for this run.")
    parser.add_argument("--force-recompute-all", action="store_true", help="Ignore both video and audio caches for this run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_pipeline(
            config_path=args.config,
            stage=args.stage,
            dry_run=args.dry_run,
            max_windows=args.max_windows,
            max_media=args.max_media,
            force_recompute_video=args.force_recompute_video,
            force_recompute_audio=args.force_recompute_audio,
            force_recompute_all=args.force_recompute_all,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        logging.getLogger(__name__).error(str(exc))
        return 1
    return 0


def run_pipeline(
    config_path: str | Path,
    stage: str = "all",
    dry_run: bool = False,
    max_windows: int | None = None,
    max_media: int | None = None,
    force_recompute_video: bool = False,
    force_recompute_audio: bool = False,
    force_recompute_all: bool = False,
) -> dict[str, str]:
    config = load_config(config_path, max_windows=max_windows, max_media=max_media)
    if force_recompute_all or force_recompute_video:
        config.video_force_recompute = True
    if force_recompute_all or force_recompute_audio:
        config.audio_force_recompute = True
    ensure_output_dirs(config)
    configure_logging(config.log_path)
    LOGGER.info("Starting poultry_data_preparation stage=%s dry_run=%s", stage, dry_run)

    run_index = stage in {"index", "video_features", "audio_features", "merge", "all"}
    run_zones = stage in {"zones", "video_features", "merge", "all"}
    run_video = stage in {"video_features", "merge", "all"}
    run_audio = stage in {"audio_features", "merge", "all"}
    run_env = stage in {"env", "merge", "all"}
    run_merge = stage in {"merge", "all"}

    media_df = read_csv_if_exists(config.metadata_output_dir / "media_manifest.csv")
    window_df = read_csv_if_exists(config.metadata_output_dir / "video_window_index.csv")
    zone_configs: list[dict] = []
    zone_feature_df = read_csv_if_exists(config.features_output_dir / "semantic_zone_video_features.csv")
    biomarker_df = read_csv_if_exists(config.features_output_dir / "semantic_biomarker_window_table.csv")
    audio_df = read_csv_if_exists(config.features_output_dir / "audio_window_features.csv")
    env_df = read_csv_if_exists(config.features_output_dir / "env_daily.csv")
    failed_frames = []

    if run_index or media_df.empty or window_df.empty:
        media_result = index_media(config, dry_run=dry_run)
        media_df = media_result.manifest_df
        window_result = index_windows(media_df, config)
        window_df = window_result.window_df

    if run_zones:
        zone_result = load_semantic_zone_configs(config, dry_run=dry_run)
        zone_configs = zone_result.zone_configs
    else:
        zone_config_path = config.zones_output_dir / "semantic_zone_configs.json"
        if zone_config_path.exists():
            zone_configs = json.loads(zone_config_path.read_text(encoding="utf-8"))

    if run_video:
        video_result = extract_video_features(config, media_df, window_df, zone_configs, dry_run=dry_run)
        zone_feature_df = getattr(video_result, "zone_feature_df", zone_feature_df)
        biomarker_df = video_result.biomarker_df
        failed_frames.append(video_result.failed_df)

    if run_audio:
        audio_result = extract_audio_features(config, media_df, window_df, dry_run=dry_run)
        audio_df = audio_result.audio_df
        failed_frames.append(audio_result.failed_df)

    if run_env:
        env_result = load_environment_tables(config)
        env_df = env_result.env_df

    if run_merge and not dry_run:
        merge_result = merge_processed_outputs(
            config,
            media_df,
            window_df,
            biomarker_df,
            audio_df,
            env_df,
            failed_frames=failed_frames,
            zone_feature_df=zone_feature_df,
        )
    elif run_merge:
        merge_result = None
        (config.reports_output_dir / "processed_multimodal_handoff_report.md").write_text(
            "# Processed Multimodal Handoff Report\n\nDry run mode skipped the merged handoff table.\n",
            encoding="utf-8",
        )
    else:
        merge_result = None

    outputs = {
        "media_manifest": str(config.metadata_output_dir / "media_manifest.csv"),
        "video_window_index": str(config.metadata_output_dir / "video_window_index.csv"),
        "semantic_zone_configs": str(config.zones_output_dir / "semantic_zone_configs.json"),
        "semantic_zone_video_features": str(config.features_output_dir / "semantic_zone_video_features.csv"),
        "semantic_biomarker_window_table": str(config.features_output_dir / "semantic_biomarker_window_table.csv"),
        "audio_window_features": str(config.features_output_dir / "audio_window_features.csv"),
        "env_daily": str(config.features_output_dir / "env_daily.csv"),
        "processed_multimodal_window_table": str(config.features_output_dir / "processed_multimodal_window_table.csv"),
    }
    if merge_result is not None:
        outputs["handoff_readme"] = str(merge_result.handoff_readme_path)
    return outputs


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
