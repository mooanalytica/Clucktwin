# Poultry Data Preparation

This repository prepares raw poultry-house video, embedded MP4 audio, semantic-zone reference images, and room-level environment spreadsheets into window-level CSV tables for downstream analysis.

The pipeline is designed to run either with a normal Python environment or through the bundled Apptainer/Singularity container. On servers, use the Apptainer wrapper so the container supplies dependencies while the checked-out source code remains the active code.

## What The Pipeline Does

For each raw MP4 video, the pipeline:

1. indexes media files and technical metadata;
2. creates fixed time windows;
3. detects semantic zones from annotated reference images;
4. extracts semantic-zone video activity features;
5. extracts embedded-audio features, including extended acoustic/event/regime features;
6. loads daily room environment data;
7. merges everything into global handoff tables;
8. writes smaller Room/session-specific copies for easier inspection.

Videos are never copied into the output folder. Only metadata, features, reports, overlays, and CSV handoff files are written.

## Expected Project Layout

The recommended Project layout is:

```text
Project_Root/
  data/
    raw/
      video/
        Room 1/
          Room 1 (10, 11, 12, 13 Aug)/
            *.MP4
        Room 2/
          Room 2 (10, 11, 12, 13 Aug)/
            *.MP4
      env/
        Combined Room 1.xlsx
        Combined Room 2.xlsx
    metadata/
      semantic_zone_refs/
        room1_reference.png
        room1_reference_with_notes_polygon.png
        room2_reference.png
        room2_reference_with_notes_polygon.png
	by_session/
          Room 1/
            Room 1 (16, 17 Aug)/
              reference.png
              reference_with_notes_polygon.png
  poultry_data_preparation/
    config/
    src/
    container_related/
    containers/
      poultry_data_preparation.sif
```

The included server configs assume this layout. In particular, the config files live under `poultry_data_preparation/config/`, and use:

```yaml
paths:
  project_root: "../.."
  raw_root: "data/raw"
  metadata_root: "data/metadata"
```

## Installation For Local Python Runs

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r poultry_data_preparation/requirements.txt
```

System tools are also required:

- `ffmpeg`
- `ffprobe`
- `exiftool`

On a cluster/server, the recommended route is to use Apptainer instead of installing these manually.

## Run With Python

Run a small smoke test:

```bash
cd /path/to/Poultry_Digital_Twin_Project
python -m poultry_data_preparation.src.main \
  --config poultry_data_preparation/config/workspace_min_smoke_0625.yaml \
  --stage all \
  --force-recompute-all
```

Run the default all-room config:

```bash
python -m poultry_data_preparation.src.main \
  --config poultry_data_preparation/config/default.yaml \
  --stage all
```

Useful CLI flags:

```bash
--stage index|zones|video_features|audio_features|env|merge|all
--dry-run
--max-media 2
--max-windows 100
--force-recompute-video
--force-recompute-audio
--force-recompute-all
```

## Run With Apptainer

Use the wrapper script instead of calling the SIF directly:

```bash
cd /path/to/Poultry_Digital_Twin_Project
bash poultry_data_preparation/container_related/run_apptainer_pipeline.sh \
  --config poultry_data_preparation/config/default.yaml \
  --stage all
```

The wrapper:

- finds `poultry_data_preparation/containers/poultry_data_preparation.sif`;
- binds the current project root into the container;
- sets `PYTHONPATH` so the checked-out source tree is used before any code baked into the SIF;
- runs `python -m poultry_data_preparation.src.main`.

This matters if the SIF was built before the latest code changes. Do not run the SIF directly unless you know the code inside it is current.

Optional cache/temp locations:

```bash
export APPTAINER_CACHEDIR=/path/to/scratch/apptainer_cache
export APPTAINER_TMPDIR=/path/to/scratch/apptainer_tmp
```

## SSH sessions
For long SSH sessions, run inside `tmux`:

```bash
cd /path/to/Poultry_Digital_Twin_Project
tmux new -s poultry_full

export APPTAINER_CACHEDIR=$PWD/poultry_data_preparation/apptainer_cache
export APPTAINER_TMPDIR=$PWD/poultry_data_preparation/apptainer_tmp

bash run_room_jobs.sh smoke
bash run_room_jobs.sh full
```

Detach with `Ctrl-b` then `d`. Reattach with:

```bash
tmux attach -t poultry_full
```

Monitor logs:

```bash
tail -f poultry_data_preparation/run_logs/full_room1_*.log
tail -f poultry_data_preparation/run_logs/full_room2_*.log
```

## Included Configs

Common configs:

```text
config/default.yaml
  Generic all-room standalone layout.

config/workspace_min_smoke_0625.yaml
  Tiny local smoke test.

config/workspace_all_sessions_smoke_0625.yaml
  Smoke test that samples every session but only a few windows per video.

config/server_room1_smoke.yaml
config/server_room2_smoke.yaml
  Server smoke configs for each room.

config/server_room1_full.yaml
config/server_room2_full.yaml
  Full room-level server configs.

config/server_room1_group1_full.yaml
config/server_room1_group2_full.yaml
config/server_room1_group3_full.yaml
  Optional Room 1 subgroups for splitting a large Room 1 run.
```

Most full configs use:

```yaml
windowing:
  window_seconds: 30
  stride_seconds: 10

video_features:
  resize_width: 640
  max_windows: null
  max_media: null

audio_features:
  sample_rate: 22050
  frame_seconds: 1.0
  analysis_frame_seconds: 0.1
  entropy_bands: 10
  regime_score_smoothing_windows: 12

output:
  write_partitioned_outputs: true
  timestamp_output_root: true
```

With `timestamp_output_root: true`, each run writes to a new output folder such as:

```text
poultry_data_preparation/outputs_full_room1_20260625_134103/
```

The explicitly configured `cache_root` is not timestamped, so reruns can reuse video/audio caches.

## Semantic Zone References

Automatic zone detection expects a clean reference image and an annotated reference image per room:

```text
room1_reference.png
room1_reference_with_notes_polygon.png
room2_reference.png
room2_reference_with_notes_polygon.png
```

The detector supports:

- non-rectangular polygons;
- multiple disconnected polygons for the same semantic zone;
- small gaps between zones.

The gaps are left unassigned and are ignored in zone-specific activity fractions.

The pipeline writes overlay images so the detected polygons can be inspected:

```text
outputs_<timestamp>/zones/semantic_zone_overlay_room_1.png
outputs_<timestamp>/zones/semantic_zone_overlay_room_2.png
```

## Output Structure

Each run creates one timestamped output root:

```text
poultry_data_preparation/outputs_<run_name>_<timestamp>/
```

Inside it:

```text
metadata/
zones/
features/
reports/
handoff_for_mvp/
Room 1/
Room 2/
```

### `metadata/`

```text
media_manifest.csv
video_window_index.csv
```

`media_manifest.csv` is one row per indexed media file. It includes:

- `media_id`
- `room_id`
- `session_id`
- `file_name`
- `relative_path`
- `absolute_path`
- `video_path`
- `start_time`
- `end_time`
- `duration_seconds`
- codec/audio metadata
- quality status and warnings

`video_window_index.csv` is one row per analysis window. It includes:

- `window_id`
- `media_id`
- `room_id`
- `session_id`
- `start_time`
- `end_time`
- `duration_seconds`
- `video_start_offset_sec`
- `has_audio`
- quality status and warnings

### `zones/`

```text
semantic_zone_configs.json
semantic_zone_overlay_room_1.png
semantic_zone_overlay_room_2.png
```

`semantic_zone_configs.json` stores the detected semantic-zone polygons. The overlay PNGs visualize these polygons over the room reference images.

### `features/`

```text
semantic_zone_video_features.csv
semantic_biomarker_window_table.csv
audio_window_features.csv
env_daily.csv
processed_multimodal_window_table.csv
```

`semantic_zone_video_features.csv` is one row per `window_id` and semantic zone. It includes:

- zone id and semantic type;
- zone area in pixels and fraction of reference image;
- activity mean/std/sum;
- number of sampled frame intervals used;
- quality status and warnings.

`semantic_biomarker_window_table.csv` is one row per window. It summarizes zone activity into window-level semantic features:

- `activity_mean`
- `normalized_activity`
- `mobility_index`
- `spatial_freedom_index`
- `occupancy_imbalance_index`
- `semantic_transition_proxy`
- `drinking_activity_fraction`
- `feeding_activity_fraction`
- `open_movement_activity_fraction`
- `resting_activity_fraction`
- `general_activity_fraction`
- `feeding_plus_drinking_activity_fraction`
- `drinking_to_feeding_activity_ratio`

`audio_window_features.csv` is one row per window. It includes:

- basic embedded-audio features: RMS, short-time energy, zero-crossing rate, centroid, bandwidth, rolloff, flatness;
- extended acoustic features: entropy, band-energy ratios, spectral flux, dominant frequency;
- call/chirp/cluck-like event rates, occupancy, and durations;
- bird/nonbird/disturbance/impact/low-frequency event summaries;
- dense/background/sparse regime scores, smoothed scores, hard label, and margin;
- audio quality status and warnings.

`env_daily.csv` is one row per room/date with daily environmental context:

- temperature summaries;
- relative humidity summaries;
- daily range;
- environment quality status.

`processed_multimodal_window_table.csv` is the main merged table. It is one row per `window_id`, combining:

- window metadata;
- semantic video biomarkers;
- audio features;
- environment context;
- final merged quality status and warnings.

### `reports/`

Reports include Markdown summaries and failure details:

```text
media_index_report.md
window_index_report.md
semantic_zone_report.md
video_feature_report.md
audio_feature_report.md
processed_multimodal_handoff_report.md
failed_windows.csv
```

`failed_windows.csv` contains windows that failed in one or more processing stages. A failed window does not necessarily mean the whole run failed.

### `handoff_for_mvp/`

This folder contains copies of the main processed tables and a short README for downstream projects.

Typical files:

```text
README_HANDOFF.md
media_manifest.csv
video_window_index.csv
semantic_zone_video_features.csv
semantic_biomarker_window_table.csv
audio_window_features.csv
env_daily.csv
processed_multimodal_window_table.csv
semantic_zone_configs.json
```

### Room/session Partitioned Outputs

When `write_partitioned_outputs: true`, the pipeline also writes smaller per-session copies:

```text
outputs_<timestamp>/
  Room 1/
    Room 1 (10, 11, 12, 13 Aug)/
      media_manifest.csv
      video_window_index.csv
      processed_multimodal_window_table.csv
      semantic_biomarker_window_table.csv
      audio_window_features.csv
      semantic_zone_video_features.csv
      env_daily.csv
      failed_windows.csv
      README_SESSION.md
```

These folders mirror the raw video folder structure under `data/raw/video/<Room>/<session>/`. They are intended for easier manual inspection and targeted downstream analysis.

## Caches And Recompute

Video and audio stages cache media-level intermediate features under the configured `cache_root`:

```text
outputs_full_room1/cache/
outputs_full_room2/cache/
```

Reruns reuse caches by default. To force recompute:

```bash
python -m poultry_data_preparation.src.main --config CONFIG --stage all --force-recompute-video
python -m poultry_data_preparation.src.main --config CONFIG --stage all --force-recompute-audio
python -m poultry_data_preparation.src.main --config CONFIG --stage all --force-recompute-all
```

With `run_room_jobs.sh`:

```bash
bash run_room_jobs.sh full --force-recompute-audio
bash run_room_jobs.sh full --force-recompute-all
```

## Notes

- Embedded MP4 audio is room/camera audio, not isolated chicken vocalization.
- Daily environment data are contextual covariates, not minute-level causal measurements.
- `media_id` and `window_id` are deterministic within a run snapshot but should not be treated as permanent IDs across different input subsets.
