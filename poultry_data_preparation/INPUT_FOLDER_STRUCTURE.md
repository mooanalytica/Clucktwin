# Input Folder Structure

This file documents the raw input layout expected by `poultry_data_preparation` and shows the current sample subset that exists in this workspace.

## Generic Expected Layout

```text
data/
  raw/
    video/
      Room 1/
        Room 1 (16, 17 Aug)/
          GX010044.MP4
          GX020044.MP4
          ...
    env/
      Combined Room 1.xlsx
      Combined Room 2.xlsx

  metadata/
    semantic_zone_refs/
      room1_reference.png
      room1_reference_with_notes_polygon.png
      by_session/
        Room 1/
          Room 1 (16, 17 Aug)/
            reference.png
            reference_with_notes_polygon.png
```


## Required vs Optional Inputs

Required for the main preprocessing workflow:

- raw MP4 video files
- semantic-zone reference image files
- room-level environment workbook files

Optional or not part of the current main preprocessing output path:

- any future external behavior-detection CSVs
- any external audio files

Semantic-zone annotations can be non-rectangular polygons. One semantic zone may contain multiple disconnected polygons.

## Timestamp Notes

No separate timestamp CSV is required.

The preprocessing code resolves video timestamps from:

1. `exiftool` MP4 metadata when available
2. `ffprobe` creation-time metadata when available
3. file modification time as a fallback

Those resolved timestamps appear later in:

- `outputs/metadata/media_manifest.csv`
- `outputs/metadata/video_window_index.csv`
- `outputs/handoff_for_mvp/media_manifest.csv`
- `outputs/handoff_for_mvp/video_window_index.csv`

