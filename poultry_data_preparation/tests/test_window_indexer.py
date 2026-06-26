from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.window_indexer import build_window_rows_for_media


def test_build_window_rows_for_media() -> None:
    media_row = pd.Series(
        {
            "media_id": "media_000001",
            "room_id": "room_1",
            "session_id": "room_1_16_17_aug",
            "start_time": "2025-08-16T06:00:00-03:00",
            "duration_seconds": 65.0,
            "video_path": "data/raw/video/Room 1/test.mp4",
            "has_audio": True,
            "quality_status": "ok",
            "warnings": "",
        }
    )
    rows = build_window_rows_for_media(media_row, window_seconds=30, stride_seconds=10)
    assert len(rows) == 4
    assert rows[0]["window_id"] == "media_000001_w0001"
    assert rows[1]["video_start_offset_sec"] == 10.0
    assert rows[-1]["video_start_offset_sec"] == 30.0
