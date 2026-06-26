from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_loader import normalize_environment_dataframe


def test_normalize_environment_dataframe() -> None:
    raw_df = pd.DataFrame(
        {
            "Date": [45984],
            "Temp_AM_Min": [20.0],
            "Temp_AM_Max": [24.0],
            "Temp_PM": [25.0],
            "RH_AM": [50.0],
            "RH_PM": [55.0],
        }
    )
    normalized_df = normalize_environment_dataframe(raw_df, room_id="room_1")
    assert list(normalized_df.columns[:2]) == ["room_id", "date"]
    assert normalized_df.loc[0, "room_id"] == "room_1"
    assert float(normalized_df.loc[0, "temp_am_mean"]) == 22.0
    assert float(normalized_df.loc[0, "temp_daily_mean"]) == 23.5
    assert float(normalized_df.loc[0, "rh_daily_mean"]) == 52.5
