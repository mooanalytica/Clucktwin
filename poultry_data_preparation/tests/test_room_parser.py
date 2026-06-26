from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.room_parser import UNKNOWN_ROOM_ID, parse_room_identifier, parse_room_identifier_from_path


def test_parse_room_from_folder_name() -> None:
    assert parse_room_identifier("Room 1").room_id == "room_1"
    assert parse_room_identifier("Room 2").room_id == "room_2"
    assert parse_room_identifier("room_3").room_id == "room_3"


def test_parse_room_from_filename() -> None:
    assert parse_room_identifier("room1_aug16_17_reference.png").room_id == "room_1"
    assert parse_room_identifier("Combined Room 2.xlsx").room_id == "room_2"


def test_parse_room_from_path() -> None:
    result = parse_room_identifier_from_path("data/raw/video/Room 1/Room 1 (16, 17 Aug)/GX010044.MP4")
    assert result.room_id == "room_1"
    assert result.display_name == "Room 1"


def test_unknown_room_when_pattern_missing() -> None:
    result = parse_room_identifier("mystery_file.mp4")
    assert result.room_id == UNKNOWN_ROOM_ID
    assert result.warning == "room_id_not_found"
