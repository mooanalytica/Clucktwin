from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


UNKNOWN_ROOM_ID = "unknown_room"


@dataclass(frozen=True)
class RoomParseResult:
    room_id: str
    display_name: str
    warning: str | None


ROOM_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9])room[\s_]*([0-9]+)(?![0-9])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])room([0-9]+)(?![0-9])", re.IGNORECASE),
]


def parse_room_identifier(text: str | Path | None) -> RoomParseResult:
    if text is None:
        return RoomParseResult(UNKNOWN_ROOM_ID, "Unknown Room", "room_id_not_found")

    raw_text = str(text)
    for pattern in ROOM_PATTERNS:
        match = pattern.search(raw_text.replace("-", " "))
        if match:
            room_number = int(match.group(1))
            return RoomParseResult(
                room_id=f"room_{room_number}",
                display_name=f"Room {room_number}",
                warning=None,
            )
    return RoomParseResult(UNKNOWN_ROOM_ID, "Unknown Room", "room_id_not_found")


def parse_room_identifier_from_path(path: str | Path) -> RoomParseResult:
    path_obj = Path(path)
    candidates = [path_obj.name] + [part for part in path_obj.parts[::-1]]
    for candidate in candidates:
        result = parse_room_identifier(candidate)
        if result.room_id != UNKNOWN_ROOM_ID:
            return result
    return RoomParseResult(UNKNOWN_ROOM_ID, "Unknown Room", "room_id_not_found")
