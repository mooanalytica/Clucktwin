from __future__ import annotations

from pathlib import Path
import sys
import uuid

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.semantic_zone_loader import _validate_zone_config, build_zone_config_from_images, discover_zone_reference_pairs


def test_build_zone_config_from_synthetic_images() -> None:
    scratch_root = Path("tmp_tests")
    scratch_root.mkdir(parents=True, exist_ok=True)
    tmp_path = scratch_root / f"zone_test_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    width, height = 200, 120
    reference = Image.new("RGB", (width, height), color=(120, 120, 120))
    annotated = reference.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle([20, 20, 70, 60], outline=(237, 28, 36), width=3)
    draw.rectangle([90, 30, 150, 90], outline=(34, 177, 76), width=3)
    draw.rectangle([30, 75, 85, 110], outline=(255, 127, 39), width=3)
    draw.rectangle([115, 10, 170, 55], outline=(0, 162, 232), width=3)
    draw.rectangle([120, 65, 185, 110], outline=(0, 162, 232), width=3)

    reference_path = tmp_path / "room1_test_reference.png"
    annotated_path = tmp_path / "room1_test_reference_with_notes.png"
    reference.save(reference_path)
    annotated.save(annotated_path)

    zone_config = build_zone_config_from_images("room_1", reference_path, annotated_path)
    assert zone_config["room_id"] == "room_1"
    assert zone_config["zone_config_id"] == "room_1_semantic_v2"
    zone_ids = [zone["zone_id"] for zone in zone_config["zones"]]
    assert zone_ids == ["drinking_zone", "feeding_zone", "open_movement_zone", "resting_zone"]
    assert zone_config["zones"][0].get("polygon") or zone_config["zones"][0].get("polygons")
    assert zone_config["zones"][1].get("polygon") or zone_config["zones"][1].get("polygons")
    assert zone_config["zones"][2].get("polygon") or zone_config["zones"][2].get("polygons")
    assert len(zone_config["zones"][3]["polygons"]) == 2


def test_build_zone_config_splits_touching_same_color_boxes() -> None:
    scratch_root = Path("tmp_tests")
    scratch_root.mkdir(parents=True, exist_ok=True)
    tmp_path = scratch_root / f"zone_test_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    width, height = 340, 220
    reference = Image.new("RGB", (width, height), color=(120, 120, 120))
    annotated = reference.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle([15, 70, 70, 145], outline=(237, 28, 36), width=4)
    draw.rectangle([85, 70, 150, 145], outline=(34, 177, 76), width=4)
    draw.rectangle([15, 10, 90, 55], outline=(255, 127, 39), width=4)
    draw.rectangle([90, 10, 210, 65], outline=(255, 127, 39), width=4)
    draw.rectangle([160, 65, 210, 145], outline=(255, 127, 39), width=4)
    draw.rectangle([215, 10, 310, 145], outline=(0, 162, 232), width=4)
    draw.rectangle([15, 155, 210, 205], outline=(0, 162, 232), width=4)

    reference_path = tmp_path / "room1_touching_reference.png"
    annotated_path = tmp_path / "room1_touching_reference_with_notes.png"
    reference.save(reference_path)
    annotated.save(annotated_path)

    zone_config = build_zone_config_from_images("room_1", reference_path, annotated_path)
    validation = _validate_zone_config(zone_config)
    open_zone = next(zone for zone in zone_config["zones"] if zone["zone_id"] == "open_movement_zone")
    resting_zone = next(zone for zone in zone_config["zones"] if zone["zone_id"] == "resting_zone")
    assert validation["valid"] is True
    assert len(open_zone["polygons"]) == 3
    assert len(resting_zone["polygons"]) == 2


def test_build_zone_config_accepts_closed_non_rectangular_polygons() -> None:
    scratch_root = Path("tmp_tests")
    scratch_root.mkdir(parents=True, exist_ok=True)
    tmp_path = scratch_root / f"zone_test_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    width, height = 320, 220
    reference = Image.new("RGB", (width, height), color=(120, 120, 120))
    annotated = reference.copy()
    draw = ImageDraw.Draw(annotated)
    draw.polygon([(20, 20), (85, 35), (65, 105), (10, 85)], outline=(237, 28, 36), width=4)
    draw.polygon([(105, 20), (175, 20), (190, 85), (145, 125), (95, 80)], outline=(34, 177, 76), width=4)
    draw.polygon([(210, 15), (300, 40), (285, 120), (220, 110)], outline=(255, 127, 39), width=4)
    draw.polygon([(35, 135), (140, 135), (165, 195), (20, 205)], outline=(0, 162, 232), width=4)

    reference_path = tmp_path / "room1_polygon_reference.png"
    annotated_path = tmp_path / "room1_polygon_reference_with_notes.png"
    reference.save(reference_path)
    annotated.save(annotated_path)

    zone_config = build_zone_config_from_images("room_1", reference_path, annotated_path)
    validation = _validate_zone_config(zone_config)
    drinking_zone = next(zone for zone in zone_config["zones"] if zone["zone_id"] == "drinking_zone")
    assert validation["valid"] is True
    x_values = {point[0] for point in drinking_zone["polygon"]}
    y_values = {point[1] for point in drinking_zone["polygon"]}
    assert len(x_values) > 2 or len(y_values) > 2


def test_build_zone_config_uses_general_zone_when_only_red_green_are_annotated() -> None:
    scratch_root = Path("tmp_tests")
    scratch_root.mkdir(parents=True, exist_ok=True)
    tmp_path = scratch_root / f"zone_test_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    width, height = 220, 140
    reference = Image.new("RGB", (width, height), color=(120, 120, 120))
    annotated = reference.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle([20, 20, 75, 80], outline=(237, 28, 36), width=4)
    draw.rectangle([100, 30, 170, 100], outline=(34, 177, 76), width=4)

    reference_path = tmp_path / "room2_reference.png"
    annotated_path = tmp_path / "room2_reference_with_notes.png"
    reference.save(reference_path)
    annotated.save(annotated_path)

    zone_config = build_zone_config_from_images("room_2", reference_path, annotated_path)
    validation = _validate_zone_config(zone_config)
    assert zone_config["zone_config_id"] == "room_2_semantic_v1"
    assert [zone["zone_id"] for zone in zone_config["zones"]] == ["drinking_zone", "feeding_zone", "general_zone"]
    assert validation["valid"] is True


def test_discover_zone_reference_pairs_prefers_polygon_annotations() -> None:
    scratch_root = Path("tmp_tests")
    scratch_root.mkdir(parents=True, exist_ok=True)
    tmp_path = scratch_root / f"zone_test_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)

    reference_path = tmp_path / "room1_reference.png"
    old_annotated_path = tmp_path / "room1_reference_with_notes.png"
    polygon_annotated_path = tmp_path / "room1_reference_with_notes_polygon.png"
    for path in (reference_path, old_annotated_path, polygon_annotated_path):
        path.write_bytes(b"placeholder")

    pairs = discover_zone_reference_pairs(tmp_path)

    assert pairs["room_1"]["reference_image"] == reference_path
    assert pairs["room_1"]["annotated_image"] == polygon_annotated_path


def test_discover_zone_reference_pairs_supports_session_reference_folders() -> None:
    scratch_root = Path("tmp_tests")
    scratch_root.mkdir(parents=True, exist_ok=True)
    tmp_path = scratch_root / f"zone_test_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)

    room_reference_path = tmp_path / "room1_reference.png"
    room_annotated_path = tmp_path / "room1_reference_with_notes_polygon.png"
    for path in (room_reference_path, room_annotated_path):
        path.write_bytes(b"placeholder")

    session_dir = tmp_path / "by_session" / "Room 1" / "Room 1 (16, 17 Aug)"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_reference_path = session_dir / "reference.png"
    session_annotated_path = session_dir / "reference_with_notes_polygon.png"
    for path in (session_reference_path, session_annotated_path):
        path.write_bytes(b"placeholder")

    pairs = discover_zone_reference_pairs(tmp_path)
    session_pair = pairs["room_1::room_1_16_17_aug"]

    assert pairs["room_1"]["reference_scope"] == "room"
    assert pairs["room_1"]["annotated_image"] == room_annotated_path
    assert session_pair["reference_scope"] == "session"
    assert session_pair["room_id"] == "room_1"
    assert session_pair["session_id"] == "room_1_16_17_aug"
    assert session_pair["session_folder"] == "Room 1 (16, 17 Aug)"
    assert session_pair["reference_image"] == session_reference_path
    assert session_pair["annotated_image"] == session_annotated_path
