from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from PIL import Image

from .config import PreparationConfig
from .room_parser import parse_room_identifier_from_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # pragma: no cover - depends on local environment
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticZoneLoadResult:
    zone_configs: list[dict]
    combined_output_path: Path
    report_path: Path
    valid_rooms: set[str]


@dataclass(frozen=True)
class ZoneColorSpec:
    zone_id: str
    display_name: str
    semantic_type: str
    color_name: str
    target_rgb: tuple[int, int, int]
    overlay_hex: str


ZONE_COLOR_SPECS = [
    ZoneColorSpec("drinking_zone", "Drinking Area", "drinking", "red", (237, 28, 36), "#ed1c24"),
    ZoneColorSpec("feeding_zone", "Feeding Area", "feeding", "green", (34, 177, 76), "#22b14c"),
    ZoneColorSpec("open_movement_zone", "Open Movement", "open_movement", "orange", (255, 127, 39), "#ff7f27"),
    ZoneColorSpec("resting_zone", "Resting Area", "resting", "blue", (0, 162, 232), "#00a2e8"),
]

ANNOTATED_REFERENCE_SUFFIXES = [
    "_reference_with_notes_polygon.png",
    "_reference_with_notes.png",
]

REQUIRED_AUTOMATIC_ZONE_IDS = {"drinking_zone", "feeding_zone"}
EXTENDED_AUTOMATIC_ZONE_IDS = {"open_movement_zone", "resting_zone"}


def load_semantic_zone_configs(config: PreparationConfig, dry_run: bool = False) -> SemanticZoneLoadResult:
    _ensure_manual_fallback_template(config)
    zone_pairs = discover_zone_reference_pairs(config.semantic_zone_ref_dir)
    manual_fallbacks = _load_manual_fallbacks(config)
    zone_configs: list[dict] = []
    warnings: list[str] = []

    for room_id, pair in zone_pairs.items():
        if not _room_is_included(room_id, config):
            continue

        LOGGER.info("Preparing semantic zones for %s", room_id)
        built_config = None
        try:
            built_config = build_zone_config_from_images(
                room_id=room_id,
                reference_image=pair["reference_image"],
                annotated_image=pair["annotated_image"],
            )
        except Exception as exc:
            warnings.append(f"{room_id}:automatic_detection_failed:{exc}")

        if built_config is not None and not _validate_zone_config(built_config)["valid"]:
            warnings.append(f"{room_id}:automatic_detection_invalid")
            built_config = None

        if built_config is None and config.semantic_allow_manual_fallback:
            manual_config = manual_fallbacks.get(room_id)
            if manual_config is not None:
                built_config = manual_config
                built_config["detection_mode"] = "manual_fallback"
                warnings.append(f"{room_id}:manual_fallback_used")

        if built_config is None:
            warnings.append(f"{room_id}:no_valid_zone_config")
            continue

        zone_configs.append(built_config)
        _write_zone_outputs(config, built_config)

    combined_output_path = config.zones_output_dir / "semantic_zone_configs.json"
    combined_output_path.write_text(json.dumps(zone_configs, indent=2), encoding="utf-8")
    report_path = config.reports_output_dir / "semantic_zone_report.md"
    report_path.write_text(_build_zone_report(zone_configs, warnings, dry_run), encoding="utf-8")
    return SemanticZoneLoadResult(
        zone_configs=zone_configs,
        combined_output_path=combined_output_path,
        report_path=report_path,
        valid_rooms={item["room_id"] for item in zone_configs},
    )


def discover_zone_reference_pairs(reference_dir: Path) -> dict[str, dict[str, Path]]:
    pairs: dict[str, dict[str, Path]] = {}
    for suffix in ANNOTATED_REFERENCE_SUFFIXES:
        for annotated_path in sorted(reference_dir.glob(f"*{suffix}"), key=lambda path: path.as_posix().lower()):
            base_name = annotated_path.name[: -len(suffix)]
            reference_path = annotated_path.with_name(base_name + "_reference.png")
            if not reference_path.exists():
                continue
            room_result = parse_room_identifier_from_path(annotated_path.name)
            pairs.setdefault(
                room_result.room_id,
                {
                    "reference_image": reference_path,
                    "annotated_image": annotated_path,
                },
            )
    return pairs


def build_zone_config_from_images(room_id: str, reference_image: Path, annotated_image: Path) -> dict:
    reference_rgb = np.asarray(Image.open(reference_image).convert("RGB"))
    annotated_rgb = np.asarray(Image.open(annotated_image).convert("RGB"))
    if reference_rgb.shape != annotated_rgb.shape:
        raise ValueError("Reference and annotated images must have the same dimensions.")

    image_height, image_width = reference_rgb.shape[:2]
    detected_polygons: dict[str, list[list[list[int]]]] = {}
    for spec in ZONE_COLOR_SPECS:
        polygons = _detect_colored_polygons(reference_rgb, annotated_rgb, spec.target_rgb)
        if polygons:
            detected_polygons[spec.zone_id] = polygons

    missing_required = REQUIRED_AUTOMATIC_ZONE_IDS - set(detected_polygons)
    if missing_required:
        missing_text = ", ".join(sorted(missing_required))
        raise ValueError(f"Missing required automatic semantic-zone annotations: {missing_text}.")

    detected_extended = EXTENDED_AUTOMATIC_ZONE_IDS & set(detected_polygons)
    missing_extended = EXTENDED_AUTOMATIC_ZONE_IDS - set(detected_polygons)
    if detected_extended and missing_extended:
        missing_text = ", ".join(sorted(missing_extended))
        raise ValueError(f"Partial v2 semantic-zone annotations detected; missing: {missing_text}.")

    zones = []
    for spec in ZONE_COLOR_SPECS:
        polygons = detected_polygons.get(spec.zone_id)
        if polygons:
            zones.append(_polygon_zone(spec, polygons))

    config_version = "v2"
    if not detected_extended:
        config_version = "v1"
        zones.append(
            {
                "zone_id": "general_zone",
                "display_name": "General Area",
                "semantic_type": "general",
                "definition": "full_frame_minus_drinking_and_feeding",
            }
        )

    return {
        "room_id": room_id,
        "zone_config_id": f"{room_id}_semantic_{config_version}",
        "reference_image": str(reference_image),
        "annotated_image": str(annotated_image),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "detection_mode": "automatic",
        "zones": zones,
    }


def build_zone_masks(zone_config: dict, image_width: int, image_height: int) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    occupied = np.zeros((image_height, image_width), dtype=bool)

    for zone in zone_config.get("zones", []):
        polygons = get_zone_polygons(zone)
        if not polygons:
            continue

        zone_mask = np.zeros((image_height, image_width), dtype=bool)
        for polygon in polygons:
            zone_mask |= polygon_to_mask(np.asarray(polygon, dtype=float), image_width, image_height)
        masks[str(zone["zone_id"])] = zone_mask
        occupied |= zone_mask

    for zone in zone_config.get("zones", []):
        definition = str(zone.get("definition", "")).strip().lower()
        zone_id = str(zone.get("zone_id", "")).strip()
        if definition in {"full_frame_minus_drinking_and_feeding", "full_frame_minus_other_zones"} and zone_id:
            masks[zone_id] = ~occupied

    return masks


def get_zone_polygons(zone: dict) -> list[list[list[float]]]:
    polygons = zone.get("polygons")
    if isinstance(polygons, list) and polygons:
        return [polygon for polygon in polygons if isinstance(polygon, list) and polygon]

    polygon = zone.get("polygon")
    if isinstance(polygon, list) and polygon:
        return [polygon]

    return []


def polygon_to_mask(polygon: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    if cv2 is None:
        x_values = polygon[:, 0]
        y_values = polygon[:, 1]
        x_min = max(0, int(np.floor(x_values.min())))
        x_max = min(image_width, int(np.ceil(x_values.max())))
        y_min = max(0, int(np.floor(y_values.min())))
        y_max = min(image_height, int(np.ceil(y_values.max())))
        mask = np.zeros((image_height, image_width), dtype=bool)
        mask[y_min:y_max, x_min:x_max] = True
        return mask

    points = np.round(polygon).astype(np.int32)
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask.astype(bool)


def scale_polygon(polygon, source_width: int, source_height: int, target_width: int, target_height: int) -> np.ndarray:
    polygon_array = np.asarray(polygon, dtype=float)
    scaled = polygon_array.copy()
    scaled[:, 0] *= float(target_width) / float(source_width)
    scaled[:, 1] *= float(target_height) / float(source_height)
    return scaled


def _ensure_manual_fallback_template(config: PreparationConfig) -> None:
    manual_path = config.config_dir / "manual_semantic_zones.yaml"
    if manual_path.exists():
        return
    template_path = Path(__file__).resolve().parents[1] / "config" / "manual_semantic_zones.yaml"
    manual_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")


def _load_manual_fallbacks(config: PreparationConfig) -> dict[str, dict]:
    manual_path = config.config_dir / "manual_semantic_zones.yaml"
    if not manual_path.exists():
        return {}
    with manual_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return {}
    fallbacks: dict[str, dict] = {}
    for room_id, item in payload.items():
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        normalized.setdefault("room_id", str(room_id))
        fallbacks[str(room_id)] = normalized
    return fallbacks


def _room_is_included(room_id: str, config: PreparationConfig) -> bool:
    include = config.rooms_include
    if include == "auto":
        return room_id not in set(config.rooms_exclude)
    if isinstance(include, str):
        return room_id == include and room_id not in set(config.rooms_exclude)
    return room_id in {str(item) for item in include} and room_id not in set(config.rooms_exclude)


def _polygon_zone(spec: ZoneColorSpec, polygons: list[list[list[int]]]) -> dict:
    zone = {
        "zone_id": spec.zone_id,
        "display_name": spec.display_name,
        "semantic_type": spec.semantic_type,
    }
    if len(polygons) == 1:
        zone["polygon"] = polygons[0]
    else:
        zone["polygons"] = polygons
    return zone


def _detect_colored_polygons(
    reference_rgb: np.ndarray,
    annotated_rgb: np.ndarray,
    target_rgb: tuple[int, int, int],
) -> list[list[list[int]]]:
    mask_uint8 = _build_color_mask(reference_rgb, annotated_rgb, target_rgb)
    if cv2 is None:
        return [_box_to_polygon(_shrink_box(box, margin=2)) for box in _detect_component_boxes(mask_uint8)]

    polygons = _detect_enclosed_polygons(mask_uint8)
    if polygons:
        return polygons
    return [_box_to_polygon(_shrink_box(box, margin=2)) for box in _detect_component_boxes(mask_uint8)]


def _build_color_mask(
    reference_rgb: np.ndarray,
    annotated_rgb: np.ndarray,
    target_rgb: tuple[int, int, int],
) -> np.ndarray:
    diff = np.abs(annotated_rgb.astype(np.int16) - reference_rgb.astype(np.int16)).sum(axis=2)
    target = np.asarray(target_rgb, dtype=np.int16).reshape(1, 1, 3)
    delta = np.abs(annotated_rgb.astype(np.int16) - target)
    mask = (diff > 20) & (delta[:, :, 0] <= 25) & (delta[:, :, 1] <= 25) & (delta[:, :, 2] <= 25)
    return mask.astype(np.uint8) * 255


def _detect_enclosed_polygons(mask_uint8: np.ndarray) -> list[list[list[int]]]:
    if cv2 is None or not np.any(mask_uint8):
        return []

    kernel = np.ones((3, 3), dtype=np.uint8)
    outline = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
    outline = cv2.dilate(outline, kernel, iterations=1)

    height, width = outline.shape
    inverse = np.where(outline > 0, 0, 255).astype(np.uint8)
    padded = np.pad(inverse, pad_width=1, mode="constant", constant_values=255)
    flood_mask = np.zeros((height + 4, width + 4), dtype=np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 127)
    enclosed = (padded[1:-1, 1:-1] == 255).astype(np.uint8) * 255

    return _polygons_from_filled_mask(enclosed)


def _polygons_from_filled_mask(mask_uint8: np.ndarray) -> list[list[list[int]]]:
    if cv2 is None or not np.any(mask_uint8):
        return []

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    polygons_with_boxes: list[tuple[tuple[int, int, int], list[list[int]]]] = []
    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        if area < 120 or width < 10 or height < 10:
            continue

        component_mask = np.where(labels == component_index, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = max(2.0, 0.01 * perimeter)
        approximated = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True).reshape(-1, 2)
        if len(approximated) < 3:
            x = int(stats[component_index, cv2.CC_STAT_LEFT])
            y = int(stats[component_index, cv2.CC_STAT_TOP])
            polygon = _box_to_polygon((x, y, width, height))
        else:
            polygon = [[int(x), int(y)] for x, y in approximated]
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        polygons_with_boxes.append(((y, x, width * height), polygon))

    return [polygon for _, polygon in sorted(polygons_with_boxes, key=lambda item: item[0])]


def _box_to_polygon(box: tuple[int, int, int, int]) -> list[list[int]]:
    x, y, width, height = box
    return [
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height],
    ]


def _detect_colored_boxes(
    reference_rgb: np.ndarray,
    annotated_rgb: np.ndarray,
    target_rgb: tuple[int, int, int],
) -> list[tuple[int, int, int, int]]:
    mask_uint8 = _build_color_mask(reference_rgb, annotated_rgb, target_rgb)
    return _detect_component_boxes(mask_uint8)


def _detect_component_boxes(mask_uint8: np.ndarray) -> list[tuple[int, int, int, int]]:
    if not np.any(mask_uint8):
        return []

    if cv2 is not None:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        boxes = []
        for component_index in range(1, component_count):
            area = int(stats[component_index, cv2.CC_STAT_AREA])
            width = int(stats[component_index, cv2.CC_STAT_WIDTH])
            height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
            if area < 120 or width < 10 or height < 10:
                continue
            boxes.append(
                (
                    int(stats[component_index, cv2.CC_STAT_LEFT]),
                    int(stats[component_index, cv2.CC_STAT_TOP]),
                    width,
                    height,
                )
            )
        return sorted(boxes, key=lambda item: (item[1], item[0], item[2] * item[3]))

    coordinates = np.argwhere(mask_uint8 > 0)
    y_values = coordinates[:, 0]
    x_values = coordinates[:, 1]
    return [
        (
            int(x_values.min()),
            int(y_values.min()),
            int(x_values.max() - x_values.min() + 1),
            int(y_values.max() - y_values.min() + 1),
        )
    ]


def _shrink_box(box: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    x, y, width, height = box
    return (x + margin, y + margin, max(1, width - (2 * margin)), max(1, height - (2 * margin)))


def _validate_zone_config(zone_config: dict) -> dict[str, object]:
    masks = build_zone_masks(zone_config, int(zone_config["image_width"]), int(zone_config["image_height"]))
    configured_zone_ids = [str(zone.get("zone_id", "")) for zone in zone_config.get("zones", []) if str(zone.get("zone_id", ""))]
    missing_zone_ids = [zone_id for zone_id in configured_zone_ids if zone_id not in masks]
    missing_required_zone_ids = sorted(REQUIRED_AUTOMATIC_ZONE_IDS - set(masks))
    empty_zone_ids = [zone_id for zone_id, mask in masks.items() if int(mask.sum()) <= 0]

    if masks:
        overlap_map = np.zeros(next(iter(masks.values())).shape, dtype=np.uint8)
        for mask in masks.values():
            overlap_map += mask.astype(np.uint8)
        overlap_pixels = int((overlap_map > 1).sum())
    else:
        overlap_pixels = 0

    valid = overlap_pixels == 0 and not missing_zone_ids and not missing_required_zone_ids and not empty_zone_ids
    return {
        "valid": valid,
        "overlap_pixels": overlap_pixels,
        "missing_zone_ids": missing_zone_ids,
        "missing_required_zone_ids": missing_required_zone_ids,
        "empty_zone_ids": empty_zone_ids,
        "zone_pixels": {zone_id: int(mask.sum()) for zone_id, mask in masks.items()},
    }


def _write_zone_outputs(config: PreparationConfig, zone_config: dict) -> None:
    room_id = zone_config["room_id"]
    output_path = config.zones_output_dir / f"semantic_zone_config_{room_id}.json"
    output_path.write_text(json.dumps(zone_config, indent=2), encoding="utf-8")

    reference_image = np.asarray(Image.open(zone_config["reference_image"]).convert("RGB"))
    overlay_path = config.zones_output_dir / f"semantic_zone_overlay_{room_id}.png"
    figure, axis = plt.subplots(figsize=(14, 8))
    axis.imshow(reference_image)
    axis.set_title(f"Semantic Zone Overlay: {room_id}")
    axis.set_axis_off()
    colors = {spec.zone_id: spec.overlay_hex for spec in ZONE_COLOR_SPECS}

    for zone in zone_config["zones"]:
        zone_polygons = get_zone_polygons(zone)
        color = colors.get(zone["zone_id"], "#444444")
        for polygon_index, polygon in enumerate(zone_polygons, start=1):
            polygon_array = np.asarray(polygon, dtype=float)
            closed = np.vstack([polygon_array, polygon_array[0]])
            axis.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2.2)
            label_position = _top_left_polygon_vertex(polygon_array)
            label_text = str(zone["display_name"])
            if len(zone_polygons) > 1:
                label_text = f"{label_text} {polygon_index}"
            axis.text(
                label_position[0] + 4,
                label_position[1] + 4,
                label_text,
                color=color,
                fontsize=10,
                ha="left",
                va="top",
                clip_on=True,
                bbox={"facecolor": "white", "alpha": 0.75},
            )

    figure.tight_layout()
    figure.savefig(overlay_path, dpi=200)
    plt.close(figure)


def _top_left_polygon_vertex(polygon: np.ndarray) -> tuple[float, float]:
    if polygon.size == 0:
        return 0.0, 0.0
    ordered = sorted(polygon.tolist(), key=lambda point: (float(point[1]), float(point[0])))
    return float(ordered[0][0]), float(ordered[0][1])


def _build_zone_report(zone_configs: list[dict], warnings: list[str], dry_run: bool) -> str:
    lines = [
        "# Semantic Zone Report",
        "",
        f"- Dry run: `{dry_run}`",
        f"- Rooms with valid semantic zones: {len(zone_configs)}",
        f"- Automatic v2 zones: {', '.join(spec.zone_id for spec in ZONE_COLOR_SPECS)}",
        "- Automatic v1 fallback: drinking_zone, feeding_zone, general_zone",
        "",
        "## Zone Configs",
        "",
    ]
    for zone_config in zone_configs:
        validation = _validate_zone_config(zone_config)
        lines.append(
            f"- `{zone_config['room_id']}` -> `{zone_config['zone_config_id']}` "
            f"(mode `{zone_config.get('detection_mode', 'unknown')}`, overlap `{validation['overlap_pixels']}` pixels, "
            f"missing `{validation['missing_zone_ids']}`, missing required `{validation['missing_required_zone_ids']}`, "
            f"empty `{validation['empty_zone_ids']}`)"
        )
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- `{warning}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Manual Fallback",
            "",
            "- If automatic detection fails for a room, define polygons in `poultry_data_preparation/config/manual_semantic_zones.yaml` and rerun the `zones` stage.",
        ]
    )
    return "\n".join(lines) + "\n"
