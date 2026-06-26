from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    demo_root = Path(__file__).resolve().parent
    workspace_root = demo_root.parent.parent
    streaming_json = demo_root / "Assets" / "StreamingAssets" / "poultry_twin_demo_timeline.json"
    fallback_json = workspace_root / "mvp_biomarker_state_twin" / "outputs" / "unity_json" / "poultry_twin_demo_timeline.json"
    scripts_dir = demo_root / "Assets" / "PoultryTwinDemo" / "Scripts"
    editor_dir = demo_root / "Assets" / "PoultryTwinDemo" / "Editor"

    expected_scripts = [
        scripts_dir / "PoultryTwinDataModels.cs",
        scripts_dir / "PoultryTwinJsonLoader.cs",
        scripts_dir / "PoultryTwinPlaybackController.cs",
        scripts_dir / "ZoneOverlayController.cs",
        scripts_dir / "DemoHudController.cs",
        scripts_dir / "PoultryTwinDemoBootstrap.cs",
        scripts_dir / "PoultryTwinCameraController.cs",
        scripts_dir / "PoultryTwinRoomLayout.cs",
        editor_dir / "CreatePoultryTwinDemoScene.cs",
    ]

    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    if not streaming_json.exists():
        if fallback_json.exists():
            streaming_json.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fallback_json, streaming_json)
            notes.append(f"Copied missing StreamingAssets JSON from fallback source: {fallback_json}")
        else:
            errors.append(f"StreamingAssets JSON missing: {streaming_json}")

    payload: dict | None = None
    if streaming_json.exists():
        try:
            payload = json.loads(streaming_json.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"JSON parse failed for {streaming_json}: {exc}")

    if payload is not None:
        validate_json_payload(payload, errors, warnings)

    for script_path in expected_scripts:
        if not script_path.exists():
            errors.append(f"Expected script missing: {script_path}")

    runtime_scripts = sorted(scripts_dir.glob("*.cs"))
    forbidden_runtime_tokens = {
        "TextMeshPro": "Runtime scripts must not depend on TextMeshPro.",
        "TMPro": "Runtime scripts must not depend on TextMeshPro.",
        "Newtonsoft": "Runtime scripts must not depend on Newtonsoft.Json unless it is explicitly installed.",
        "using UnityEditor": "Runtime scripts must not reference UnityEditor namespaces.",
        "UnityEditor.": "Runtime scripts must not reference UnityEditor namespaces.",
    }

    for script_path in runtime_scripts:
        script_text = script_path.read_text(encoding="utf-8")
        for token, message in forbidden_runtime_tokens.items():
            if token in script_text:
                errors.append(f"{script_path}: {message}")

    if not editor_dir.exists():
        warnings.append(f"Optional editor folder missing: {editor_dir}")

    print("Unity demo filesystem validation")
    print(f"Project root: {demo_root}")
    print(f"StreamingAssets JSON: {streaming_json}")

    for note in notes:
        print(f"[INFO] {note}")

    for warning in warnings:
        print(f"[WARN] {warning}")

    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        print("Validation failed.")
        return 1

    timeline_count = len(payload.get("timeline", [])) if payload is not None else 0
    print(f"[OK] JSON timeline frames: {timeline_count}")
    print(f"[OK] Runtime scripts found: {len(runtime_scripts)}")
    print("Validation passed.")
    return 0


def validate_json_payload(payload: dict, errors: list[str], warnings: list[str]) -> None:
    timeline = payload.get("timeline")
    if not isinstance(timeline, list):
        errors.append("JSON root is missing a timeline list.")
        return

    if len(timeline) == 0:
        errors.append("Timeline is empty.")
        return

    rooms = payload.get("rooms")
    if rooms is None:
        warnings.append("JSON root is missing rooms. Runtime will rely on fallback zone layout logic.")
    elif not isinstance(rooms, list):
        errors.append("JSON root field 'rooms' is not a list.")

    first_frame = timeline[0]
    if not isinstance(first_frame, dict):
        errors.append("First timeline entry is not an object.")
        return

    required_frame_sections = ["metrics", "state", "welfare", "zones"]
    for section in required_frame_sections:
        if section not in first_frame:
            errors.append(f"First timeline frame is missing '{section}'.")

    metrics = first_frame.get("metrics")
    if isinstance(metrics, dict):
        for key in ["mobility_index", "spatial_freedom_index", "occupancy_imbalance_index", "activity_mean"]:
            if key not in metrics:
                errors.append(f"First timeline frame metrics are missing '{key}'.")
    else:
        errors.append("First timeline frame metrics are missing or malformed.")

    state = first_frame.get("state")
    if isinstance(state, dict):
        for key in ["state_id", "state_label"]:
            if key not in state:
                errors.append(f"First timeline frame state is missing '{key}'.")
        if "state_probability" not in state:
            warnings.append("First timeline frame state is missing 'state_probability'. Runtime will display the default value.")
    else:
        errors.append("First timeline frame state is missing or malformed.")

    welfare = first_frame.get("welfare")
    if isinstance(welfare, dict):
        for key in ["risk_score", "risk_level", "sustained_risk_flag"]:
            if key not in welfare:
                errors.append(f"First timeline frame welfare is missing '{key}'.")
    else:
        errors.append("First timeline frame welfare is missing or malformed.")

    zones = first_frame.get("zones")
    if not isinstance(zones, list):
        errors.append("First timeline frame zones are missing or malformed.")
    elif len(zones) == 0:
        errors.append("First timeline frame has no zone overlays.")
    else:
        first_zone = zones[0]
        if not isinstance(first_zone, dict):
            errors.append("First zone entry is not an object.")
        else:
            for key in ["zone_id", "activity", "activity_norm", "overlay_intensity"]:
                if key not in first_zone:
                    errors.append(f"First zone entry is missing '{key}'.")


if __name__ == "__main__":
    raise SystemExit(main())
