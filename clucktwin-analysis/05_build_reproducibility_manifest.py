"""Build the compact reproducibility manifest and copy versioned polygon inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
AUDIT = ROOT / "analysis" / "audit"
POLYGONS = ROOT / "analysis" / "polygons"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_session_config(path: Path, session_id: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    configs = payload if isinstance(payload, list) else [payload]
    for config in configs:
        if str(config.get("session_id", "")) == session_id:
            return config
    raise RuntimeError(f"No zone config for {session_id} in {path}")


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    POLYGONS.mkdir(parents=True, exist_ok=True)
    inputs = [
        WORKSPACE / "experiments" / "room1_normalized_zone_quality_hmm_20260727" / "processed" / "normalized_multimodal_30s.parquet",
        WORKSPACE / "experiments" / "room1_normalized_zone_quality_hmm_20260727" / "audit" / "session_geometry_normalization.csv",
        WORKSPACE / "experiments" / "room1_normalized_zone_quality_hmm_20260727" / "tables" / "normalized_session_metrics.csv",
        WORKSPACE / "experiments" / "room1_growth_caretaker_dynamics_verified_fall_20260727" / "audit" / "media_inventory.csv",
        WORKSPACE / "experiments" / "room1_growth_caretaker_dynamics_verified_fall_20260727" / "tables" / "caretaker_events_analysis.csv",
        WORKSPACE / "experiments" / "physical_virtual_fidelity_room1_20260708" / "outputs_daytime_50_outline_final" / "clip_annotations.csv",
        WORKSPACE / "Caretaker_labels" / "Room1_caretaker_event_table.csv",
        WORKSPACE / "data" / "processed" / "Room1_Fall_caretaker_event_table.csv",
    ]
    rows = []
    for path in inputs:
        rows.append(
            {
                "role": "analysis_input",
                "path": str(path),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256(path) if path.is_file() else None,
            }
        )

    ledger = pd.read_csv(AUDIT / "session_reference_manifest.csv")
    ledger.to_csv(AUDIT / "session_ledger.csv", index=False)
    for item in ledger.itertuples(index=False):
        source = Path(item.zone_config)
        destination = POLYGONS / f"{item.session_id}.json"
        if source.is_file():
            session_config = load_session_config(source, item.session_id)
            destination.write_text(json.dumps(session_config, indent=2, sort_keys=True), encoding="utf-8")
        rows.append(
            {
                "role": "session_polygon_config",
                "session_id": item.session_id,
                "path": str(destination),
                "source_path": str(source),
                "exists": destination.is_file(),
                "bytes": destination.stat().st_size if destination.is_file() else None,
                "sha256": sha256(destination) if destination.is_file() else None,
                "source_container_sha256": sha256(source) if source.is_file() else None,
            }
        )
    pd.DataFrame(rows).to_csv(AUDIT / "input_data_manifest_sha256.csv", index=False)


if __name__ == "__main__":
    main()
