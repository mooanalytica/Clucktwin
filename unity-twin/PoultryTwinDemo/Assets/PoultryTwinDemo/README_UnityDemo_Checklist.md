# Unity Demo Checklist

This checklist is for a source-only workflow. It does not require launching Unity.

## Before Manual Unity Testing

1. Confirm the analytics pipeline has generated:
   `Assets/StreamingAssets/poultry_twin_demo_timeline.json`
2. Run:
   `python May_version/PoultryTwinDemo/validate_demo_files.py`
3. Confirm the validator reports:
   - JSON exists
   - timeline length is greater than `0`
   - frames include `metrics`, `state`, `welfare`, and `zones`
   - runtime scripts exist
   - no runtime script references `TextMeshPro`, `Newtonsoft`, or `UnityEditor`

## Expected Runtime Behavior After Pressing Play

- `PoultryTwinDemoBootstrap` creates the runtime controller in any simple scene.
- If no camera exists, it creates a top-down orthographic main camera.
- If no light exists, it creates a directional light.
- `ZoneOverlayController` creates the `Room 1` floorplan, room objects, and clickable semantic zone overlays at runtime.
- `PoultryTwinJsonLoader` loads:
  `Application.streamingAssetsPath/poultry_twin_demo_timeline.json`
- `PoultryTwinPlaybackController` starts at frame `0`, can play/pause, step, loop, and reset.
- `DemoHudController` displays dashboard cards, selected-zone info, abnormality text, and a draggable timeline with `OnGUI`.

## Controls To Verify Manually In Unity

- `Space`: play / pause
- `Left`: previous frame
- `Right`: next frame
- `Up`: faster
- `Down`: slower
- `R`: reset to frame `0`
- `Tab`: toggle debug camera mode
- `F`: return to the presentation camera

## Failure Modes The HUD Should Make Visible

- JSON missing
- JSON parse failed
- timeline empty
- no zones found
- missing required metrics
