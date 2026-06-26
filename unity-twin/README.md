# Poultry Twin Demo: Room 1 Unity Prototype

This repository contains a Unity visualization prototype for a poultry digital twin demo focused on **Room 1**. The Unity project is a front-end viewer: it plays back precomputed analytics from a JSON timeline and renders a simplified 3D room, semantic activity zones, chicken proxies/models, behavior animations, and a dashboard HUD.


## What This Demo Shows

- A 3D reconstruction of Room 1.
- Four semantic zones: feeder, drinker, resting/perching, and open movement.
- Timeline playback driven by precomputed 10-second frames.
- Zone-level activity, occupancy, behavior mix, and provisional state/risk summaries.
- Chicken visual proxies using JSON-embedded bird positions and behavior labels.
- Optional animated chicken models when the required Unity Asset Store package is imported locally.
- A semi-transparent dashboard HUD with current window, selected zone, metrics, abnormality summary, and timeline controls.

## Repository Layout

Expected Unity project structure:

```text
Assets/
  PoultryTwinDemo/
    Scripts/
    Editor/
    README_UnityDemo_Checklist.md
  Scenes/
  Settings/
  StreamingAssets/
    poultry_twin_demo_timeline.json
    room1_reference.png
    room1_zone_overlay.png
  InputSystem_Actions.inputactions
Packages/
ProjectSettings/
README.md
validate_demo_files.py
```

The runtime JSON file is:

```text
Assets/StreamingAssets/poultry_twin_demo_timeline.json
```

Unity reads this JSON at runtime. It does **not** directly read raw video files or behavior CSV files during playback.

## Unity Version

This project was developed with:

```text
Unity 6000.4.6f1
```

Opening the project with a nearby Unity 6 version may work, but `6000.4.6f1` is the safest option for exact reproduction.

## Required Files

To reproduce the source version of the demo, the following folders/files should be included:

```text
Assets/PoultryTwinDemo/
Assets/Scenes/
Assets/Settings/
Assets/StreamingAssets/
Assets/InputSystem_Actions.inputactions
Assets/InputSystem_Actions.inputactions.meta
Packages/
ProjectSettings/
validate_demo_files.py
```


## Paid Chicken Model Assets

The current local demo can use animated chicken models from a paid Unity Asset Store package:

```text
Assets/ANIMALS FULL PACK/Farm Animals Pack/Chicken/
```

This repository can not publicly redistribute paid Asset Store content 
To reproduce the animated chicken-model version locally:

1. Purchase/import the same chicken asset package in Unity. `https://assetstore.unity.com/packages/3d/characters/animals/birds/chickens-5029`
2. Ensure the imported chicken assets are available under:

   ```text
   Assets/ANIMALS FULL PACK/Farm Animals Pack/Chicken/
   ```

3. Recreate or provide local-only runtime prefabs under:

   ```text
   Assets/Resources/PoultryTwin/
     Chicken1_PBR.prefab
     Chicken2_PBR.prefab
     Chicken3_PBR.prefab
   ```

These runtime prefabs are loaded by `ZoneOverlayController` through Unity `Resources.Load`.

## Quick Start

1. Clone or copy the repository.
2. Open Unity Hub.
3. Add/open the Unity project folder.
4. Open:

   ```text
   Assets/Scenes/SampleScene.unity
   ```

5. Press Play.

At runtime, the bootstrap/controller scripts create or configure the camera, lighting, room layout, semantic zone overlays, playback controller, bird visuals, and HUD.


## Source Validation

Before opening Unity, you can run a lightweight file validation:

```bash
python validate_demo_files.py
```

The validator checks that:

- `Assets/StreamingAssets/poultry_twin_demo_timeline.json` exists.
- The timeline is non-empty.
- Required runtime scripts are present.
- The runtime scripts avoid unsupported dependencies for this source-only demo.

## Data Pipeline Notes

The Unity timeline JSON was generated outside Unity by a Python pipeline. In the current design:

- Room and zone metrics remain aggregated over 30-second windows for stability.
- Unity frames advance with a 10-second stride.
- Bird-level visual actions are intended to use the center 10-second slice for each overlapping 30-second window.
- Bird positions and behavior labels are already embedded in:

  ```text
  timeline[].birds
  ```

## Development Notes

Main runtime scripts:

```text
Assets/PoultryTwinDemo/Scripts/PoultryTwinJsonLoader.cs
Assets/PoultryTwinDemo/Scripts/PoultryTwinPlaybackController.cs
Assets/PoultryTwinDemo/Scripts/ZoneOverlayController.cs
Assets/PoultryTwinDemo/Scripts/DemoHudController.cs
Assets/PoultryTwinDemo/Scripts/PoultryTwinRoomLayout.cs
Assets/PoultryTwinDemo/Scripts/PoultryTwinCameraController.cs
Assets/PoultryTwinDemo/Scripts/PoultryTwinDemoBootstrap.cs
```

Editor utilities:

```text
Assets/PoultryTwinDemo/Editor/CreatePoultryTwinDemoScene.cs
Assets/PoultryTwinDemo/Editor/CapturePoultryTwinDemo.cs
```
