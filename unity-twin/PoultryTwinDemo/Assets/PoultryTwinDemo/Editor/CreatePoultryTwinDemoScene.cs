using PoultryTwinDemo;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PoultryTwinDemoEditor
{
    public static class CreatePoultryTwinDemoScene
    {
        [MenuItem("Poultry Twin/Create Demo Scene")]
        public static void CreateScene()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            GameObject cameraObject = new GameObject("Main Camera");
            Camera cameraComponent = cameraObject.AddComponent<Camera>();
            cameraObject.tag = "MainCamera";
            cameraObject.transform.position = PoultryTwinRoomLayout.CameraPosition;
            cameraObject.transform.rotation = Quaternion.Euler(PoultryTwinRoomLayout.CameraRotation);
            cameraComponent.clearFlags = CameraClearFlags.SolidColor;
            cameraComponent.backgroundColor = new Color(0.02f, 0.05f, 0.08f);
            cameraComponent.orthographic = false;
            cameraComponent.fieldOfView = PoultryTwinRoomLayout.CameraFieldOfView;
            PoultryTwinCameraController cameraController = cameraObject.AddComponent<PoultryTwinCameraController>();
            cameraController.SetPresentationView(PoultryTwinRoomLayout.CameraPosition, PoultryTwinRoomLayout.CameraRotation, PoultryTwinRoomLayout.CameraFocusPoint);

            GameObject lightObject = new GameObject("Directional Light");
            Light lightComponent = lightObject.AddComponent<Light>();
            lightComponent.type = LightType.Directional;
            lightComponent.intensity = 1.32f;
            lightComponent.shadows = LightShadows.Soft;
            lightComponent.shadowStrength = 0.72f;
            lightObject.transform.rotation = Quaternion.Euler(59.0f, -28.0f, 0.0f);

            GameObject rootObject = new GameObject("PoultryTwinDemoRoot");
            GameObject zoneOverlayRoot = new GameObject("ZoneOverlayRoot");
            zoneOverlayRoot.transform.SetParent(rootObject.transform, false);

            PoultryTwinJsonLoader loader = rootObject.AddComponent<PoultryTwinJsonLoader>();
            ZoneOverlayController overlayController = rootObject.AddComponent<ZoneOverlayController>();
            overlayController.SetZoneRoot(zoneOverlayRoot.transform);

            DemoHudController hudController = rootObject.AddComponent<DemoHudController>();
            PoultryTwinPlaybackController playbackController = rootObject.AddComponent<PoultryTwinPlaybackController>();
            playbackController.SetReferences(loader, overlayController, hudController);
            hudController.SetPlaybackController(playbackController);

            EditorSceneManager.SaveScene(scene, "Assets/Scenes/PoultryTwinDemoScene.unity");
            Selection.activeGameObject = rootObject;
            Debug.Log("Created Poultry Twin demo scene at Assets/Scenes/PoultryTwinDemoScene.unity");
        }
    }
}
