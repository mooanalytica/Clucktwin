using UnityEngine;

namespace PoultryTwinDemo
{
    public static class PoultryTwinDemoBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Bootstrap()
        {
            EnsureCamera();
            EnsureLight();
            EnsureRuntimeController();
        }

        private static void EnsureRuntimeController()
        {
            PoultryTwinPlaybackController playbackController = Object.FindFirstObjectByType<PoultryTwinPlaybackController>();
            GameObject rootObject = playbackController != null
                ? playbackController.gameObject
                : new GameObject("PoultryTwinDemoRuntime");

            PoultryTwinJsonLoader loader = GetOrAddComponent<PoultryTwinJsonLoader>(rootObject);
            ZoneOverlayController overlayController = GetOrAddComponent<ZoneOverlayController>(rootObject);
            DemoHudController hudController = GetOrAddComponent<DemoHudController>(rootObject);
            playbackController = GetOrAddComponent<PoultryTwinPlaybackController>(rootObject);

            Transform zoneOverlayRoot = rootObject.transform.Find("ZoneOverlayRoot");
            if (zoneOverlayRoot == null)
            {
                zoneOverlayRoot = rootObject.transform.Find("ZoneOverlays");
            }

            if (zoneOverlayRoot == null)
            {
                GameObject zoneRootObject = new GameObject("ZoneOverlayRoot");
                zoneRootObject.transform.SetParent(rootObject.transform, false);
                zoneOverlayRoot = zoneRootObject.transform;
            }

            overlayController.SetZoneRoot(zoneOverlayRoot);
            playbackController.SetReferences(loader, overlayController, hudController);
            hudController.SetPlaybackController(playbackController);
        }

        private static void EnsureCamera()
        {
            Camera mainCamera = Camera.main;
            if (mainCamera != null)
            {
                ConfigureCamera(mainCamera);
                return;
            }

            Camera anyCamera = Object.FindFirstObjectByType<Camera>();
            if (anyCamera != null)
            {
                anyCamera.tag = "MainCamera";
                ConfigureCamera(anyCamera);
                return;
            }

            GameObject cameraObject = new GameObject("PoultryTwinCamera");
            cameraObject.tag = "MainCamera";
            Camera cameraComponent = cameraObject.AddComponent<Camera>();
            cameraObject.AddComponent<AudioListener>();
            ConfigureCamera(cameraComponent);
        }

        private static void EnsureLight()
        {
            Light anyLight = Object.FindFirstObjectByType<Light>();
            if (anyLight != null)
            {
                ConfigureLight(anyLight);
                return;
            }

            GameObject lightObject = new GameObject("PoultryTwinLight");
            Light lightComponent = lightObject.AddComponent<Light>();
            ConfigureLight(lightComponent);
        }

        private static T GetOrAddComponent<T>(GameObject target) where T : Component
        {
            T existing = target.GetComponent<T>();
            return existing != null ? existing : target.AddComponent<T>();
        }

        private static void ConfigureCamera(Camera cameraComponent)
        {
            if (cameraComponent == null)
            {
                return;
            }

            cameraComponent.clearFlags = CameraClearFlags.SolidColor;
            cameraComponent.backgroundColor = new Color(0.02f, 0.05f, 0.08f);
            cameraComponent.orthographic = false;
            cameraComponent.fieldOfView = PoultryTwinRoomLayout.CameraFieldOfView;
            cameraComponent.nearClipPlane = 0.1f;
            cameraComponent.farClipPlane = 100.0f;
            PoultryTwinCameraController cameraController = GetOrAddComponent<PoultryTwinCameraController>(cameraComponent.gameObject);
            cameraController.SetPresentationView(PoultryTwinRoomLayout.CameraPosition, PoultryTwinRoomLayout.CameraRotation, PoultryTwinRoomLayout.CameraFocusPoint);
        }

        private static void ConfigureLight(Light lightComponent)
        {
            if (lightComponent == null)
            {
                return;
            }

            lightComponent.type = LightType.Directional;
            lightComponent.intensity = 1.32f;
            lightComponent.shadows = LightShadows.Soft;
            lightComponent.shadowStrength = 0.72f;
            lightComponent.transform.rotation = Quaternion.Euler(59.0f, -28.0f, 0.0f);
        }
    }
}
