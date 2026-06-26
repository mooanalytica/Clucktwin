using System;
using System.IO;
using PoultryTwinDemo;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PoultryTwinDemoEditor
{
    public static class CapturePoultryTwinDemo
    {
        private const int CaptureWidth = 1920;
        private const int CaptureHeight = 1080;

        [MenuItem("Poultry Twin/Capture Demo Views")]
        public static void CaptureDemoViews()
        {
            try
            {
                string outputDirectory = GetOutputDirectory();
                Directory.CreateDirectory(outputDirectory);

                PoultryTwinTimelineFile data = BuildSceneAndLoadData(out ZoneOverlayController overlayController);
                PoultryTwinTimelineFrame frame = FindCaptureFrame(data);
                if (frame != null)
                {
                    overlayController.ApplyFrame(frame, 0.0f);
                }

                RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
                RenderSettings.ambientLight = new Color(0.72f, 0.72f, 0.70f, 1.0f);

                Camera camera = CreateCaptureCamera();
                CaptureLookAt(
                    camera,
                    "01_overview",
                    PoultryTwinRoomLayout.CameraPosition,
                    PoultryTwinRoomLayout.CameraFocusPoint,
                    PoultryTwinRoomLayout.CameraFieldOfView,
                    outputDirectory
                );

                CaptureView(
                    camera,
                    "02_top_down",
                    new Vector3(0.0f, 15.4f, 0.0f),
                    Quaternion.Euler(90.0f, 0.0f, 0.0f),
                    38.0f,
                    true,
                    8.8f,
                    outputDirectory
                );

                Vector3 feederCenter = GetZoneCenter(PoultryTwinRoomLayout.FeederZoneId, 0.24f);
                CaptureLookAt(
                    camera,
                    "03_feeder_close",
                    feederCenter + new Vector3(-3.0f, 4.1f, -3.6f),
                    feederCenter + new Vector3(0.0f, 0.25f, 0.0f),
                    34.0f,
                    outputDirectory
                );

                Vector3 drinkerCenter = GetZoneCenter(PoultryTwinRoomLayout.DrinkerZoneId, 0.26f);
                CaptureLookAt(
                    camera,
                    "04_drinker_close",
                    drinkerCenter + new Vector3(-2.6f, 3.7f, -2.8f),
                    drinkerCenter + new Vector3(0.0f, 0.25f, 0.0f),
                    32.0f,
                    outputDirectory
                );

                Debug.Log("Poultry Twin captures written to: " + outputDirectory);
                AssetDatabase.Refresh();
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorApplication.Exit(1);
            }
        }

        private static PoultryTwinTimelineFile BuildSceneAndLoadData(out ZoneOverlayController overlayController)
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            GameObject lightObject = new GameObject("Capture Directional Light");
            Light lightComponent = lightObject.AddComponent<Light>();
            lightComponent.type = LightType.Directional;
            lightComponent.intensity = 1.25f;
            lightComponent.shadows = LightShadows.Soft;
            lightComponent.shadowStrength = 0.58f;
            lightObject.transform.rotation = Quaternion.Euler(58.0f, -28.0f, 0.0f);

            GameObject rootObject = new GameObject("PoultryTwinCaptureRoot");
            GameObject zoneOverlayRoot = new GameObject("ZoneOverlayRoot");
            zoneOverlayRoot.transform.SetParent(rootObject.transform, false);

            PoultryTwinJsonLoader loader = rootObject.AddComponent<PoultryTwinJsonLoader>();
            overlayController = rootObject.AddComponent<ZoneOverlayController>();
            overlayController.SetZoneRoot(zoneOverlayRoot.transform);

            if (!loader.TryLoad())
            {
                throw new InvalidOperationException("Could not load timeline JSON: " + loader.LastErrorMessage);
            }

            overlayController.Initialize(loader.Data);
            return loader.Data;
        }

        private static PoultryTwinTimelineFrame FindCaptureFrame(PoultryTwinTimelineFile data)
        {
            if (data == null || data.timeline == null || data.timeline.Length == 0)
            {
                return null;
            }

            for (int index = 0; index < data.timeline.Length; index++)
            {
                PoultryTwinTimelineFrame frame = data.timeline[index];
                if (frame != null && frame.birds != null && frame.birds.Length >= 20)
                {
                    return frame;
                }
            }

            return data.timeline[0];
        }

        private static Camera CreateCaptureCamera()
        {
            GameObject cameraObject = new GameObject("PoultryTwinCaptureCamera");
            Camera cameraComponent = cameraObject.AddComponent<Camera>();
            cameraComponent.clearFlags = CameraClearFlags.SolidColor;
            cameraComponent.backgroundColor = new Color(0.02f, 0.05f, 0.08f, 1.0f);
            cameraComponent.nearClipPlane = 0.05f;
            cameraComponent.farClipPlane = 120.0f;
            cameraComponent.allowHDR = true;
            cameraComponent.allowMSAA = true;
            return cameraComponent;
        }

        private static void CaptureLookAt(
            Camera camera,
            string captureName,
            Vector3 position,
            Vector3 target,
            float fieldOfView,
            string outputDirectory)
        {
            Quaternion rotation = Quaternion.LookRotation((target - position).normalized, Vector3.up);
            CaptureView(camera, captureName, position, rotation, fieldOfView, false, 0.0f, outputDirectory);
        }

        private static void CaptureView(
            Camera camera,
            string captureName,
            Vector3 position,
            Quaternion rotation,
            float fieldOfView,
            bool orthographic,
            float orthographicSize,
            string outputDirectory)
        {
            camera.transform.position = position;
            camera.transform.rotation = rotation;
            camera.orthographic = orthographic;
            camera.orthographicSize = orthographicSize;
            camera.fieldOfView = fieldOfView;

            RenderTexture renderTexture = new RenderTexture(CaptureWidth, CaptureHeight, 24, RenderTextureFormat.ARGB32);
            Texture2D texture = new Texture2D(CaptureWidth, CaptureHeight, TextureFormat.RGB24, false);
            RenderTexture previousActive = RenderTexture.active;
            RenderTexture previousTarget = camera.targetTexture;

            try
            {
                camera.targetTexture = renderTexture;
                RenderTexture.active = renderTexture;
                camera.Render();
                texture.ReadPixels(new Rect(0, 0, CaptureWidth, CaptureHeight), 0, 0);
                texture.Apply();

                string path = Path.Combine(outputDirectory, captureName + ".png");
                File.WriteAllBytes(path, texture.EncodeToPNG());
                Debug.Log("Captured " + path);
            }
            finally
            {
                camera.targetTexture = previousTarget;
                RenderTexture.active = previousActive;
                UnityEngine.Object.DestroyImmediate(texture);
                renderTexture.Release();
                UnityEngine.Object.DestroyImmediate(renderTexture);
            }
        }

        private static Vector3 GetZoneCenter(string zoneId, float y)
        {
            PoultryTwinRoomLayout.ZoneProfile profile = PoultryTwinRoomLayout.GetProfileOrDefault(zoneId);
            Rect largestRect = default;
            float largestArea = 0.0f;
            Rect[] rects = profile.WorldRects;
            if (rects != null)
            {
                for (int index = 0; index < rects.Length; index++)
                {
                    Rect rect = rects[index];
                    float area = Mathf.Abs(rect.width * rect.height);
                    if (area > largestArea)
                    {
                        largestRect = rect;
                        largestArea = area;
                    }
                }
            }

            return largestArea > 0.0f
                ? new Vector3(largestRect.center.x, y, largestRect.center.y)
                : new Vector3(0.0f, y, 0.0f);
        }

        private static string GetOutputDirectory()
        {
            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            return Path.Combine(projectRoot, "Captures", "PoultryTwinDemo_" + stamp);
        }
    }
}
