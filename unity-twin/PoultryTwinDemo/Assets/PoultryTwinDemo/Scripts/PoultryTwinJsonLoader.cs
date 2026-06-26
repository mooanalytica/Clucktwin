using System;
using System.IO;
using UnityEngine;

namespace PoultryTwinDemo
{
    public class PoultryTwinJsonLoader : MonoBehaviour
    {
        [SerializeField] private string jsonFileName = "poultry_twin_demo_timeline.json";
        [SerializeField] private string overrideJsonPath = string.Empty;
        [SerializeField] private bool loadOnAwake;

        public PoultryTwinTimelineFile Data { get; private set; }
        public string ResolvedPath { get; private set; }
        public string LastErrorMessage { get; private set; }
        public string LastWarningMessage { get; private set; }

        public bool IsLoaded
        {
            get { return Data != null && Data.timeline != null && Data.timeline.Length > 0; }
        }

        private void Awake()
        {
            if (loadOnAwake)
            {
                TryLoad();
            }
        }

        public bool TryLoad()
        {
            Data = null;
            LastErrorMessage = string.Empty;
            LastWarningMessage = string.Empty;
            ResolvedPath = GetResolvedPath();

            if (string.IsNullOrEmpty(ResolvedPath))
            {
                return SetError("JSON path could not be resolved from StreamingAssets.");
            }

            if (!File.Exists(ResolvedPath))
            {
                return SetError("JSON missing: " + ResolvedPath);
            }

            string jsonText;
            try
            {
                jsonText = File.ReadAllText(ResolvedPath);
            }
            catch (Exception exception)
            {
                return SetError("JSON read failed: " + exception.Message);
            }

            if (string.IsNullOrWhiteSpace(jsonText))
            {
                return SetError("JSON file is empty: " + ResolvedPath);
            }

            try
            {
                Data = JsonUtility.FromJson<PoultryTwinTimelineFile>(jsonText);
            }
            catch (Exception exception)
            {
                Data = null;
                return SetError("JSON parse failed: " + exception.Message);
            }

            if (Data == null)
            {
                return SetError("JSON parse failed: root payload was null.");
            }

            if (Data.timeline == null || Data.timeline.Length == 0)
            {
                return SetError("Timeline empty in JSON: " + ResolvedPath);
            }

            PoultryTwinRoom primaryRoom = Data.GetPrimaryRoom();
            if (primaryRoom == null || primaryRoom.zones == null || primaryRoom.zones.Length == 0)
            {
                PoultryTwinTimelineFrame firstFrame = Data.timeline[0];
                if (firstFrame == null || firstFrame.zones == null || firstFrame.zones.Length == 0)
                {
                    LastWarningMessage = "No zones found in room definitions or timeline frames.";
                }
                else
                {
                    LastWarningMessage = "Room zone definitions are missing; runtime will use a fallback grid layout.";
                }
            }

            if (!string.IsNullOrEmpty(LastWarningMessage))
            {
                Debug.LogWarning(LastWarningMessage);
            }

            return true;
        }

        public string GetResolvedPath()
        {
            if (!string.IsNullOrWhiteSpace(overrideJsonPath))
            {
                if (Path.IsPathRooted(overrideJsonPath))
                {
                    return overrideJsonPath;
                }

                string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
                return Path.GetFullPath(Path.Combine(projectRoot, overrideJsonPath));
            }

            return Path.Combine(Application.streamingAssetsPath, jsonFileName);
        }

        public void SetOverrideJsonPath(string filePath)
        {
            overrideJsonPath = filePath ?? string.Empty;
        }

        private bool SetError(string message)
        {
            LastErrorMessage = message;
            Debug.LogError(message);
            return false;
        }
    }
}
