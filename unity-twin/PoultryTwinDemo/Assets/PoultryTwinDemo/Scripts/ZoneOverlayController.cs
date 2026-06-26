using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace PoultryTwinDemo
{
    public class ZoneOverlayController : MonoBehaviour
    {
        [SerializeField] private Transform zoneRoot;
        [SerializeField] private float surfaceHeight = 0.035f;
        [SerializeField] private float outlineThickness = 0.06f;
        [SerializeField] private float overlayLift = 0.08f;
        [SerializeField] private float birdFloorHeight = 0.18f;
        [SerializeField] private float birdRestingHeight = 0.96f;
        [SerializeField] private float birdScale = 0.28f;
        [SerializeField] private bool useChickenModel = true;
        [SerializeField] private GameObject[] chickenPrefabs;
        [SerializeField]
        private string[] chickenResourcePaths =
        {
            "PoultryTwin/Chicken1_PBR",
            "PoultryTwin/Chicken2_PBR",
            "PoultryTwin/Chicken3_PBR",
        };
        [SerializeField] private float chickenModelScale = 1.0f;
        [SerializeField] private Vector3 chickenModelEulerOffset = Vector3.zero;
        [SerializeField] private float chickenModelHeight = 0.02f;
        [SerializeField] private float birdMoveSpeed = 3.8f;
        [SerializeField] private float birdTurnSpeed = 540.0f;
        [SerializeField] private float birdTeleportDistance = 3.2f;
        [SerializeField] private float birdIdentityReuseDistance = 1.4f;
        [SerializeField] private float birdWalkAnimationThreshold = 0.08f;
        [SerializeField] private bool displayFirstZoneProxyMode = true;
        [SerializeField] private bool preferWalkRootMotionClipForWalking = true;
        [SerializeField] private bool useAnimatorRootMotionForDisplayMovement = true;
        [SerializeField] private float displayZoneIdentityReuseDistance = 3.2f;
        [SerializeField] private float displayZoneSlotJitter = 0.16f;
        [SerializeField] private float displayActiveWanderRadius = 0.28f;
        [SerializeField] private float rootMotionArrivalDistance = 0.26f;
        [SerializeField] private int displayWingFlapFrameGap = 12;
        [SerializeField] private int displayMaxWingFlapsPerFrame = 1;
        [SerializeField] private Color roomShellColor = new Color(0.88f, 0.88f, 0.84f, 1.0f);
        [SerializeField] private Color roomFrameColor = new Color(0.20f, 0.22f, 0.24f, 1.0f);
        [SerializeField] private Color floorColor = new Color(0.62f, 0.50f, 0.30f, 1.0f);
        [SerializeField] private Color floorAccentColor = new Color(0.78f, 0.67f, 0.42f, 1.0f);
        [SerializeField] private Color inactiveZoneColor = new Color(0.68f, 0.71f, 0.74f, 0.10f);
        [SerializeField] private Color selectedZoneColor = new Color(0.95f, 0.91f, 0.80f, 0.52f);
        [SerializeField] private Color abnormalZoneColor = new Color(0.84f, 0.43f, 0.33f, 0.42f);

        private readonly Dictionary<string, ZoneVisualRecord> zoneVisuals = new Dictionary<string, ZoneVisualRecord>();
        private readonly Dictionary<string, PoultryTwinZoneFrame> currentZoneLookup = new Dictionary<string, PoultryTwinZoneFrame>();
        private readonly Dictionary<string, float> zoneActivityReferenceLookup = new Dictionary<string, float>();
        private readonly Dictionary<Material, Material> chickenMaterialLookup = new Dictionary<Material, Material>();
        private readonly Dictionary<string, BirdVisualRecord> birdVisualLookup = new Dictionary<string, BirdVisualRecord>();
        private readonly HashSet<string> currentFrameBirdKeys = new HashSet<string>();
        private readonly List<BirdVisualRecord> birdVisuals = new List<BirdVisualRecord>();

        private Transform environmentRoot;
        private Transform birdRoot;
        private Camera cachedCamera;
        private Texture2D roomLitterTexture;
        private PoultryTwinTimelineFrame currentFrame;
        private string selectedZoneId;
        private int lastDisplayedWingFlapFrameIndex = -100000;
        private int displayedWingFlapsThisFrame;

        public bool HasZones { get; private set; }
        public string StatusMessage { get; private set; }
        public string LastErrorMessage { get; private set; }

        public string SelectedZoneId
        {
            get { return selectedZoneId; }
        }

        public string SelectedZoneDisplayName
        {
            get
            {
                if (string.IsNullOrEmpty(selectedZoneId))
                {
                    return "None";
                }

                ZoneVisualRecord record;
                return zoneVisuals.TryGetValue(selectedZoneId, out record) ? record.DisplayName : selectedZoneId;
            }
        }

        public bool HasSelectedZone
        {
            get { return !string.IsNullOrEmpty(selectedZoneId); }
        }

        public void SetZoneRoot(Transform rootTransform)
        {
            zoneRoot = rootTransform;
        }

        public void SelectZoneById(string zoneId)
        {
            if (string.IsNullOrEmpty(zoneId) || !zoneVisuals.ContainsKey(zoneId))
            {
                return;
            }

            selectedZoneId = zoneId;
            ApplyFrame(currentFrame);
        }

        public void ClearSelection()
        {
            selectedZoneId = string.Empty;
            ApplyFrame(currentFrame);
        }

        public void Initialize(PoultryTwinTimelineFile data)
        {
            EnsureRoots();
            ClearChildren(environmentRoot);
            ClearChildren(zoneRoot);
            ClearChildren(birdRoot);

            zoneVisuals.Clear();
            currentZoneLookup.Clear();
            zoneActivityReferenceLookup.Clear();
            chickenMaterialLookup.Clear();
            birdVisualLookup.Clear();
            currentFrameBirdKeys.Clear();
            birdVisuals.Clear();
            currentFrame = null;
            selectedZoneId = string.Empty;
            HasZones = false;
            StatusMessage = string.Empty;
            LastErrorMessage = string.Empty;

            BuildZoneActivityCalibration(data);
            BuildRoomEnvironment();
            BuildSemanticZoneVisuals();
            BuildBirdPool(data);

            HasZones = zoneVisuals.Count > 0;
            StatusMessage = HasZones
                ? "Ready. Simplified 3D Room 1 scene built with zone overlays and bird proxies."
                : "No zones available.";
        }

        public void ApplyFrame(PoultryTwinTimelineFrame frame)
        {
            ApplyFrame(frame, -1.0f);
        }

        public void ApplyFrame(PoultryTwinTimelineFrame frame, float birdTransitionSeconds)
        {
            currentFrame = frame;
            currentZoneLookup.Clear();

            if (frame != null && frame.zones != null)
            {
                for (int index = 0; index < frame.zones.Length; index++)
                {
                    PoultryTwinZoneFrame zoneFrame = frame.zones[index];
                    if (zoneFrame == null || string.IsNullOrEmpty(zoneFrame.zone_id))
                    {
                        continue;
                    }

                    string semanticZoneId;
                    if (!PoultryTwinRoomLayout.TryMapExternalZoneId(zoneFrame.zone_id, out semanticZoneId))
                    {
                        semanticZoneId = zoneFrame.zone_id;
                    }

                    currentZoneLookup[semanticZoneId] = zoneFrame;
                }
            }

            float riskScore = frame != null && frame.welfare != null ? Mathf.Clamp01(frame.welfare.risk_score) : 0.0f;
            string eventPhase = frame != null && frame.@event != null ? frame.@event.event_phase : string.Empty;

            foreach (KeyValuePair<string, ZoneVisualRecord> entry in zoneVisuals)
            {
                ZoneVisualRecord record = entry.Value;
                PoultryTwinZoneFrame zoneFrame;
                bool hasFrame = currentZoneLookup.TryGetValue(entry.Key, out zoneFrame);
                float activity = hasFrame ? GetDisplayValue(zoneFrame) : 0.0f;
                float normalized = hasFrame ? GetNormalizedValue(zoneFrame) : 0.0f;
                bool isSelected = entry.Key == selectedZoneId;
                bool isAbnormal = hasFrame && !IsPendingZoneFrame(zoneFrame) && IsZoneAbnormal(entry.Key, normalized, riskScore);
                UpdateZoneVisual(record, normalized, isSelected, isAbnormal, eventPhase);
            }

            UpdateBirdVisuals(frame, birdTransitionSeconds);
        }

        public bool TryGetSelectedZoneFrame(out PoultryTwinZoneFrame zoneFrame)
        {
            zoneFrame = null;
            if (string.IsNullOrEmpty(selectedZoneId))
            {
                return false;
            }

            return currentZoneLookup.TryGetValue(selectedZoneId, out zoneFrame);
        }

        public string GetSelectedZoneSummary()
        {
            PoultryTwinZoneFrame zoneFrame;
            if (!TryGetSelectedZoneFrame(out zoneFrame))
            {
                return "Click a highlighted zone to inspect its activity and occupancy proxies.";
            }

            float activity = GetDisplayValue(zoneFrame);
            float normalized = GetNormalizedValue(zoneFrame);
            return
                "Activity " + activity.ToString("0.00") +
                " | Relative intensity " + normalized.ToString("0.00") +
                " | Occupancy share " + zoneFrame.occupancy_share.ToString("0.00");
        }

        public float GetDisplayActivityValue(PoultryTwinZoneFrame zoneFrame)
        {
            return GetDisplayValue(zoneFrame);
        }

        private void Update()
        {
            if (HasZones)
            {
                UpdateBirdMotion(Time.deltaTime);
            }

            if (!HasZones)
            {
                return;
            }

            if (PoultryTwinInputAdapter.GetKeyDown(KeyCode.Escape) && !string.IsNullOrEmpty(selectedZoneId))
            {
                ClearSelection();
            }

            if (!PoultryTwinInputAdapter.GetMouseButtonDown(0))
            {
                return;
            }

            if (PoultryTwinCameraController.ActiveInstance != null && PoultryTwinCameraController.ActiveInstance.IsNavigatingScene)
            {
                return;
            }

            Vector2 pointerPosition;
            if (!PoultryTwinInputAdapter.TryGetMousePosition(out pointerPosition))
            {
                return;
            }

            if (DemoHudController.ActiveInstance != null && DemoHudController.ActiveInstance.ContainsScreenPoint(pointerPosition))
            {
                return;
            }

            if (cachedCamera == null)
            {
                cachedCamera = Camera.main;
            }

            if (cachedCamera == null)
            {
                return;
            }

            Ray ray = cachedCamera.ScreenPointToRay(pointerPosition);
            RaycastHit[] hits = Physics.RaycastAll(ray, 100.0f);
            ZoneSurfaceMarker marker = FindClosestZoneMarker(hits);
            if (marker == null)
            {
                return;
            }

            selectedZoneId = marker.ZoneId;
            ApplyFrame(currentFrame);
        }

        private void LateUpdate()
        {
            if (!HasZones || !displayFirstZoneProxyMode || !useAnimatorRootMotionForDisplayMovement)
            {
                return;
            }

            for (int index = 0; index < birdVisuals.Count; index++)
            {
                ApplyAnimatorRootMotionToProxy(birdVisuals[index]);
            }
        }

        private void EnsureRoots()
        {
            if (environmentRoot == null)
            {
                Transform existingEnvironment = transform.Find("RoomEnvironmentRoot");
                if (existingEnvironment == null)
                {
                    GameObject environmentObject = new GameObject("RoomEnvironmentRoot");
                    environmentObject.transform.SetParent(transform, false);
                    existingEnvironment = environmentObject.transform;
                }

                environmentRoot = existingEnvironment;
            }

            if (zoneRoot == null)
            {
                Transform existingZoneRoot = transform.Find("ZoneOverlayRoot");
                if (existingZoneRoot == null)
                {
                    GameObject zoneRootObject = new GameObject("ZoneOverlayRoot");
                    zoneRootObject.transform.SetParent(transform, false);
                    existingZoneRoot = zoneRootObject.transform;
                }

                zoneRoot = existingZoneRoot;
            }

            if (birdRoot == null)
            {
                Transform existingBirdRoot = transform.Find("BirdProxyRoot");
                if (existingBirdRoot == null)
                {
                    GameObject birdRootObject = new GameObject("BirdProxyRoot");
                    birdRootObject.transform.SetParent(transform, false);
                    existingBirdRoot = birdRootObject.transform;
                }

                birdRoot = existingBirdRoot;
            }
        }

        private ZoneSurfaceMarker FindClosestZoneMarker(RaycastHit[] hits)
        {
            if (hits == null || hits.Length == 0)
            {
                return null;
            }

            float closestDistance = float.MaxValue;
            ZoneSurfaceMarker closestMarker = null;
            for (int index = 0; index < hits.Length; index++)
            {
                RaycastHit hit = hits[index];
                ZoneSurfaceMarker marker = hit.collider != null ? hit.collider.GetComponent<ZoneSurfaceMarker>() : null;
                if (marker == null || hit.distance >= closestDistance)
                {
                    continue;
                }

                closestDistance = hit.distance;
                closestMarker = marker;
            }

            return closestMarker;
        }

        private void ClearChildren(Transform root)
        {
            if (root == null)
            {
                return;
            }

            for (int index = root.childCount - 1; index >= 0; index--)
            {
                GameObject child = root.GetChild(index).gameObject;
                if (Application.isPlaying)
                {
                    Destroy(child);
                }
                else
                {
                    DestroyImmediate(child);
                }
            }
        }

        private void BuildZoneActivityCalibration(PoultryTwinTimelineFile data)
        {
            zoneActivityReferenceLookup.Clear();
            if (data == null || data.timeline == null)
            {
                return;
            }

            Dictionary<string, List<float>> valuesByZone = new Dictionary<string, List<float>>();
            for (int frameIndex = 0; frameIndex < data.timeline.Length; frameIndex++)
            {
                PoultryTwinTimelineFrame frame = data.timeline[frameIndex];
                if (frame == null || frame.zones == null)
                {
                    continue;
                }

                for (int zoneIndex = 0; zoneIndex < frame.zones.Length; zoneIndex++)
                {
                    PoultryTwinZoneFrame zoneFrame = frame.zones[zoneIndex];
                    if (zoneFrame == null || IsPendingZoneFrame(zoneFrame))
                    {
                        continue;
                    }

                    string semanticZoneId;
                    if (!PoultryTwinRoomLayout.TryMapExternalZoneId(zoneFrame.zone_id, out semanticZoneId))
                    {
                        semanticZoneId = zoneFrame.zone_id;
                    }

                    float rawValue = GetRawActivityValue(zoneFrame);
                    if (rawValue <= 0.0f)
                    {
                        continue;
                    }

                    List<float> zoneValues;
                    if (!valuesByZone.TryGetValue(semanticZoneId, out zoneValues))
                    {
                        zoneValues = new List<float>();
                        valuesByZone[semanticZoneId] = zoneValues;
                    }

                    zoneValues.Add(rawValue);
                }
            }

            foreach (KeyValuePair<string, List<float>> entry in valuesByZone)
            {
                List<float> sortedValues = entry.Value;
                if (sortedValues.Count == 0)
                {
                    continue;
                }

                sortedValues.Sort();
                float percentile95 = GetPercentile(sortedValues, 0.95f);
                float maxValue = sortedValues[sortedValues.Count - 1];
                zoneActivityReferenceLookup[entry.Key] = Mathf.Max(percentile95, maxValue * 0.35f, 0.0001f);
            }
        }

        private void BuildRoomEnvironment()
        {
            Vector2 roomSize = PoultryTwinRoomLayout.RoomSize;
            float halfWidth = roomSize.x * 0.5f;
            float halfDepth = roomSize.y * 0.5f;

            CreateCube("OuterApron", environmentRoot, new Vector3(0.0f, -0.16f, 0.0f), new Vector3(roomSize.x + 4.8f, 0.22f, roomSize.y + 4.8f), new Color(0.82f, 0.82f, 0.79f, 1.0f));
            CreateCube("RoomBase", environmentRoot, new Vector3(0.0f, -0.03f, 0.0f), new Vector3(roomSize.x + 0.28f, 0.18f, roomSize.y + 0.28f), roomFrameColor);

            GameObject floor = CreateCube("Floor", environmentRoot, new Vector3(0.0f, 0.02f, 0.0f), new Vector3(roomSize.x - 0.18f, 0.08f, roomSize.y - 0.18f), floorColor);
            ApplyLitterMaterial(floor.GetComponent<Renderer>(), new Vector2(3.0f, 4.8f));

            GameObject litterTop = CreateCube("LitterTop", environmentRoot, new Vector3(0.0f, 0.067f, 0.0f), new Vector3(roomSize.x - 0.70f, 0.012f, roomSize.y - 0.70f), floorAccentColor);
            ApplyLitterMaterial(litterTop.GetComponent<Renderer>(), new Vector2(3.5f, 5.4f));

            float wallHeight = 2.65f;
            CreateCube("LeftWall", environmentRoot, new Vector3(-halfWidth - 0.18f, wallHeight * 0.5f, 0.0f), new Vector3(0.36f, wallHeight, roomSize.y + 0.65f), roomShellColor);
            CreateCube("BottomWall", environmentRoot, new Vector3(0.0f, wallHeight * 0.5f, -halfDepth - 0.18f), new Vector3(roomSize.x + 0.72f, wallHeight, 0.36f), roomShellColor);
            CreateCube("TopWall", environmentRoot, new Vector3(0.0f, wallHeight * 0.5f, halfDepth + 0.18f), new Vector3(roomSize.x + 0.72f, wallHeight, 0.36f), roomShellColor);
            CreateCube("RightWallUpper", environmentRoot, new Vector3(halfWidth + 0.18f, wallHeight * 0.5f, 4.7f), new Vector3(0.36f, wallHeight, 7.0f), roomShellColor);
            CreateCube("RightWallLower", environmentRoot, new Vector3(halfWidth + 0.18f, wallHeight * 0.5f, -4.9f), new Vector3(0.36f, wallHeight, 5.8f), roomShellColor);
            CreateCube("DoorHeader", environmentRoot, new Vector3(halfWidth + 0.18f, 2.0f, 0.7f), new Vector3(0.36f, 0.30f, 2.4f), roomFrameColor);
            CreateCube("DoorThreshold", environmentRoot, new Vector3(halfWidth - 0.02f, 0.04f, 0.7f), new Vector3(0.24f, 0.04f, 2.3f), new Color(0.80f, 0.81f, 0.82f, 1.0f));

            BuildWallPanelDetails(roomSize, wallHeight);
            BuildFloorLitterDetail(roomSize);
            BuildLowPerchRail(roomSize);
            BuildCeilingFixtures(roomSize, wallHeight);
            BuildWallEquipment(roomSize, wallHeight);
            BuildFeeder(wallHeight);
            BuildDrinker(wallHeight);
        }

        private void BuildRestingRack()
        {
            PoultryTwinRoomLayout.ZoneProfile profile = PoultryTwinRoomLayout.GetProfileOrDefault(PoultryTwinRoomLayout.RestingZoneId);
            Rect platformRect = GetLargestWorldRect(profile.WorldRects);
            if (platformRect.width <= 0.0f || platformRect.height <= 0.0f)
            {
                return;
            }

            float platformWidth = Mathf.Clamp(platformRect.width * 0.56f, 0.84f, 1.04f);
            float platformDepth = Mathf.Max(4.8f, platformRect.height * 0.88f);
            float platformX = platformRect.center.x + (platformRect.width * 0.05f);
            float platformZ = platformRect.center.y;
            CreateCube("RestingPlatform", environmentRoot, new Vector3(platformX, 0.74f, platformZ), new Vector3(platformWidth, 0.14f, platformDepth), new Color(0.22f, 0.24f, 0.27f, 1.0f));
            CreateCube("RestingSideRail", environmentRoot, new Vector3(platformX - (platformWidth * 0.42f), 0.95f, platformZ), new Vector3(0.08f, 0.28f, platformDepth - 0.46f), roomFrameColor);
            for (int index = 0; index < 5; index++)
            {
                float supportZ = (platformZ - (platformDepth * 0.40f)) + (index * (platformDepth * 0.20f));
                CreateCube("RestingSupport_" + index, environmentRoot, new Vector3(platformX - 0.16f, 0.34f, supportZ), new Vector3(0.12f, 0.62f, 0.12f), roomFrameColor);
            }

            CreateCube(
                "RestingRamp",
                environmentRoot,
                new Vector3(platformRect.xMin - 0.14f, 0.35f, platformRect.yMin + 1.55f),
                new Vector3(0.30f, 0.38f, 2.4f),
                new Color(0.18f, 0.20f, 0.22f, 1.0f),
                new Vector3(-18.0f, 0.0f, 0.0f)
            );
        }

        private void BuildFeeder(float wallHeight)
        {
            PoultryTwinRoomLayout.ZoneProfile profile = PoultryTwinRoomLayout.GetProfileOrDefault(PoultryTwinRoomLayout.FeederZoneId);
            Rect feederRect = GetLargestWorldRect(profile.WorldRects);
            if (feederRect.width <= 0.0f || feederRect.height <= 0.0f)
            {
                return;
            }

            Vector3 feederCenter = new Vector3(feederRect.center.x, 0.15f, feederRect.center.y);
            Color feederGreen = new Color(0.52f, 0.84f, 0.22f, 1.0f);
            Color feederDarkGreen = new Color(0.26f, 0.55f, 0.23f, 1.0f);
            Color feedColor = new Color(0.80f, 0.69f, 0.40f, 1.0f);

            CreateCylinder("FeederTray", environmentRoot, new Vector3(feederCenter.x, 0.14f, feederCenter.z), new Vector3(0.92f, 0.07f, 0.92f), feederDarkGreen);
            CreateCylinder("FeederBowl", environmentRoot, new Vector3(feederCenter.x, 0.22f, feederCenter.z), new Vector3(0.74f, 0.08f, 0.74f), feederGreen);
            CreateCylinder("FeederFeedRing", environmentRoot, new Vector3(feederCenter.x, 0.285f, feederCenter.z), new Vector3(0.56f, 0.015f, 0.56f), feedColor);
            CreateCylinder("FeederHopper", environmentRoot, new Vector3(feederCenter.x, 0.56f, feederCenter.z), new Vector3(0.42f, 0.32f, 0.42f), feederGreen);
            CreateCylinder("FeederCap", environmentRoot, new Vector3(feederCenter.x, 0.91f, feederCenter.z), new Vector3(0.54f, 0.045f, 0.54f), feederDarkGreen);
            CreateCableBetweenPoints(
                "FeederHangerCable",
                environmentRoot,
                new Vector3(feederCenter.x, wallHeight + 0.05f, feederCenter.z),
                new Vector3(feederCenter.x, 0.98f, feederCenter.z),
                0.022f,
                new Color(0.10f, 0.11f, 0.12f, 1.0f)
            );
        }

        private void BuildDrinker(float wallHeight)
        {
            PoultryTwinRoomLayout.ZoneProfile profile = PoultryTwinRoomLayout.GetProfileOrDefault(PoultryTwinRoomLayout.DrinkerZoneId);
            Rect drinkerRect = GetLargestWorldRect(profile.WorldRects);
            if (drinkerRect.width <= 0.0f || drinkerRect.height <= 0.0f)
            {
                return;
            }

            Vector3 drinkerCenter = new Vector3(drinkerRect.center.x, 0.18f, drinkerRect.center.y);
            Color drinkerRed = new Color(0.90f, 0.20f, 0.16f, 1.0f);
            Color drinkerHighlight = new Color(1.0f, 0.43f, 0.28f, 1.0f);
            Color waterColor = new Color(0.62f, 0.82f, 0.93f, 0.82f);

            CreateCylinder("DrinkerTray", environmentRoot, new Vector3(drinkerCenter.x, 0.13f, drinkerCenter.z), new Vector3(0.72f, 0.07f, 0.72f), drinkerRed);
            CreateCylinder("DrinkerWater", environmentRoot, new Vector3(drinkerCenter.x, 0.185f, drinkerCenter.z), new Vector3(0.58f, 0.012f, 0.58f), waterColor);
            CreateCylinder("DrinkerBowl", environmentRoot, new Vector3(drinkerCenter.x, 0.25f, drinkerCenter.z), new Vector3(0.54f, 0.09f, 0.54f), drinkerHighlight);
            CreateCylinder("DrinkerBellLower", environmentRoot, new Vector3(drinkerCenter.x, 0.43f, drinkerCenter.z), new Vector3(0.44f, 0.12f, 0.44f), drinkerRed);
            CreateCylinder("DrinkerBellUpper", environmentRoot, new Vector3(drinkerCenter.x, 0.58f, drinkerCenter.z), new Vector3(0.28f, 0.11f, 0.28f), drinkerHighlight);
            CreateCylinder("DrinkerPost", environmentRoot, new Vector3(drinkerCenter.x, 0.86f, drinkerCenter.z), new Vector3(0.06f, 0.28f, 0.06f), roomFrameColor);
            CreateCableBetweenPoints(
                "DrinkerHangerCable",
                environmentRoot,
                new Vector3(drinkerCenter.x, wallHeight + 0.05f, drinkerCenter.z),
                new Vector3(drinkerCenter.x, 1.12f, drinkerCenter.z),
                0.022f,
                new Color(0.10f, 0.11f, 0.12f, 1.0f)
            );
        }

        private void BuildWallPanelDetails(Vector2 roomSize, float wallHeight)
        {
            float halfWidth = roomSize.x * 0.5f;
            float halfDepth = roomSize.y * 0.5f;
            float innerLeftX = -halfWidth + 0.018f;
            float innerRightX = halfWidth - 0.018f;
            float innerBottomZ = -halfDepth + 0.018f;
            float innerTopZ = halfDepth - 0.018f;

            Color grooveColor = new Color(0.70f, 0.70f, 0.66f, 1.0f);
            Color baseboardColor = new Color(0.42f, 0.41f, 0.37f, 1.0f);
            Color trimColor = new Color(0.76f, 0.76f, 0.72f, 1.0f);

            CreateCube("LeftBaseboard", environmentRoot, new Vector3(innerLeftX, 0.16f, 0.0f), new Vector3(0.08f, 0.28f, roomSize.y - 0.18f), baseboardColor);
            CreateCube("BottomBaseboard", environmentRoot, new Vector3(0.0f, 0.16f, innerBottomZ), new Vector3(roomSize.x - 0.18f, 0.28f, 0.08f), baseboardColor);
            CreateCube("TopBaseboard", environmentRoot, new Vector3(0.0f, 0.16f, innerTopZ), new Vector3(roomSize.x - 0.18f, 0.28f, 0.08f), baseboardColor);
            CreateCube("RightBaseboardUpper", environmentRoot, new Vector3(innerRightX, 0.16f, 4.7f), new Vector3(0.08f, 0.28f, 6.9f), baseboardColor);
            CreateCube("RightBaseboardLower", environmentRoot, new Vector3(innerRightX, 0.16f, -4.9f), new Vector3(0.08f, 0.28f, 5.7f), baseboardColor);

            CreateCube("LeftUpperTrim", environmentRoot, new Vector3(innerLeftX, wallHeight - 0.12f, 0.0f), new Vector3(0.06f, 0.08f, roomSize.y - 0.20f), trimColor);
            CreateCube("TopUpperTrim", environmentRoot, new Vector3(0.0f, wallHeight - 0.12f, innerTopZ), new Vector3(roomSize.x - 0.20f, 0.08f, 0.06f), trimColor);
            CreateCube("BottomUpperTrim", environmentRoot, new Vector3(0.0f, wallHeight - 0.12f, innerBottomZ), new Vector3(roomSize.x - 0.20f, 0.08f, 0.06f), trimColor);

            float panelY = 0.35f + ((wallHeight - 0.58f) * 0.5f);
            float panelHeight = wallHeight - 0.58f;
            for (int index = 1; index < 22; index++)
            {
                float z = Mathf.Lerp(-halfDepth + 0.30f, halfDepth - 0.30f, index / 22.0f);
                CreateCube("LeftWallGroove_" + index, environmentRoot, new Vector3(innerLeftX + 0.006f, panelY, z), new Vector3(0.026f, panelHeight, 0.018f), grooveColor);

                if (z < -2.05f || z > 1.32f)
                {
                    CreateCube("RightWallGroove_" + index, environmentRoot, new Vector3(innerRightX - 0.006f, panelY, z), new Vector3(0.026f, panelHeight, 0.018f), grooveColor);
                }
            }

            for (int index = 1; index < 15; index++)
            {
                float x = Mathf.Lerp(-halfWidth + 0.30f, halfWidth - 0.30f, index / 15.0f);
                CreateCube("TopWallGroove_" + index, environmentRoot, new Vector3(x, panelY, innerTopZ - 0.006f), new Vector3(0.018f, panelHeight, 0.026f), grooveColor);
                CreateCube("BottomWallGroove_" + index, environmentRoot, new Vector3(x, panelY, innerBottomZ + 0.006f), new Vector3(0.018f, panelHeight, 0.026f), grooveColor);
            }

            Color panelGlass = new Color(0.52f, 0.56f, 0.56f, 0.38f);
            CreateCube("ObservationWindow", environmentRoot, new Vector3(-1.65f, 1.15f, innerTopZ - 0.012f), new Vector3(1.85f, 1.05f, 0.035f), panelGlass);
            CreateCube("ObservationWindowFrameTop", environmentRoot, new Vector3(-1.65f, 1.70f, innerTopZ - 0.035f), new Vector3(2.02f, 0.055f, 0.055f), roomFrameColor);
            CreateCube("ObservationWindowFrameBottom", environmentRoot, new Vector3(-1.65f, 0.60f, innerTopZ - 0.035f), new Vector3(2.02f, 0.055f, 0.055f), roomFrameColor);
            CreateCube("ObservationWindowFrameLeft", environmentRoot, new Vector3(-2.60f, 1.15f, innerTopZ - 0.035f), new Vector3(0.055f, 1.16f, 0.055f), roomFrameColor);
            CreateCube("ObservationWindowFrameRight", environmentRoot, new Vector3(-0.70f, 1.15f, innerTopZ - 0.035f), new Vector3(0.055f, 1.16f, 0.055f), roomFrameColor);
        }

        private void BuildFloorLitterDetail(Vector2 roomSize)
        {
            float halfWidth = roomSize.x * 0.5f;
            float halfDepth = roomSize.y * 0.5f;
            Color strawLight = new Color(0.86f, 0.72f, 0.45f, 1.0f);
            Color strawDark = new Color(0.46f, 0.33f, 0.18f, 1.0f);

            for (int index = 0; index < 150; index++)
            {
                float x = Mathf.Lerp(-halfWidth + 0.42f, halfWidth - 0.42f, Hash01(index * 17 + 3));
                float z = Mathf.Lerp(-halfDepth + 0.42f, halfDepth - 0.42f, Hash01(index * 29 + 11));
                float length = Mathf.Lerp(0.045f, 0.18f, Hash01(index * 31 + 5));
                float width = Mathf.Lerp(0.010f, 0.032f, Hash01(index * 37 + 7));
                float rotation = Hash01(index * 43 + 13) * 180.0f;
                Color fleckColor = Color.Lerp(strawDark, strawLight, Hash01(index * 53 + 19));
                CreateCube(
                    "LitterFleck_" + index,
                    environmentRoot,
                    new Vector3(x, 0.076f, z),
                    new Vector3(width, 0.006f, length),
                    fleckColor,
                    new Vector3(0.0f, rotation, 0.0f)
                );
            }
        }

        private void BuildLowPerchRail(Vector2 roomSize)
        {
            float halfDepth = roomSize.y * 0.5f;
            float railZ = -halfDepth + 1.65f;
            Color railColor = new Color(0.82f, 0.83f, 0.80f, 1.0f);
            Vector3 start = new Vector3(-3.55f, 0.24f, railZ);
            Vector3 end = new Vector3(2.45f, 0.24f, railZ);
            CreateCableBetweenPoints("LowPerchRail_Main", environmentRoot, start, end, 0.055f, railColor);

            for (int index = 0; index < 4; index++)
            {
                float x = Mathf.Lerp(start.x + 0.42f, end.x - 0.42f, index / 3.0f);
                CreateCableBetweenPoints(
                    "LowPerchRail_Leg_" + index,
                    environmentRoot,
                    new Vector3(x, 0.075f, railZ),
                    new Vector3(x, 0.235f, railZ),
                    0.035f,
                    railColor
                );
            }
        }

        private void BuildCeilingFixtures(Vector2 roomSize, float wallHeight)
        {
            float ceilingY = wallHeight + 0.03f;
            CreateLedStrip("LedStrip_Back", new Vector3(-0.95f, ceilingY, 3.75f), 3.45f, -6.0f);
            CreateLedStrip("LedStrip_Front", new Vector3(1.55f, ceilingY, -3.35f), 3.05f, 3.0f);

            Color cableColor = new Color(0.08f, 0.085f, 0.09f, 1.0f);
            CreateCablePolyline(
                "CeilingCable_Left",
                new[]
                {
                    new Vector3(-4.95f, ceilingY - 0.12f, 4.8f),
                    new Vector3(-3.75f, ceilingY - 0.24f, 4.35f),
                    new Vector3(-2.15f, ceilingY - 0.18f, 4.05f),
                    new Vector3(-0.98f, ceilingY - 0.08f, 3.76f),
                },
                0.018f,
                cableColor
            );
            CreateCablePolyline(
                "CeilingCable_Right",
                new[]
                {
                    new Vector3(4.85f, ceilingY - 0.10f, -2.15f),
                    new Vector3(3.62f, ceilingY - 0.28f, -2.62f),
                    new Vector3(2.32f, ceilingY - 0.18f, -3.05f),
                    new Vector3(1.58f, ceilingY - 0.08f, -3.34f),
                },
                0.018f,
                cableColor
            );
        }

        private void BuildWallEquipment(Vector2 roomSize, float wallHeight)
        {
            float halfWidth = roomSize.x * 0.5f;
            float halfDepth = roomSize.y * 0.5f;
            float leftX = -halfWidth + 0.055f;
            float rightX = halfWidth - 0.055f;
            float topZ = halfDepth - 0.055f;
            float bottomZ = -halfDepth + 0.055f;
            Color equipmentBox = new Color(0.74f, 0.75f, 0.72f, 1.0f);
            Color equipmentDark = new Color(0.10f, 0.11f, 0.12f, 1.0f);
            Color cableColor = new Color(0.07f, 0.075f, 0.08f, 1.0f);

            CreateCube("TopWallControlBox", environmentRoot, new Vector3(1.95f, 1.58f, topZ - 0.005f), new Vector3(0.58f, 0.42f, 0.08f), equipmentBox);
            CreateCube("TopWallControlPanel", environmentRoot, new Vector3(1.95f, 1.57f, topZ - 0.055f), new Vector3(0.34f, 0.16f, 0.025f), equipmentDark);
            CreateCablePolyline(
                "TopWallControlCable",
                new[]
                {
                    new Vector3(1.95f, wallHeight - 0.15f, topZ - 0.06f),
                    new Vector3(1.72f, 2.05f, topZ - 0.06f),
                    new Vector3(1.95f, 1.82f, topZ - 0.06f),
                },
                0.016f,
                cableColor
            );

            CreateCube("LeftWallSensorBox", environmentRoot, new Vector3(leftX + 0.005f, 1.62f, 3.72f), new Vector3(0.08f, 0.34f, 0.46f), equipmentDark);
            Renderer blueIndicatorRenderer = CreateSphere("LeftWallBlueIndicator", environmentRoot, new Vector3(leftX + 0.055f, 1.72f, 3.58f), new Vector3(0.055f, 0.055f, 0.055f), new Color(0.18f, 0.34f, 0.90f, 1.0f)).GetComponent<Renderer>();
            SetRendererEmission(blueIndicatorRenderer, new Color(0.18f, 0.34f, 0.90f, 1.0f), 1.8f);
            CreateCablePolyline(
                "LeftWallSensorCable",
                new[]
                {
                    new Vector3(leftX + 0.02f, wallHeight - 0.10f, 4.45f),
                    new Vector3(leftX + 0.02f, 2.12f, 4.05f),
                    new Vector3(leftX + 0.02f, 1.82f, 3.72f),
                },
                0.016f,
                cableColor
            );

            CreateCube("RightWallCameraMount", environmentRoot, new Vector3(rightX - 0.005f, 1.28f, -2.78f), new Vector3(0.09f, 0.24f, 0.34f), equipmentDark);
            CreateCube("RightWallCameraLens", environmentRoot, new Vector3(rightX - 0.070f, 1.24f, -2.70f), new Vector3(0.055f, 0.11f, 0.16f), new Color(0.02f, 0.025f, 0.03f, 1.0f));
            CreateCablePolyline(
                "RightWallCameraCable",
                new[]
                {
                    new Vector3(rightX - 0.02f, wallHeight - 0.18f, -2.15f),
                    new Vector3(rightX - 0.02f, 1.85f, -2.52f),
                    new Vector3(rightX - 0.02f, 1.40f, -2.78f),
                },
                0.016f,
                cableColor
            );

            CreateCube("BottomWallSmallSensor", environmentRoot, new Vector3(-3.85f, 1.02f, bottomZ + 0.005f), new Vector3(0.34f, 0.22f, 0.08f), equipmentDark);
            CreateCableBetweenPoints("BottomWallSensorCable", environmentRoot, new Vector3(-3.85f, wallHeight - 0.28f, bottomZ + 0.02f), new Vector3(-3.85f, 1.16f, bottomZ + 0.02f), 0.014f, cableColor);
        }

        private void CreateLedStrip(string name, Vector3 center, float length, float yawDegrees)
        {
            Color housingColor = new Color(0.21f, 0.22f, 0.22f, 1.0f);
            Color lensColor = new Color(1.0f, 0.96f, 0.82f, 1.0f);
            Vector3 euler = new Vector3(0.0f, yawDegrees, 0.0f);
            Quaternion rotation = Quaternion.Euler(euler);
            Vector3 forward = rotation * Vector3.forward;

            CreateCube(name + "_Housing", environmentRoot, center, new Vector3(0.16f, 0.08f, length), housingColor, euler);
            Renderer lensRenderer = CreateCube(name + "_Lens", environmentRoot, center + (Vector3.down * 0.065f), new Vector3(0.13f, 0.025f, length * 0.92f), lensColor, euler).GetComponent<Renderer>();
            SetRendererEmission(lensRenderer, lensColor, 1.65f);

            int diodeCount = 12;
            for (int index = 0; index < diodeCount; index++)
            {
                float offset = (((index + 0.5f) / diodeCount) - 0.5f) * length * 0.78f;
                Renderer diodeRenderer = CreateCube(
                    name + "_Diode_" + index,
                    environmentRoot,
                    center + (Vector3.down * 0.086f) + (forward * offset),
                    new Vector3(0.105f, 0.014f, 0.075f),
                    lensColor,
                    euler
                ).GetComponent<Renderer>();
                SetRendererEmission(diodeRenderer, lensColor, 2.1f);
            }

            AddPointLight(name + "_Glow", center + (Vector3.down * 0.45f), lensColor, 1.1f, 6.0f);
            CreateCableBetweenPoints(name + "_CordA", environmentRoot, center - (forward * length * 0.40f), center - (forward * length * 0.40f) + (Vector3.down * 0.26f), 0.018f, new Color(0.08f, 0.085f, 0.09f, 1.0f));
            CreateCableBetweenPoints(name + "_CordB", environmentRoot, center + (forward * length * 0.40f), center + (forward * length * 0.40f) + (Vector3.down * 0.22f), 0.018f, new Color(0.08f, 0.085f, 0.09f, 1.0f));
        }

        private void BuildSemanticZoneVisuals()
        {
            PoultryTwinRoomLayout.ZoneProfile[] profiles = PoultryTwinRoomLayout.GetRoom1Profiles();
            for (int profileIndex = 0; profileIndex < profiles.Length; profileIndex++)
            {
                PoultryTwinRoomLayout.ZoneProfile profile = profiles[profileIndex];
                GameObject zoneObject = new GameObject(profile.SemanticZoneId + "_Zone");
                zoneObject.transform.SetParent(zoneRoot, false);

                List<ZoneSurfaceRecord> surfaces = new List<ZoneSurfaceRecord>();
                Rect[] worldRects = profile.WorldRects;
                for (int rectIndex = 0; rectIndex < worldRects.Length; rectIndex++)
                {
                    surfaces.Add(BuildZoneSurface(zoneObject.transform, profile, worldRects[rectIndex], rectIndex));
                }

                zoneVisuals[profile.SemanticZoneId] = new ZoneVisualRecord(
                    profile.DisplayName,
                    profile.AccentColor,
                    surfaces.ToArray()
                );
            }
        }

        private ZoneSurfaceRecord BuildZoneSurface(
            Transform parent,
            PoultryTwinRoomLayout.ZoneProfile profile,
            Rect worldRect,
            int surfaceIndex)
        {
            Vector3 fillPosition = RectToCenter(worldRect);
            Vector3 fillScale = new Vector3(worldRect.width, surfaceHeight, worldRect.height);
            GameObject fill = CreateCube(profile.SemanticZoneId + "_Fill_" + surfaceIndex, parent, fillPosition, fillScale, inactiveZoneColor);
            EnableCollider(fill);
            ZoneSurfaceMarker marker = fill.AddComponent<ZoneSurfaceMarker>();
            marker.ZoneId = profile.SemanticZoneId;

            float outlineHeight = surfaceHeight * 1.15f;
            float outlineY = overlayLift + (outlineHeight * 0.5f);
            Renderer[] outlines =
            {
                CreateCube(profile.SemanticZoneId + "_Top_" + surfaceIndex, parent, new Vector3(worldRect.center.x, outlineY, worldRect.yMax), new Vector3(worldRect.width + outlineThickness, outlineHeight, outlineThickness), profile.AccentColor).GetComponent<Renderer>(),
                CreateCube(profile.SemanticZoneId + "_Bottom_" + surfaceIndex, parent, new Vector3(worldRect.center.x, outlineY, worldRect.yMin), new Vector3(worldRect.width + outlineThickness, outlineHeight, outlineThickness), profile.AccentColor).GetComponent<Renderer>(),
                CreateCube(profile.SemanticZoneId + "_Left_" + surfaceIndex, parent, new Vector3(worldRect.xMin, outlineY, worldRect.center.y), new Vector3(outlineThickness, outlineHeight, worldRect.height + outlineThickness), profile.AccentColor).GetComponent<Renderer>(),
                CreateCube(profile.SemanticZoneId + "_Right_" + surfaceIndex, parent, new Vector3(worldRect.xMax, outlineY, worldRect.center.y), new Vector3(outlineThickness, outlineHeight, worldRect.height + outlineThickness), profile.AccentColor).GetComponent<Renderer>(),
            };

            Renderer fillRenderer = fill.GetComponent<Renderer>();
            ConfigureOverlayRenderer(fillRenderer);
            for (int index = 0; index < outlines.Length; index++)
            {
                ConfigureOverlayRenderer(outlines[index]);
            }

            return new ZoneSurfaceRecord(fillRenderer, outlines);
        }

        private void BuildBirdPool(PoultryTwinTimelineFile data)
        {
            int birdCount = 30;
            PoultryTwinRoom room = data != null ? data.GetPrimaryRoom() : null;
            if (data != null && data.timeline != null && data.timeline.Length > 0 && data.timeline[0] != null && data.timeline[0].bird_count > 0)
            {
                birdCount = data.timeline[0].bird_count;
            }
            else if (room == null)
            {
                birdCount = 30;
            }

            GameObject[] availableChickenPrefabs = ResolveChickenPrefabs();
            bool hasChickenModels = useChickenModel && availableChickenPrefabs.Length > 0;
            for (int index = 0; index < birdCount; index++)
            {
                GameObject birdObject = new GameObject("BirdProxy_" + (index + 1).ToString("00"));
                birdObject.transform.SetParent(birdRoot, false);
                birdObject.transform.localPosition = new Vector3(0.0f, -10.0f, 0.0f);
                birdObject.transform.localRotation = Quaternion.identity;
                birdObject.transform.localScale = Vector3.one;

                Renderer[] rendererComponents;
                Animator animatorComponent = null;
                Transform animatorRootTransform = null;
                float animatorBaseSpeed = 1.0f;
                bool usesChickenModel = false;

                if (hasChickenModels)
                {
                    GameObject prefab = availableChickenPrefabs[index % availableChickenPrefabs.Length];
                    GameObject chickenModel = Instantiate(prefab, birdObject.transform);
                    chickenModel.name = "ChickenModel";
                    chickenModel.transform.localPosition = Vector3.zero;
                    chickenModel.transform.localRotation = Quaternion.Euler(chickenModelEulerOffset);
                    chickenModel.transform.localScale = Vector3.one * chickenModelScale;

                    DisableCollidersInChildren(chickenModel);
                    rendererComponents = chickenModel.GetComponentsInChildren<Renderer>(true);
                    for (int rendererIndex = 0; rendererIndex < rendererComponents.Length; rendererIndex++)
                    {
                        RepairChickenRendererMaterials(rendererComponents[rendererIndex]);
                        ConfigureOverlayRenderer(rendererComponents[rendererIndex]);
                    }

                    animatorComponent = chickenModel.GetComponentInChildren<Animator>();
                    if (animatorComponent != null)
                    {
                        animatorComponent.applyRootMotion = displayFirstZoneProxyMode && useAnimatorRootMotionForDisplayMovement;
                        animatorRootTransform = animatorComponent.transform;
                        animatorBaseSpeed = 0.92f + ((index % 5) * 0.04f);
                        animatorComponent.speed = animatorBaseSpeed;
                    }

                    usesChickenModel = true;
                }
                else
                {
                    GameObject sphereObject = CreateSphere("BirdProxySphere", birdObject.transform, Vector3.zero, Vector3.one * birdScale, new Color(0.72f, 0.74f, 0.76f, 0.95f));
                    Renderer rendererComponent = sphereObject.GetComponent<Renderer>();
                    ConfigureOverlayRenderer(rendererComponent);
                    rendererComponents = rendererComponent != null ? new[] { rendererComponent } : new Renderer[0];
                }

                birdObject.SetActive(false);
                birdVisuals.Add(new BirdVisualRecord(birdObject, rendererComponents, animatorComponent, animatorRootTransform, usesChickenModel, animatorBaseSpeed));
            }
        }

        private GameObject[] ResolveChickenPrefabs()
        {
            List<GameObject> prefabs = new List<GameObject>();
            if (chickenPrefabs != null)
            {
                for (int index = 0; index < chickenPrefabs.Length; index++)
                {
                    GameObject prefab = chickenPrefabs[index];
                    if (prefab != null && !prefabs.Contains(prefab))
                    {
                        prefabs.Add(prefab);
                    }
                }
            }

            if (useChickenModel && chickenResourcePaths != null)
            {
                for (int index = 0; index < chickenResourcePaths.Length; index++)
                {
                    string resourcePath = chickenResourcePaths[index];
                    if (string.IsNullOrEmpty(resourcePath))
                    {
                        continue;
                    }

                    GameObject prefab = Resources.Load<GameObject>(resourcePath);
                    if (prefab != null && !prefabs.Contains(prefab))
                    {
                        prefabs.Add(prefab);
                    }
                }
            }

            if (useChickenModel && prefabs.Count == 0)
            {
                Debug.LogWarning("No chicken prefab found in Resources/PoultryTwin. Falling back to sphere bird proxies.");
            }

            return prefabs.ToArray();
        }

        private void UpdateBirdVisuals(PoultryTwinTimelineFrame frame, float transitionSeconds)
        {
            for (int index = 0; index < birdVisuals.Count; index++)
            {
                BirdVisualRecord record = birdVisuals[index];
                record.AssignedThisFrame = false;
            }

            if (frame == null || frame.birds == null)
            {
                currentFrameBirdKeys.Clear();
                DisableUnassignedBirds();
                return;
            }

            currentFrameBirdKeys.Clear();
            Dictionary<string, int> zoneBirdCounts = new Dictionary<string, int>();
            Dictionary<string, int> zoneSlotCounters = new Dictionary<string, int>();
            for (int index = 0; index < frame.birds.Length; index++)
            {
                PoultryTwinBirdFrame birdFrame = frame.birds[index];
                if (birdFrame == null)
                {
                    continue;
                }

                string semanticZoneId = GetSemanticZoneId(birdFrame.zone_id);
                IncrementCounter(zoneBirdCounts, semanticZoneId);
            }

            displayedWingFlapsThisFrame = 0;
            for (int index = 0; index < frame.birds.Length; index++)
            {
                PoultryTwinBirdFrame birdFrame = frame.birds[index];
                if (birdFrame == null)
                {
                    continue;
                }

                string semanticZoneId = GetSemanticZoneId(birdFrame.zone_id);
                int zoneSlotIndex = GetAndIncrementCounter(zoneSlotCounters, semanticZoneId);
                currentFrameBirdKeys.Add(GetBirdVisualKey(birdFrame, index, semanticZoneId, zoneSlotIndex));
            }

            int feederBirdCount = 0;
            int drinkerBirdCount = 0;
            for (int index = 0; index < frame.birds.Length; index++)
            {
                string resourceZoneId = displayFirstZoneProxyMode
                    ? GetSemanticZoneId(frame.birds[index] != null ? frame.birds[index].zone_id : string.Empty)
                    : GetResourceZoneForBird(frame.birds[index]);
                if (resourceZoneId == PoultryTwinRoomLayout.FeederZoneId)
                {
                    feederBirdCount++;
                }
                else if (resourceZoneId == PoultryTwinRoomLayout.DrinkerZoneId)
                {
                    drinkerBirdCount++;
                }
            }

            int feederSlotIndex = 0;
            int drinkerSlotIndex = 0;
            zoneSlotCounters.Clear();
            for (int index = 0; index < frame.birds.Length; index++)
            {
                PoultryTwinBirdFrame birdFrame = frame.birds[index];
                if (birdFrame == null)
                {
                    continue;
                }

                float birdHeight = useChickenModel ? chickenModelHeight : birdFloorHeight;
                Vector3 targetPosition = new Vector3(birdFrame.world_x, birdHeight, birdFrame.world_z);
                Vector3 facingPoint = targetPosition + Vector3.forward;
                bool hasFacingPoint = false;
                string semanticZoneId = GetSemanticZoneId(birdFrame.zone_id);
                int zoneSlotIndex = GetAndIncrementCounter(zoneSlotCounters, semanticZoneId);
                int zoneBirdCount = 0;
                zoneBirdCounts.TryGetValue(semanticZoneId, out zoneBirdCount);

                if (displayFirstZoneProxyMode)
                {
                    hasFacingPoint = TryGetDisplayZoneTarget(
                        semanticZoneId,
                        zoneSlotIndex,
                        zoneBirdCount,
                        birdHeight,
                        index,
                        frame.frame_index,
                        birdFrame.behavior,
                        out targetPosition,
                        out facingPoint
                    );
                }

                string resourceZoneId = displayFirstZoneProxyMode ? semanticZoneId : GetResourceZoneForBird(birdFrame);
                if (!displayFirstZoneProxyMode && resourceZoneId == PoultryTwinRoomLayout.FeederZoneId)
                {
                    Vector3 ringTarget;
                    Vector3 resourceFacingPoint;
                    hasFacingPoint = TryGetResourceRingTarget(resourceZoneId, feederSlotIndex, feederBirdCount, birdHeight, index, out ringTarget, out resourceFacingPoint);
                    if (hasFacingPoint)
                    {
                        targetPosition = ringTarget;
                        facingPoint = resourceFacingPoint;
                    }

                    feederSlotIndex++;
                }
                else if (!displayFirstZoneProxyMode && resourceZoneId == PoultryTwinRoomLayout.DrinkerZoneId)
                {
                    Vector3 ringTarget;
                    Vector3 resourceFacingPoint;
                    hasFacingPoint = TryGetResourceRingTarget(resourceZoneId, drinkerSlotIndex, drinkerBirdCount, birdHeight, index, out ringTarget, out resourceFacingPoint);
                    if (hasFacingPoint)
                    {
                        targetPosition = ringTarget;
                        facingPoint = resourceFacingPoint;
                    }

                    drinkerSlotIndex++;
                }

                string visualKey = GetBirdVisualKey(birdFrame, index, semanticZoneId, zoneSlotIndex);
                bool matchedByKey;
                bool reusedByProximity;
                BirdVisualRecord visual = ResolveBirdVisual(visualKey, targetPosition, out matchedByKey, out reusedByProximity);
                if (visual == null)
                {
                    break;
                }

                if (!hasFacingPoint)
                {
                    facingPoint = targetPosition + visual.GameObject.transform.forward;
                }

                bool identityChanged = visual.TrackKey != visualKey;
                AssignBirdVisualKey(visual, visualKey);
                visual.AssignedThisFrame = true;
                visual.ZoneId = semanticZoneId;
                visual.Behavior = GetDisplayAnimationBehavior(birdFrame.behavior, semanticZoneId, frame.frame_index);

                visual.TargetPosition = targetPosition;
                visual.FacingPoint = facingPoint;
                visual.HasFacingPoint = hasFacingPoint;
                visual.HasTarget = true;

                bool updateTransition = transitionSeconds >= 0.0f;
                float reassignmentDistance = visual.HasActivePosition
                    ? Vector3.Distance(visual.GameObject.transform.localPosition, targetPosition)
                    : 0.0f;
                bool shouldSnap =
                    (identityChanged && !matchedByKey && !reusedByProximity) ||
                    !visual.GameObject.activeSelf ||
                    !visual.HasActivePosition ||
                    transitionSeconds <= 0.0f ||
                    (!matchedByKey && !reusedByProximity && reassignmentDistance > Mathf.Max(0.25f, birdTeleportDistance));
                if (shouldSnap)
                {
                    visual.GameObject.transform.localPosition = targetPosition;
                    visual.TransitionStartPosition = targetPosition;
                    visual.TransitionElapsed = 0.0f;
                    visual.TransitionDuration = 0.0f;
                    visual.HasActivePosition = true;
                    visual.LastMotionDelta = Vector3.zero;
                    ResetAnimatorRootTransform(visual);
                }
                else if (updateTransition)
                {
                    visual.TransitionStartPosition = visual.GameObject.transform.localPosition;
                    visual.TransitionElapsed = 0.0f;
                    visual.TransitionDuration = Mathf.Max(0.05f, transitionSeconds);
                }
                else
                {
                    visual.TransitionStartPosition = visual.GameObject.transform.localPosition;
                    visual.TransitionElapsed = 0.0f;
                    visual.TransitionDuration = 0.0f;
                }

                visual.GameObject.SetActive(true);

                bool isMoving = !shouldSnap && Vector3.Distance(visual.GameObject.transform.localPosition, visual.TargetPosition) > birdWalkAnimationThreshold;
                UpdateBirdAnimation(visual, isMoving);

                if (!visual.UsesChickenModel)
                {
                    Color birdColor = GetBirdColor(visual.Behavior);
                    bool dimBird = HasSelectedZone && semanticZoneId != selectedZoneId;
                    if (dimBird)
                    {
                        birdColor = Color.Lerp(birdColor, new Color(0.68f, 0.70f, 0.72f, 0.65f), 0.68f);
                    }

                    ApplyMaterialColor(visual.RendererComponents, birdColor, 0.16f);
                }
            }

            DisableUnassignedBirds();
        }

        private string GetBirdVisualKey(PoultryTwinBirdFrame birdFrame, int fallbackIndex, string semanticZoneId, int zoneSlotIndex)
        {
            if (displayFirstZoneProxyMode)
            {
                string zoneKey = string.IsNullOrEmpty(semanticZoneId) ? "unknown" : semanticZoneId;
                return "display:" + zoneKey + ":" + zoneSlotIndex.ToString("00");
            }

            return GetBirdVisualKey(birdFrame, fallbackIndex);
        }

        private string GetBirdVisualKey(PoultryTwinBirdFrame birdFrame, int fallbackIndex)
        {
            if (birdFrame != null)
            {
                if (!string.IsNullOrEmpty(birdFrame.track_id))
                {
                    return "track:" + birdFrame.track_id;
                }

                if (!string.IsNullOrEmpty(birdFrame.bird_id))
                {
                    return "bird:" + birdFrame.bird_id;
                }
            }

            return "slot:" + fallbackIndex.ToString("00");
        }

        private string GetSemanticZoneId(string zoneId)
        {
            string semanticZoneId;
            if (!string.IsNullOrEmpty(zoneId) && PoultryTwinRoomLayout.TryMapExternalZoneId(zoneId, out semanticZoneId))
            {
                return semanticZoneId;
            }

            return string.IsNullOrEmpty(zoneId) ? "unknown" : zoneId;
        }

        private void IncrementCounter(Dictionary<string, int> counterLookup, string key)
        {
            if (counterLookup == null)
            {
                return;
            }

            if (string.IsNullOrEmpty(key))
            {
                key = "unknown";
            }

            int value;
            counterLookup.TryGetValue(key, out value);
            counterLookup[key] = value + 1;
        }

        private int GetAndIncrementCounter(Dictionary<string, int> counterLookup, string key)
        {
            if (counterLookup == null)
            {
                return 0;
            }

            if (string.IsNullOrEmpty(key))
            {
                key = "unknown";
            }

            int value;
            counterLookup.TryGetValue(key, out value);
            counterLookup[key] = value + 1;
            return value;
        }

        private BirdVisualRecord ResolveBirdVisual(string visualKey, Vector3 targetPosition, out bool matchedByKey, out bool reusedByProximity)
        {
            matchedByKey = false;
            reusedByProximity = false;
            if (!string.IsNullOrEmpty(visualKey))
            {
                BirdVisualRecord existingRecord;
                if (birdVisualLookup.TryGetValue(visualKey, out existingRecord) && existingRecord != null && !existingRecord.AssignedThisFrame)
                {
                    matchedByKey = true;
                    return existingRecord;
                }
            }

            BirdVisualRecord nearestReusableRecord = null;
            float nearestReusableDistance = float.MaxValue;
            for (int index = 0; index < birdVisuals.Count; index++)
            {
                BirdVisualRecord candidate = birdVisuals[index];
                if (candidate.AssignedThisFrame || !candidate.GameObject.activeSelf || !candidate.HasActivePosition || IsReservedForCurrentFrame(candidate))
                {
                    continue;
                }

                float distance = Vector3.Distance(candidate.GameObject.transform.localPosition, targetPosition);
                if (distance < nearestReusableDistance)
                {
                    nearestReusableDistance = distance;
                    nearestReusableRecord = candidate;
                }
            }

            if (nearestReusableRecord != null && nearestReusableDistance <= GetBirdIdentityReuseDistance())
            {
                reusedByProximity = true;
                return nearestReusableRecord;
            }

            for (int index = 0; index < birdVisuals.Count; index++)
            {
                BirdVisualRecord candidate = birdVisuals[index];
                if (!candidate.AssignedThisFrame && !candidate.GameObject.activeSelf)
                {
                    return candidate;
                }
            }

            for (int index = 0; index < birdVisuals.Count; index++)
            {
                BirdVisualRecord candidate = birdVisuals[index];
                if (!candidate.AssignedThisFrame && !IsReservedForCurrentFrame(candidate))
                {
                    return candidate;
                }
            }

            return null;
        }

        private float GetBirdIdentityReuseDistance()
        {
            return Mathf.Max(0.05f, displayFirstZoneProxyMode ? displayZoneIdentityReuseDistance : birdIdentityReuseDistance);
        }

        private bool IsReservedForCurrentFrame(BirdVisualRecord candidate)
        {
            return candidate != null && !string.IsNullOrEmpty(candidate.TrackKey) && currentFrameBirdKeys.Contains(candidate.TrackKey);
        }

        private void AssignBirdVisualKey(BirdVisualRecord visual, string visualKey)
        {
            if (visual == null || visual.TrackKey == visualKey)
            {
                return;
            }

            if (!string.IsNullOrEmpty(visual.TrackKey))
            {
                BirdVisualRecord mappedRecord;
                if (birdVisualLookup.TryGetValue(visual.TrackKey, out mappedRecord) && mappedRecord == visual)
                {
                    birdVisualLookup.Remove(visual.TrackKey);
                }
            }

            visual.TrackKey = visualKey;
            if (!string.IsNullOrEmpty(visualKey))
            {
                birdVisualLookup[visualKey] = visual;
            }
        }

        private void DisableUnassignedBirds()
        {
            for (int index = 0; index < birdVisuals.Count; index++)
            {
                BirdVisualRecord record = birdVisuals[index];
                if (record.AssignedThisFrame)
                {
                    continue;
                }

                record.HasTarget = false;
                record.HasFacingPoint = false;
                record.HasActivePosition = false;
                record.LastMotionDelta = Vector3.zero;
                record.Behavior = string.Empty;
                record.ZoneId = string.Empty;
                AssignBirdVisualKey(record, string.Empty);
                ResetAnimatorRootTransform(record);
                if (record.GameObject.activeSelf)
                {
                    record.GameObject.SetActive(false);
                }
            }
        }

        private string GetResourceZoneForBird(PoultryTwinBirdFrame birdFrame)
        {
            if (birdFrame == null)
            {
                return string.Empty;
            }

            string zoneId = birdFrame.zone_id ?? string.Empty;
            string behavior = birdFrame.behavior ?? string.Empty;
            string semanticZoneId;
            if (!PoultryTwinRoomLayout.TryMapExternalZoneId(zoneId, out semanticZoneId))
            {
                semanticZoneId = zoneId;
            }

            if (semanticZoneId == PoultryTwinRoomLayout.FeederZoneId || behavior.ToLowerInvariant() == "feeding")
            {
                return PoultryTwinRoomLayout.FeederZoneId;
            }

            if (semanticZoneId == PoultryTwinRoomLayout.DrinkerZoneId || behavior.ToLowerInvariant() == "drinking")
            {
                return PoultryTwinRoomLayout.DrinkerZoneId;
            }

            return string.Empty;
        }

        private bool TryGetResourceRingTarget(
            string resourceZoneId,
            int slotIndex,
            int totalSlots,
            float birdHeight,
            int visualIndex,
            out Vector3 targetPosition,
            out Vector3 facingPoint)
        {
            targetPosition = Vector3.zero;
            facingPoint = Vector3.zero;
            PoultryTwinRoomLayout.ZoneProfile profile = PoultryTwinRoomLayout.GetProfileOrDefault(resourceZoneId);
            Rect resourceRect = GetLargestWorldRect(profile.WorldRects);
            if (resourceRect.width <= 0.0f || resourceRect.height <= 0.0f)
            {
                return false;
            }

            Vector3 center = new Vector3(resourceRect.center.x, birdHeight, resourceRect.center.y);
            int slotsInRing = Mathf.Clamp(totalSlots, 6, 12);
            int ringIndex = slotsInRing > 0 ? slotIndex / slotsInRing : 0;
            int ringSlot = slotsInRing > 0 ? slotIndex % slotsInRing : 0;
            float baseRadius = resourceZoneId == PoultryTwinRoomLayout.FeederZoneId ? 0.74f : 0.62f;
            float radius = baseRadius + (ringIndex * 0.34f);
            float angleOffset = resourceZoneId == PoultryTwinRoomLayout.FeederZoneId ? 0.26f : -0.14f;
            float angle = angleOffset + ((Mathf.PI * 2.0f) * ringSlot / Mathf.Max(1, slotsInRing));
            float microOffset = (((visualIndex * 37) % 11) - 5) * 0.012f;
            radius += microOffset;

            targetPosition = new Vector3(
                center.x + (Mathf.Cos(angle) * radius),
                birdHeight,
                center.z + (Mathf.Sin(angle) * radius)
            );
            facingPoint = center;
            return true;
        }

        private bool TryGetDisplayZoneTarget(
            string semanticZoneId,
            int slotIndex,
            int totalSlots,
            float birdHeight,
            int visualIndex,
            int frameIndex,
            string behavior,
            out Vector3 targetPosition,
            out Vector3 facingPoint)
        {
            targetPosition = Vector3.zero;
            facingPoint = Vector3.zero;
            if (semanticZoneId == PoultryTwinRoomLayout.FeederZoneId || semanticZoneId == PoultryTwinRoomLayout.DrinkerZoneId)
            {
                return TryGetResourceRingTarget(semanticZoneId, slotIndex, totalSlots, birdHeight, visualIndex, out targetPosition, out facingPoint);
            }

            PoultryTwinRoomLayout.ZoneProfile profile = PoultryTwinRoomLayout.GetProfileOrDefault(semanticZoneId);
            if (profile.WorldRects == null || profile.WorldRects.Length == 0)
            {
                return false;
            }

            int rectIndex = GetDisplayRectIndexForSlot(profile.WorldRects, slotIndex, totalSlots);
            Rect rect = profile.WorldRects[Mathf.Clamp(rectIndex, 0, profile.WorldRects.Length - 1)];
            int localSlotIndex = 0;
            int localSlotCount = 0;
            for (int index = 0; index < Mathf.Max(1, totalSlots); index++)
            {
                if (GetDisplayRectIndexForSlot(profile.WorldRects, index, totalSlots) != rectIndex)
                {
                    continue;
                }

                if (index < slotIndex)
                {
                    localSlotIndex++;
                }

                localSlotCount++;
            }

            localSlotCount = Mathf.Max(1, localSlotCount);
            float aspect = Mathf.Max(0.35f, Mathf.Abs(rect.width) / Mathf.Max(0.35f, Mathf.Abs(rect.height)));
            int columns = Mathf.Max(1, Mathf.CeilToInt(Mathf.Sqrt(localSlotCount * aspect)));
            int rows = Mathf.Max(1, Mathf.CeilToInt((float)localSlotCount / columns));
            int column = localSlotIndex % columns;
            int row = localSlotIndex / columns;

            float marginX = Mathf.Min(0.48f, Mathf.Abs(rect.width) * 0.18f);
            float marginZ = Mathf.Min(0.48f, Mathf.Abs(rect.height) * 0.18f);
            float usableWidth = Mathf.Max(0.12f, Mathf.Abs(rect.width) - (marginX * 2.0f));
            float usableDepth = Mathf.Max(0.12f, Mathf.Abs(rect.height) - (marginZ * 2.0f));
            float x = rect.xMin + marginX + (((column + 0.5f) / columns) * usableWidth);
            float z = rect.yMin + marginZ + (((row + 0.5f) / rows) * usableDepth);

            float jitterX = (Hash01((visualIndex * 97) + (slotIndex * 19) + 11) - 0.5f) * displayZoneSlotJitter;
            float jitterZ = (Hash01((visualIndex * 101) + (slotIndex * 23) + 17) - 0.5f) * displayZoneSlotJitter;
            string normalizedBehavior = (behavior ?? string.Empty).ToLowerInvariant();
            if (normalizedBehavior == "active" || semanticZoneId == PoultryTwinRoomLayout.OpenMovementZoneId)
            {
                float phase = (frameIndex * 0.37f) + (slotIndex * 1.71f) + visualIndex;
                jitterX += Mathf.Sin(phase) * displayActiveWanderRadius;
                jitterZ += Mathf.Cos(phase * 0.73f) * displayActiveWanderRadius;
            }

            x = Mathf.Clamp(x + jitterX, rect.xMin + 0.18f, rect.xMax - 0.18f);
            z = Mathf.Clamp(z + jitterZ, rect.yMin + 0.18f, rect.yMax - 0.18f);
            targetPosition = new Vector3(x, birdHeight, z);

            float facingPhase = (slotIndex * 1.37f) + (frameIndex * 0.08f);
            facingPoint = targetPosition + new Vector3(Mathf.Sin(facingPhase), 0.0f, Mathf.Cos(facingPhase));
            return true;
        }

        private int GetDisplayRectIndexForSlot(Rect[] rects, int slotIndex, int totalSlots)
        {
            if (rects == null || rects.Length == 0)
            {
                return 0;
            }

            float totalArea = 0.0f;
            for (int index = 0; index < rects.Length; index++)
            {
                totalArea += Mathf.Abs(rects[index].width * rects[index].height);
            }

            if (totalArea <= 0.0001f)
            {
                return 0;
            }

            float slotAreaPosition = (((slotIndex + 0.5f) / Mathf.Max(1, totalSlots)) * totalArea);
            float cumulativeArea = 0.0f;
            for (int index = 0; index < rects.Length; index++)
            {
                cumulativeArea += Mathf.Abs(rects[index].width * rects[index].height);
                if (slotAreaPosition <= cumulativeArea)
                {
                    return index;
                }
            }

            return rects.Length - 1;
        }

        private string GetDisplayAnimationBehavior(string behavior, string semanticZoneId, int frameIndex)
        {
            string normalizedBehavior = (behavior ?? string.Empty).ToLowerInvariant();
            if (!displayFirstZoneProxyMode || normalizedBehavior != "wing_flapping")
            {
                return normalizedBehavior;
            }

            if (semanticZoneId == PoultryTwinRoomLayout.FeederZoneId || semanticZoneId == PoultryTwinRoomLayout.DrinkerZoneId)
            {
                return "idle";
            }

            if (displayedWingFlapsThisFrame >= Mathf.Max(0, displayMaxWingFlapsPerFrame))
            {
                return "active";
            }

            if (frameIndex <= lastDisplayedWingFlapFrameIndex)
            {
                lastDisplayedWingFlapFrameIndex = frameIndex - Mathf.Max(1, displayWingFlapFrameGap) - 1;
            }

            if (frameIndex - lastDisplayedWingFlapFrameIndex < Mathf.Max(1, displayWingFlapFrameGap))
            {
                return "active";
            }

            displayedWingFlapsThisFrame++;
            lastDisplayedWingFlapFrameIndex = frameIndex;
            return "wing_flapping";
        }

        private void UpdateBirdMotion(float deltaTime)
        {
            if (deltaTime <= 0.0f)
            {
                return;
            }

            for (int index = 0; index < birdVisuals.Count; index++)
            {
                BirdVisualRecord record = birdVisuals[index];
                if (record == null || !record.GameObject.activeSelf || !record.HasTarget)
                {
                    continue;
                }

                if (ShouldUseAnimatorRootMotionMovement(record))
                {
                    UpdateRootMotionBirdIntent(record, deltaTime);
                    continue;
                }

                ResetAnimatorRootTransform(record);
                Vector3 currentPosition = record.GameObject.transform.localPosition;
                Vector3 targetPosition = record.TargetPosition;
                Vector3 previousPosition = currentPosition;
                if (record.TransitionDuration > 0.0f)
                {
                    record.TransitionElapsed = Mathf.Min(record.TransitionElapsed + deltaTime, record.TransitionDuration);
                    float transitionT = Mathf.Clamp01(record.TransitionElapsed / record.TransitionDuration);
                    transitionT = transitionT * transitionT * (3.0f - (2.0f * transitionT));
                    currentPosition = Vector3.Lerp(record.TransitionStartPosition, targetPosition, transitionT);
                    record.GameObject.transform.localPosition = currentPosition;
                }

                else
                {
                    Vector3 remainingDelta = targetPosition - currentPosition;
                    float remainingDistance = remainingDelta.magnitude;
                    if (remainingDistance > Mathf.Max(0.25f, birdTeleportDistance))
                    {
                        currentPosition = targetPosition;
                    }
                    else if (remainingDistance > 0.001f)
                    {
                        currentPosition = Vector3.MoveTowards(
                            currentPosition,
                            targetPosition,
                            Mathf.Max(0.05f, birdMoveSpeed) * deltaTime
                        );
                    }

                    record.GameObject.transform.localPosition = currentPosition;
                }

                Vector3 planarDelta = new Vector3(targetPosition.x - currentPosition.x, 0.0f, targetPosition.z - currentPosition.z);
                Vector3 motionDelta = new Vector3(currentPosition.x - previousPosition.x, 0.0f, currentPosition.z - previousPosition.z);
                bool isMoving =
                    motionDelta.sqrMagnitude > 0.00001f ||
                    planarDelta.sqrMagnitude > birdWalkAnimationThreshold * birdWalkAnimationThreshold;
                Vector3 facingDelta = Vector3.zero;
                if (record.HasFacingPoint && planarDelta.sqrMagnitude < 0.12f)
                {
                    facingDelta = new Vector3(record.FacingPoint.x - currentPosition.x, 0.0f, record.FacingPoint.z - currentPosition.z);
                }
                else if (isMoving)
                {
                    facingDelta = motionDelta;
                    if (facingDelta.sqrMagnitude < 0.0001f)
                    {
                        facingDelta = planarDelta;
                    }
                }
                else if (record.HasFacingPoint)
                {
                    facingDelta = new Vector3(record.FacingPoint.x - currentPosition.x, 0.0f, record.FacingPoint.z - currentPosition.z);
                }

                if (facingDelta.sqrMagnitude > 0.0001f)
                {
                    Quaternion targetRotation = Quaternion.LookRotation(facingDelta.normalized, Vector3.up);
                    record.GameObject.transform.localRotation = Quaternion.RotateTowards(
                        record.GameObject.transform.localRotation,
                        targetRotation,
                        Mathf.Max(30.0f, birdTurnSpeed) * deltaTime
                    );
                }

                float planarMotionSpeed = motionDelta.magnitude / Mathf.Max(0.001f, deltaTime);
                record.LastMotionDelta = motionDelta;
                UpdateBirdAnimation(record, isMoving, planarMotionSpeed);
            }
        }

        private void UpdateRootMotionBirdIntent(BirdVisualRecord record, float deltaTime)
        {
            if (record == null || record.GameObject == null)
            {
                return;
            }

            Vector3 currentPosition = record.GameObject.transform.localPosition;
            Vector3 targetPosition = record.TargetPosition;
            Vector3 planarDelta = new Vector3(targetPosition.x - currentPosition.x, 0.0f, targetPosition.z - currentPosition.z);
            float planarDistance = planarDelta.magnitude;

            record.TransitionElapsed = 0.0f;
            record.TransitionDuration = 0.0f;
            record.TransitionStartPosition = currentPosition;

            if (planarDistance > Mathf.Max(0.25f, birdTeleportDistance))
            {
                record.GameObject.transform.localPosition = targetPosition;
                record.LastMotionDelta = Vector3.zero;
                ResetAnimatorRootTransform(record);
                UpdateBirdAnimation(record, false, 0.0f);
                return;
            }

            bool isMoving = planarDistance > Mathf.Max(0.02f, rootMotionArrivalDistance);
            Vector3 facingDelta = Vector3.zero;
            if (isMoving)
            {
                facingDelta = planarDelta;
            }
            else if (record.HasFacingPoint)
            {
                facingDelta = new Vector3(record.FacingPoint.x - currentPosition.x, 0.0f, record.FacingPoint.z - currentPosition.z);
            }

            if (facingDelta.sqrMagnitude > 0.0001f)
            {
                Quaternion targetRotation = Quaternion.LookRotation(facingDelta.normalized, Vector3.up);
                record.GameObject.transform.localRotation = Quaternion.RotateTowards(
                    record.GameObject.transform.localRotation,
                    targetRotation,
                    Mathf.Max(30.0f, birdTurnSpeed) * deltaTime
                );
            }

            float planarMotionSpeed = record.LastMotionDelta.magnitude / Mathf.Max(0.001f, deltaTime);
            UpdateBirdAnimation(record, isMoving, planarMotionSpeed);
        }

        private void ApplyAnimatorRootMotionToProxy(BirdVisualRecord record)
        {
            if (record == null || record.AnimatorRootTransform == null)
            {
                return;
            }

            if (!ShouldUseAnimatorRootMotionMovement(record) || !record.GameObject.activeSelf)
            {
                record.LastMotionDelta = Vector3.zero;
                ResetAnimatorRootTransform(record);
                return;
            }

            Vector3 currentPosition = record.GameObject.transform.localPosition;
            Vector3 targetPosition = record.TargetPosition;
            Vector3 planarTargetDelta = new Vector3(targetPosition.x - currentPosition.x, 0.0f, targetPosition.z - currentPosition.z);
            float remainingDistance = planarTargetDelta.magnitude;
            if (remainingDistance <= Mathf.Max(0.02f, rootMotionArrivalDistance))
            {
                record.LastMotionDelta = Vector3.zero;
                ResetAnimatorRootTransform(record);
                return;
            }

            Vector3 rootOffset = record.AnimatorRootTransform.localPosition - record.AnimatorRootBaseLocalPosition;
            Vector3 planarRootOffset = new Vector3(rootOffset.x, 0.0f, rootOffset.z);
            float rootMotionStep = planarRootOffset.magnitude;
            float maxStep = Mathf.Max(0.05f, birdMoveSpeed) * Time.deltaTime;
            if (rootMotionStep <= 0.00005f)
            {
                rootMotionStep = maxStep;
            }

            rootMotionStep = Mathf.Min(rootMotionStep, maxStep);
            float appliedStep = Mathf.Min(rootMotionStep, remainingDistance);
            Vector3 motionDelta = planarTargetDelta.normalized * appliedStep;
            record.GameObject.transform.localPosition = currentPosition + motionDelta;
            record.LastMotionDelta = motionDelta;
            ResetAnimatorRootTransform(record);
        }

        private bool ShouldUseAnimatorRootMotionMovement(BirdVisualRecord record)
        {
            return displayFirstZoneProxyMode &&
                useAnimatorRootMotionForDisplayMovement &&
                record != null &&
                record.UsesChickenModel &&
                record.AnimatorComponent != null &&
                record.AnimatorRootTransform != null &&
                HasAnimatorState(record, "Walk_RM");
        }

        private void ResetAnimatorRootTransform(BirdVisualRecord record)
        {
            if (record == null || record.AnimatorRootTransform == null)
            {
                return;
            }

            record.AnimatorRootTransform.localPosition = record.AnimatorRootBaseLocalPosition;
            record.AnimatorRootTransform.localRotation = record.AnimatorRootBaseLocalRotation;
        }

        private void UpdateBirdAnimation(BirdVisualRecord record, bool isMoving)
        {
            UpdateBirdAnimation(record, isMoving, 0.0f);
        }

        private void UpdateBirdAnimation(BirdVisualRecord record, bool isMoving, float planarMotionSpeed)
        {
            if (record == null || !record.UsesChickenModel || record.AnimatorComponent == null)
            {
                return;
            }

            string nextState = GetBirdAnimationState(record, record.Behavior, isMoving);
            UpdateAnimatorPlaybackSpeed(record, nextState, isMoving, planarMotionSpeed);
            if (record.CurrentAnimationState == nextState && IsAnimatorInOrEnteringState(record.AnimatorComponent, nextState))
            {
                return;
            }

            int stateHash = Animator.StringToHash(nextState);
            if (!record.AnimatorComponent.HasState(0, stateHash))
            {
                return;
            }

            record.AnimatorComponent.CrossFade(stateHash, 0.18f, 0);
            record.CurrentAnimationState = nextState;
        }

        private bool IsAnimatorInOrEnteringState(Animator animatorComponent, string stateName)
        {
            if (animatorComponent == null || string.IsNullOrEmpty(stateName))
            {
                return false;
            }

            AnimatorStateInfo currentState = animatorComponent.GetCurrentAnimatorStateInfo(0);
            if (currentState.IsName(stateName))
            {
                return true;
            }

            if (animatorComponent.IsInTransition(0))
            {
                AnimatorStateInfo nextState = animatorComponent.GetNextAnimatorStateInfo(0);
                return nextState.IsName(stateName);
            }

            return false;
        }

        private void UpdateAnimatorPlaybackSpeed(BirdVisualRecord record, string nextState, bool isMoving, float planarMotionSpeed)
        {
            if (record == null || record.AnimatorComponent == null)
            {
                return;
            }

            float speedMultiplier = 1.0f;
            if (isMoving && IsWalkAnimationState(nextState))
            {
                speedMultiplier = ShouldUseAnimatorRootMotionMovement(record)
                    ? 1.0f
                    : Mathf.Clamp(0.72f + (planarMotionSpeed * 0.42f), 0.72f, 1.36f);
            }

            record.AnimatorComponent.speed = record.BaseAnimatorSpeed * speedMultiplier;
        }

        private string GetBirdAnimationState(BirdVisualRecord record, string behavior, bool isMoving)
        {
            string normalizedBehavior = (behavior ?? string.Empty).ToLowerInvariant();
            if (isMoving)
            {
                return GetPreferredWalkAnimationState(record);
            }

            if (normalizedBehavior == "wing_flapping" && IsInResourceDisplayArea(record))
            {
                return "Idle1";
            }

            switch (normalizedBehavior)
            {
                case "active":
                    return "Idle1";
                case "feeding":
                case "drinking":
                    return "Idle1";
                case "preening":
                    return "Idle2";
                case "perching":
                    return "IdleLay";
                case "wing_flapping":
                    return "GetHit";
                case "idle":
                    return "Idle1";
                default:
                    return "Idle1";
            }
        }

        private string GetPreferredWalkAnimationState(BirdVisualRecord record)
        {
            if (preferWalkRootMotionClipForWalking && HasAnimatorState(record, "Walk_RM"))
            {
                return "Walk_RM";
            }

            return "Walk";
        }

        private bool IsInResourceDisplayArea(BirdVisualRecord record)
        {
            if (record == null)
            {
                return false;
            }

            if (record.ZoneId == PoultryTwinRoomLayout.FeederZoneId || record.ZoneId == PoultryTwinRoomLayout.DrinkerZoneId)
            {
                return true;
            }

            Vector3 targetPosition = record.TargetPosition;
            return IsNearResourceCenter(PoultryTwinRoomLayout.FeederZoneId, targetPosition, 1.45f) ||
                IsNearResourceCenter(PoultryTwinRoomLayout.DrinkerZoneId, targetPosition, 1.25f);
        }

        private bool IsNearResourceCenter(string resourceZoneId, Vector3 position, float radius)
        {
            PoultryTwinRoomLayout.ZoneProfile profile = PoultryTwinRoomLayout.GetProfileOrDefault(resourceZoneId);
            Rect resourceRect = GetLargestWorldRect(profile.WorldRects);
            if (resourceRect.width <= 0.0f || resourceRect.height <= 0.0f)
            {
                return false;
            }

            float deltaX = position.x - resourceRect.center.x;
            float deltaZ = position.z - resourceRect.center.y;
            return ((deltaX * deltaX) + (deltaZ * deltaZ)) <= radius * radius;
        }

        private bool IsWalkAnimationState(string stateName)
        {
            return stateName == "Walk" || stateName == "Walk_RM";
        }

        private bool HasAnimatorState(BirdVisualRecord record, string stateName)
        {
            return record != null &&
                record.AnimatorComponent != null &&
                record.AnimatorComponent.HasState(0, Animator.StringToHash(stateName));
        }

        private void DisableCollidersInChildren(GameObject target)
        {
            if (target == null)
            {
                return;
            }

            Collider[] colliders = target.GetComponentsInChildren<Collider>(true);
            for (int index = 0; index < colliders.Length; index++)
            {
                colliders[index].enabled = false;
            }
        }

        private Color GetBirdColor(string behavior)
        {
            switch ((behavior ?? string.Empty).ToLowerInvariant())
            {
                case "feeding":
                    return new Color(0.40f, 0.74f, 0.38f, 0.95f);
                case "drinking":
                    return new Color(0.91f, 0.33f, 0.27f, 0.95f);
                case "active":
                    return new Color(0.93f, 0.70f, 0.26f, 0.95f);
                case "preening":
                    return new Color(0.48f, 0.68f, 0.86f, 0.95f);
                case "perching":
                    return new Color(0.32f, 0.42f, 0.54f, 0.95f);
                case "wing_flapping":
                    return new Color(0.96f, 0.82f, 0.42f, 0.95f);
                case "idle":
                default:
                    return new Color(0.82f, 0.84f, 0.86f, 0.95f);
            }
        }

        private void UpdateZoneVisual(
            ZoneVisualRecord record,
            float normalized,
            bool isSelected,
            bool isAbnormal,
            string eventPhase)
        {
            if (record == null)
            {
                return;
            }

            Color baseColor = Color.Lerp(inactiveZoneColor, record.AccentColor, Mathf.Clamp01(normalized * 0.95f));
            baseColor = ApplyEventTint(baseColor, eventPhase);
            if (isAbnormal)
            {
                baseColor = Color.Lerp(baseColor, abnormalZoneColor, 0.58f);
            }
            if (isSelected)
            {
                baseColor = Color.Lerp(baseColor, selectedZoneColor, 0.52f);
            }

            baseColor.a = isSelected
                ? 0.28f
                : Mathf.Lerp(0.06f, 0.16f, Mathf.Clamp01(normalized));
            if (isAbnormal)
            {
                baseColor.a = Mathf.Max(baseColor.a, 0.22f);
            }

            Color outlineColor = Color.Lerp(record.AccentColor, Color.white, isSelected ? 0.36f : 0.14f);
            outlineColor.a = isSelected ? 0.72f : 0.48f;
            float outlineEmissionStrength = 0.04f + (normalized * 0.12f) + (isSelected ? 0.08f : 0.0f);

            for (int surfaceIndex = 0; surfaceIndex < record.Surfaces.Length; surfaceIndex++)
            {
                ZoneSurfaceRecord surface = record.Surfaces[surfaceIndex];
                ApplyMaterialColor(surface.FillRenderer, baseColor, 0.0f);
                ApplyMaterialColor(surface.OutlineRenderers, outlineColor, outlineEmissionStrength);

                Vector3 fillScale = surface.FillRenderer.transform.localScale;
                fillScale.y = isSelected ? surfaceHeight * 1.75f : surfaceHeight;
                surface.FillRenderer.transform.localScale = fillScale;

                Vector3 fillPosition = surface.FillRenderer.transform.localPosition;
                fillPosition.y = overlayLift + (fillScale.y * 0.5f);
                surface.FillRenderer.transform.localPosition = fillPosition;
            }
        }

        private bool IsZoneAbnormal(string zoneId, float normalized, float riskScore)
        {
            if (zoneId == PoultryTwinRoomLayout.FeederZoneId || zoneId == PoultryTwinRoomLayout.DrinkerZoneId)
            {
                return riskScore >= 0.60f && normalized < 0.28f;
            }

            if (zoneId == PoultryTwinRoomLayout.RestingZoneId)
            {
                return riskScore >= 0.55f && normalized > 0.68f;
            }

            if (zoneId == PoultryTwinRoomLayout.OpenMovementZoneId)
            {
                return riskScore >= 0.60f && normalized < 0.28f;
            }

            return false;
        }

        private float GetDisplayValue(PoultryTwinZoneFrame zoneFrame)
        {
            return GetCalibratedActivityValue(zoneFrame);
        }

        private float GetNormalizedValue(PoultryTwinZoneFrame zoneFrame)
        {
            if (zoneFrame == null)
            {
                return 0.0f;
            }

            return Mathf.Clamp01(GetCalibratedActivityValue(zoneFrame));
        }

        private bool IsPendingZoneFrame(PoultryTwinZoneFrame zoneFrame)
        {
            return zoneFrame != null && zoneFrame.source_zone_id == "pending_scene_zone";
        }

        private float GetCalibratedActivityValue(PoultryTwinZoneFrame zoneFrame)
        {
            if (zoneFrame == null || IsPendingZoneFrame(zoneFrame))
            {
                return 0.0f;
            }

            string semanticZoneId;
            if (!PoultryTwinRoomLayout.TryMapExternalZoneId(zoneFrame.zone_id, out semanticZoneId))
            {
                semanticZoneId = zoneFrame.zone_id;
            }

            float rawValue = GetRawActivityValue(zoneFrame);
            if (rawValue <= 0.0f)
            {
                return 0.0f;
            }

            float reference = 0.05f;
            if (zoneActivityReferenceLookup.ContainsKey(semanticZoneId))
            {
                reference = zoneActivityReferenceLookup[semanticZoneId];
            }

            float scaled = Mathf.Clamp01(rawValue / Mathf.Max(reference, 0.0001f));
            return Mathf.Sqrt(scaled);
        }

        private float GetRawActivityValue(PoultryTwinZoneFrame zoneFrame)
        {
            if (zoneFrame == null)
            {
                return 0.0f;
            }

            if (zoneFrame.activity_norm > 0.0f)
            {
                return zoneFrame.activity_norm;
            }

            if (zoneFrame.overlay_intensity > 0.0f)
            {
                return zoneFrame.overlay_intensity;
            }

            return zoneFrame.activity;
        }

        private float GetPercentile(List<float> sortedValues, float percentile)
        {
            if (sortedValues == null || sortedValues.Count == 0)
            {
                return 0.0f;
            }

            if (sortedValues.Count == 1)
            {
                return sortedValues[0];
            }

            float clampedPercentile = Mathf.Clamp01(percentile);
            float rawIndex = (sortedValues.Count - 1) * clampedPercentile;
            int lowerIndex = Mathf.FloorToInt(rawIndex);
            int upperIndex = Mathf.CeilToInt(rawIndex);
            if (lowerIndex == upperIndex)
            {
                return sortedValues[lowerIndex];
            }

            float blend = rawIndex - lowerIndex;
            return Mathf.Lerp(sortedValues[lowerIndex], sortedValues[upperIndex], blend);
        }

        private Color ApplyEventTint(Color color, string eventPhase)
        {
            if (eventPhase == "during")
            {
                return Color.Lerp(color, new Color(0.94f, 0.67f, 0.25f, color.a), 0.18f);
            }

            if (eventPhase == "recovery")
            {
                return Color.Lerp(color, new Color(0.88f, 0.82f, 0.42f, color.a), 0.10f);
            }

            return color;
        }

        private void ApplyLitterMaterial(Renderer rendererComponent, Vector2 textureScale)
        {
            Material material = GetMutableRendererMaterial(rendererComponent);
            if (material == null)
            {
                return;
            }

            Texture2D texture = GetOrCreateRoomLitterTexture();
            SetMaterialTextureIfPresent(material, "_BaseMap", texture, textureScale);
            SetMaterialTextureIfPresent(material, "_MainTex", texture, textureScale);
            SetMaterialFloatIfPresent(material, "_Smoothness", 0.16f);
            SetMaterialFloatIfPresent(material, "_Metallic", 0.0f);
        }

        private Texture2D GetOrCreateRoomLitterTexture()
        {
            if (roomLitterTexture != null)
            {
                return roomLitterTexture;
            }

            int textureSize = 128;
            roomLitterTexture = new Texture2D(textureSize, textureSize, TextureFormat.RGBA32, false);
            roomLitterTexture.name = "Room1_Litter_Runtime";
            roomLitterTexture.wrapMode = TextureWrapMode.Repeat;
            roomLitterTexture.filterMode = FilterMode.Bilinear;

            Color darkStraw = new Color(0.38f, 0.27f, 0.13f, 1.0f);
            Color midStraw = new Color(0.68f, 0.52f, 0.28f, 1.0f);
            Color lightStraw = new Color(0.88f, 0.74f, 0.46f, 1.0f);
            for (int y = 0; y < textureSize; y++)
            {
                for (int x = 0; x < textureSize; x++)
                {
                    float broadNoise = Mathf.PerlinNoise((x + 17.0f) * 0.045f, (y + 31.0f) * 0.045f);
                    float fineNoise = Mathf.PerlinNoise((x + 3.0f) * 0.19f, (y + 11.0f) * 0.19f);
                    float value = Mathf.Clamp01((broadNoise * 0.68f) + (fineNoise * 0.32f));
                    Color color = Color.Lerp(darkStraw, lightStraw, value);

                    float fiber = Hash01((x * 73856093) ^ (y * 19349663));
                    if (fiber > 0.82f)
                    {
                        color = Color.Lerp(color, lightStraw, 0.55f);
                    }
                    else if (fiber < 0.12f)
                    {
                        color = Color.Lerp(color, midStraw, 0.45f);
                    }

                    roomLitterTexture.SetPixel(x, y, color);
                }
            }

            roomLitterTexture.Apply(false, false);
            return roomLitterTexture;
        }

        private GameObject CreateCableBetweenPoints(string name, Transform parent, Vector3 startPoint, Vector3 endPoint, float diameter, Color color)
        {
            Vector3 delta = endPoint - startPoint;
            float length = delta.magnitude;
            if (length <= 0.0001f)
            {
                return null;
            }

            GameObject cable = CreateCylinder(
                name,
                parent,
                startPoint + (delta * 0.5f),
                new Vector3(diameter, length * 0.5f, diameter),
                color
            );
            cable.transform.localRotation = Quaternion.FromToRotation(Vector3.up, delta.normalized);
            return cable;
        }

        private void CreateCablePolyline(string name, Vector3[] points, float diameter, Color color)
        {
            if (points == null || points.Length < 2)
            {
                return;
            }

            for (int index = 0; index < points.Length - 1; index++)
            {
                CreateCableBetweenPoints(name + "_" + index, environmentRoot, points[index], points[index + 1], diameter, color);
            }
        }

        private Light AddPointLight(string name, Vector3 localPosition, Color color, float intensity, float range)
        {
            GameObject lightObject = new GameObject(name);
            lightObject.transform.SetParent(environmentRoot, false);
            lightObject.transform.localPosition = localPosition;
            Light lightComponent = lightObject.AddComponent<Light>();
            lightComponent.type = LightType.Point;
            lightComponent.color = color;
            lightComponent.intensity = intensity;
            lightComponent.range = range;
            lightComponent.shadows = LightShadows.None;
            return lightComponent;
        }

        private void SetRendererEmission(Renderer rendererComponent, Color color, float strength)
        {
            Material material = GetMutableRendererMaterial(rendererComponent);
            if (material == null)
            {
                return;
            }

            if (material.HasProperty("_EmissionColor"))
            {
                material.EnableKeyword("_EMISSION");
                material.SetColor("_EmissionColor", color * strength);
            }
        }

        private void SetMaterialTextureIfPresent(Material material, string propertyName, Texture texture, Vector2 scale)
        {
            if (material == null || texture == null || !material.HasProperty(propertyName))
            {
                return;
            }

            material.SetTexture(propertyName, texture);
            material.SetTextureScale(propertyName, scale);
        }

        private float Hash01(int seed)
        {
            unchecked
            {
                uint hash = (uint)seed;
                hash ^= 2747636419u;
                hash *= 2654435769u;
                hash ^= hash >> 16;
                hash *= 2654435769u;
                hash ^= hash >> 16;
                return (hash & 0x00FFFFFFu) / 16777215.0f;
            }
        }

        private GameObject CreateCube(string name, Transform parent, Vector3 localPosition, Vector3 localScale, Color color)
        {
            return CreatePrimitive(PrimitiveType.Cube, name, parent, localPosition, localScale, color, Vector3.zero);
        }

        private GameObject CreateCube(string name, Transform parent, Vector3 localPosition, Vector3 localScale, Color color, Vector3 localEulerAngles)
        {
            return CreatePrimitive(PrimitiveType.Cube, name, parent, localPosition, localScale, color, localEulerAngles);
        }

        private GameObject CreateCylinder(string name, Transform parent, Vector3 localPosition, Vector3 localScale, Color color)
        {
            return CreatePrimitive(PrimitiveType.Cylinder, name, parent, localPosition, localScale, color, Vector3.zero);
        }

        private GameObject CreateSphere(string name, Transform parent, Vector3 localPosition, Vector3 localScale, Color color)
        {
            return CreatePrimitive(PrimitiveType.Sphere, name, parent, localPosition, localScale, color, Vector3.zero);
        }

        private GameObject CreatePrimitive(PrimitiveType primitiveType, string name, Transform parent, Vector3 localPosition, Vector3 localScale, Color color, Vector3 localEulerAngles)
        {
            GameObject gameObject = GameObject.CreatePrimitive(primitiveType);
            gameObject.name = name;
            gameObject.transform.SetParent(parent, false);
            gameObject.transform.localPosition = localPosition;
            gameObject.transform.localRotation = Quaternion.Euler(localEulerAngles);
            gameObject.transform.localScale = localScale;

            Renderer rendererComponent = gameObject.GetComponent<Renderer>();
            Material material = CreateMaterial(color);
            if (material != null)
            {
                if (Application.isPlaying)
                {
                    rendererComponent.material = material;
                }
                else
                {
                    rendererComponent.sharedMaterial = material;
                }
            }

            DisableCollider(gameObject);
            return gameObject;
        }

        private Material CreateMaterial(Color color)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
            {
                shader = Shader.Find("Unlit/Texture");
            }

            if (shader == null)
            {
                shader = Shader.Find("Unlit/Color");
            }

            if (shader == null)
            {
                shader = Shader.Find("Universal Render Pipeline/Lit");
            }

            if (shader == null)
            {
                shader = Shader.Find("Standard");
            }

            if (shader == null)
            {
                LastErrorMessage = "No compatible shader found for the Room 1 scene.";
                return null;
            }

            Material material = new Material(shader);
            material.color = color;
            if (material.HasProperty("_BaseColor"))
            {
                material.SetColor("_BaseColor", color);
            }

            if (material.HasProperty("_Smoothness"))
            {
                material.SetFloat("_Smoothness", 0.10f);
            }

            if (material.HasProperty("_Metallic"))
            {
                material.SetFloat("_Metallic", 0.0f);
            }

            if (material.HasProperty("_EmissionColor"))
            {
                if (color.a >= 0.99f)
                {
                    material.EnableKeyword("_EMISSION");
                    material.SetColor("_EmissionColor", color * 0.075f);
                }
                else
                {
                    material.DisableKeyword("_EMISSION");
                    material.SetColor("_EmissionColor", Color.black);
                }
            }

            ConfigureMaterialBlendMode(material, color.a < 0.99f);
            return material;
        }

        private Material GetMutableRendererMaterial(Renderer rendererComponent)
        {
            if (rendererComponent == null)
            {
                return null;
            }

            return Application.isPlaying ? rendererComponent.material : rendererComponent.sharedMaterial;
        }

        private void ApplyMaterialColor(Renderer rendererComponent, Color color, float emissionStrength)
        {
            Material material = GetMutableRendererMaterial(rendererComponent);
            if (material == null)
            {
                return;
            }

            material.color = color;
            if (material.HasProperty("_BaseColor"))
            {
                material.SetColor("_BaseColor", color);
            }

            if (material.HasProperty("_EmissionColor"))
            {
                if (emissionStrength > 0.001f)
                {
                    material.EnableKeyword("_EMISSION");
                    material.SetColor("_EmissionColor", color * emissionStrength);
                }
                else
                {
                    material.DisableKeyword("_EMISSION");
                    material.SetColor("_EmissionColor", Color.black);
                }
            }

            ConfigureMaterialBlendMode(material, color.a < 0.99f);
        }

        private void ApplyMaterialColor(Renderer[] rendererComponents, Color color, float emissionStrength)
        {
            if (rendererComponents == null)
            {
                return;
            }

            for (int index = 0; index < rendererComponents.Length; index++)
            {
                ApplyMaterialColor(rendererComponents[index], color, emissionStrength);
            }
        }

        private void RepairChickenRendererMaterials(Renderer rendererComponent)
        {
            if (rendererComponent == null)
            {
                return;
            }

            Material[] sourceMaterials = rendererComponent.sharedMaterials;
            if (sourceMaterials == null || sourceMaterials.Length == 0)
            {
                return;
            }

            Material[] repairedMaterials = new Material[sourceMaterials.Length];
            bool changed = false;
            for (int index = 0; index < sourceMaterials.Length; index++)
            {
                Material sourceMaterial = sourceMaterials[index];
                Material repairedMaterial = GetOrCreateChickenMaterial(sourceMaterial);
                repairedMaterials[index] = repairedMaterial;
                changed = changed || repairedMaterial != sourceMaterial;
            }

            if (changed)
            {
                rendererComponent.sharedMaterials = repairedMaterials;
            }
        }

        private Material GetOrCreateChickenMaterial(Material sourceMaterial)
        {
            if (sourceMaterial == null)
            {
                return null;
            }

            Material cachedMaterial;
            if (chickenMaterialLookup.TryGetValue(sourceMaterial, out cachedMaterial))
            {
                return cachedMaterial;
            }

            Shader shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null)
            {
                shader = Shader.Find("Standard");
            }

            if (shader == null)
            {
                LastErrorMessage = "No compatible shader found for chicken model materials.";
                return sourceMaterial;
            }

            Material material = new Material(shader);
            material.name = sourceMaterial.name + "_RuntimeURP";
            Color baseColor = GetMaterialColor(sourceMaterial, "_Color", Color.white);
            SetMaterialColorIfPresent(material, "_BaseColor", baseColor);
            SetMaterialColorIfPresent(material, "_Color", baseColor);
            CopyMaterialTexture(sourceMaterial, "_MainTex", material, "_BaseMap");
            CopyMaterialTexture(sourceMaterial, "_MainTex", material, "_MainTex");
            CopyMaterialTexture(sourceMaterial, "_BumpMap", material, "_BumpMap");
            CopyMaterialTexture(sourceMaterial, "_MetallicGlossMap", material, "_MetallicGlossMap");
            CopyMaterialTexture(sourceMaterial, "_OcclusionMap", material, "_OcclusionMap");
            CopyMaterialFloat(sourceMaterial, "_Cutoff", material, "_Cutoff", 0.159f);
            CopyMaterialFloat(sourceMaterial, "_BumpScale", material, "_BumpScale", 1.0f);
            CopyMaterialFloat(sourceMaterial, "_OcclusionStrength", material, "_OcclusionStrength", 1.0f);

            if (material.HasProperty("_Metallic"))
            {
                material.SetFloat("_Metallic", 0.0f);
            }

            if (material.HasProperty("_Smoothness"))
            {
                material.SetFloat("_Smoothness", 0.42f);
            }

            EnableAlphaClip(material);
            chickenMaterialLookup[sourceMaterial] = material;
            return material;
        }

        private Color GetMaterialColor(Material material, string propertyName, Color fallback)
        {
            return material != null && material.HasProperty(propertyName) ? material.GetColor(propertyName) : fallback;
        }

        private void SetMaterialColorIfPresent(Material material, string propertyName, Color color)
        {
            if (material != null && material.HasProperty(propertyName))
            {
                material.SetColor(propertyName, color);
            }
        }

        private void SetMaterialFloatIfPresent(Material material, string propertyName, float value)
        {
            if (material != null && material.HasProperty(propertyName))
            {
                material.SetFloat(propertyName, value);
            }
        }

        private void CopyMaterialTexture(Material sourceMaterial, string sourceProperty, Material targetMaterial, string targetProperty)
        {
            if (sourceMaterial == null || targetMaterial == null || !sourceMaterial.HasProperty(sourceProperty) || !targetMaterial.HasProperty(targetProperty))
            {
                return;
            }

            Texture texture = sourceMaterial.GetTexture(sourceProperty);
            if (texture == null)
            {
                return;
            }

            targetMaterial.SetTexture(targetProperty, texture);
            targetMaterial.SetTextureScale(targetProperty, sourceMaterial.GetTextureScale(sourceProperty));
            targetMaterial.SetTextureOffset(targetProperty, sourceMaterial.GetTextureOffset(sourceProperty));
        }

        private void CopyMaterialFloat(Material sourceMaterial, string sourceProperty, Material targetMaterial, string targetProperty, float fallback)
        {
            if (targetMaterial == null || !targetMaterial.HasProperty(targetProperty))
            {
                return;
            }

            float value = sourceMaterial != null && sourceMaterial.HasProperty(sourceProperty)
                ? sourceMaterial.GetFloat(sourceProperty)
                : fallback;
            targetMaterial.SetFloat(targetProperty, value);
        }

        private void EnableAlphaClip(Material material)
        {
            if (material == null)
            {
                return;
            }

            material.SetOverrideTag("RenderType", "TransparentCutout");
            material.EnableKeyword("_ALPHATEST_ON");
            material.renderQueue = (int)RenderQueue.AlphaTest;
            if (material.HasProperty("_AlphaClip"))
            {
                material.SetFloat("_AlphaClip", 1.0f);
            }

            if (material.HasProperty("_Surface"))
            {
                material.SetFloat("_Surface", 0.0f);
            }

            if (material.HasProperty("_ZWrite"))
            {
                material.SetFloat("_ZWrite", 1.0f);
            }
        }

        private void ConfigureOverlayRenderer(Renderer rendererComponent)
        {
            if (rendererComponent == null)
            {
                return;
            }

            rendererComponent.shadowCastingMode = ShadowCastingMode.Off;
            rendererComponent.receiveShadows = false;
        }

        private void ConfigureMaterialBlendMode(Material material, bool transparent)
        {
            if (material == null || !transparent)
            {
                return;
            }

            material.SetOverrideTag("RenderType", "Transparent");
            material.SetInt("_SrcBlend", (int)BlendMode.SrcAlpha);
            material.SetInt("_DstBlend", (int)BlendMode.OneMinusSrcAlpha);
            material.SetInt("_ZWrite", 0);
            material.DisableKeyword("_ALPHATEST_ON");
            material.EnableKeyword("_ALPHABLEND_ON");
            material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            material.renderQueue = (int)RenderQueue.Transparent;

            if (material.HasProperty("_Surface"))
            {
                material.SetFloat("_Surface", 1.0f);
            }
        }

        private void DisableCollider(GameObject target)
        {
            if (target == null)
            {
                return;
            }

            Collider colliderComponent = target.GetComponent<Collider>();
            if (colliderComponent != null)
            {
                colliderComponent.enabled = false;
            }
        }

        private void EnableCollider(GameObject target)
        {
            if (target == null)
            {
                return;
            }

            Collider colliderComponent = target.GetComponent<Collider>();
            if (colliderComponent != null)
            {
                colliderComponent.enabled = true;
            }
        }

        private Vector3 RectToCenter(Rect rect)
        {
            return new Vector3(rect.x + (rect.width * 0.5f), overlayLift + (surfaceHeight * 0.5f), rect.y + (rect.height * 0.5f));
        }

        private Rect GetLargestWorldRect(Rect[] rects)
        {
            if (rects == null || rects.Length == 0)
            {
                return default;
            }

            Rect largestRect = rects[0];
            float largestArea = largestRect.width * largestRect.height;
            for (int index = 1; index < rects.Length; index++)
            {
                float area = rects[index].width * rects[index].height;
                if (area <= largestArea)
                {
                    continue;
                }

                largestRect = rects[index];
                largestArea = area;
            }

            return largestRect;
        }

        private sealed class ZoneVisualRecord
        {
            public readonly string DisplayName;
            public readonly Color AccentColor;
            public readonly ZoneSurfaceRecord[] Surfaces;

            public ZoneVisualRecord(string displayName, Color accentColor, ZoneSurfaceRecord[] surfaces)
            {
                DisplayName = displayName;
                AccentColor = accentColor;
                Surfaces = surfaces;
            }
        }

        private sealed class ZoneSurfaceRecord
        {
            public readonly Renderer FillRenderer;
            public readonly Renderer[] OutlineRenderers;

            public ZoneSurfaceRecord(Renderer fillRenderer, Renderer[] outlineRenderers)
            {
                FillRenderer = fillRenderer;
                OutlineRenderers = outlineRenderers;
            }
        }

        private sealed class BirdVisualRecord
        {
            public readonly GameObject GameObject;
            public readonly Renderer[] RendererComponents;
            public readonly Animator AnimatorComponent;
            public readonly Transform AnimatorRootTransform;
            public readonly Vector3 AnimatorRootBaseLocalPosition;
            public readonly Quaternion AnimatorRootBaseLocalRotation;
            public readonly bool UsesChickenModel;
            public readonly float BaseAnimatorSpeed;
            public Vector3 TransitionStartPosition;
            public Vector3 TargetPosition;
            public Vector3 FacingPoint;
            public Vector3 LastMotionDelta;
            public float TransitionElapsed;
            public float TransitionDuration;
            public bool HasTarget;
            public bool HasFacingPoint;
            public bool HasActivePosition;
            public bool AssignedThisFrame;
            public string Behavior;
            public string ZoneId;
            public string TrackKey;
            public string CurrentAnimationState;

            public BirdVisualRecord(GameObject gameObject, Renderer[] rendererComponents, Animator animatorComponent, Transform animatorRootTransform, bool usesChickenModel, float baseAnimatorSpeed)
            {
                GameObject = gameObject;
                RendererComponents = rendererComponents;
                AnimatorComponent = animatorComponent;
                AnimatorRootTransform = animatorRootTransform;
                AnimatorRootBaseLocalPosition = animatorRootTransform != null ? animatorRootTransform.localPosition : Vector3.zero;
                AnimatorRootBaseLocalRotation = animatorRootTransform != null ? animatorRootTransform.localRotation : Quaternion.identity;
                UsesChickenModel = usesChickenModel;
                BaseAnimatorSpeed = baseAnimatorSpeed;
                TransitionStartPosition = Vector3.zero;
                TargetPosition = Vector3.zero;
                FacingPoint = Vector3.zero;
                LastMotionDelta = Vector3.zero;
                TransitionElapsed = 0.0f;
                TransitionDuration = 0.0f;
                HasTarget = false;
                HasFacingPoint = false;
                HasActivePosition = false;
                AssignedThisFrame = false;
                Behavior = string.Empty;
                ZoneId = string.Empty;
                TrackKey = string.Empty;
                CurrentAnimationState = string.Empty;
            }
        }
    }

    public sealed class ZoneSurfaceMarker : MonoBehaviour
    {
        public string ZoneId;
    }
}
