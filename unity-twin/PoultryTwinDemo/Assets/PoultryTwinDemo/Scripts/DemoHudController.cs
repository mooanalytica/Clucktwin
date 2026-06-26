using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace PoultryTwinDemo
{
    public class DemoHudController : MonoBehaviour
    {
        private const string DisclaimerText = "Prototype scientific visualization from precomputed video analytics. This is not a validated welfare diagnosis.";
        private const string CompactDisclaimerText = "Prototype visualization only; not a validated diagnosis.";
        private const float PanelHeaderHeight = 72.0f;
        private const float EditorGameViewBottomReserve = 78.0f;

        [SerializeField] private PoultryTwinPlaybackController playbackController;

        private readonly List<Rect> interactiveRects = new List<Rect>();
        private readonly List<DateBand> dateBands = new List<DateBand>();

        private GUIStyle panelStyle;
        private GUIStyle headerButtonStyle;
        private GUIStyle titleStyle;
        private GUIStyle sectionStyle;
        private GUIStyle bodyStyle;
        private GUIStyle smallStyle;
        private GUIStyle alertStyle;
        private GUIStyle metricLabelStyle;
        private GUIStyle metricValueStyle;
        private GUIStyle foldoutTitleStyle;
        private GUIStyle foldoutSubtitleStyle;

        private PoultryTwinTimelineFrame currentFrame;
        private int cachedFrameCount = -1;
        private Texture2D roomMapTexture;
        private Texture2D solidTexture;

        private bool focusExpanded = true;
        private bool metricsExpanded;
        private bool behaviorExpanded;
        private bool abnormalityExpanded;

        public static DemoHudController ActiveInstance { get; private set; }

        private struct MetricItem
        {
            public string Label;
            public string Description;
            public float Value;
            public bool Available;
        }

        private sealed class DateBand
        {
            public string RawDate;
            public string DisplayLabel;
            public int StartIndex;
            public int Count;
            public Color Color;
        }

        private void OnEnable()
        {
            ActiveInstance = this;
        }

        private void OnDisable()
        {
            if (ActiveInstance == this)
            {
                ActiveInstance = null;
            }
        }

        public void SetPlaybackController(PoultryTwinPlaybackController controller)
        {
            playbackController = controller;
        }

        public void Initialize(PoultryTwinPlaybackController controller)
        {
            playbackController = controller;
            Refresh();
        }

        public void Refresh()
        {
            currentFrame = playbackController != null ? playbackController.GetCurrentFrame() : null;
            RebuildDateBandsIfNeeded();
        }

        public void ShowFrame(PoultryTwinTimelineFrame frame)
        {
            currentFrame = frame;
        }

        public bool ContainsScreenPoint(Vector2 screenPoint)
        {
            if (!PoultryTwinInputAdapter.IsValidScreenPoint(screenPoint))
            {
                return false;
            }

            Vector2 guiPoint = new Vector2(screenPoint.x, Screen.height - screenPoint.y);
            for (int index = 0; index < interactiveRects.Count; index++)
            {
                if (interactiveRects[index].Contains(guiPoint))
                {
                    return true;
                }
            }

            return false;
        }

        private void OnGUI()
        {
            EnsureStyles();
            RebuildDateBandsIfNeeded();
            interactiveRects.Clear();
            EnsureTextures();

            float screenWidth = Screen.width;
            float screenHeight = Screen.height;
            float gutter = 18.0f;
            float rightWidth = Mathf.Clamp(screenWidth * 0.27f, 300.0f, 368.0f);
            float overviewMaxWidth = Mathf.Max(320.0f, screenWidth - rightWidth - (gutter * 3.0f));
            float overviewWidth = Mathf.Min(Mathf.Clamp(screenWidth * 0.18f, 320.0f, 380.0f), overviewMaxWidth);

            Rect overviewRect = new Rect(gutter, gutter, overviewWidth, 410.0f);

            float timelineHeight = 176.0f;
            float bottomReserve = GetBottomUiReserve();
            float timelineY = screenHeight - gutter - bottomReserve - timelineHeight;
            Rect timelineRect = new Rect(gutter, timelineY, screenWidth - (gutter * 2.0f), timelineHeight);

            float cardX = screenWidth - gutter - rightWidth;
            float rightColumnY = gutter;
            Rect focusRect = BuildPanelRect(cardX, rightColumnY, rightWidth, focusExpanded, 288.0f);
            rightColumnY = focusRect.yMax + gutter;
            Rect metricsRect = BuildPanelRect(cardX, rightColumnY, rightWidth, metricsExpanded, 336.0f);
            float maxAbnormalityHeight = Mathf.Max(PanelHeaderHeight, timelineY - gutter - (metricsRect.yMax + gutter));
            float abnormalityContentHeight = abnormalityExpanded
                ? Mathf.Max(0.0f, Mathf.Min(188.0f, maxAbnormalityHeight - PanelHeaderHeight))
                : 0.0f;
            Rect abnormalityRect = BuildPanelRect(cardX, metricsRect.yMax + gutter, rightWidth, abnormalityExpanded, abnormalityContentHeight);

            float leftColumnY = overviewRect.yMax + gutter;
            float maxBehaviorHeight = Mathf.Max(PanelHeaderHeight, timelineY - gutter - leftColumnY);
            float behaviorContentHeight = behaviorExpanded
                ? Mathf.Max(0.0f, Mathf.Min(336.0f, maxBehaviorHeight - PanelHeaderHeight))
                : 0.0f;
            Rect behaviorRect = BuildPanelRect(gutter, leftColumnY, overviewWidth, behaviorExpanded, behaviorContentHeight);

            DrawOverview(overviewRect);
            DrawFocusPanel(focusRect);
            DrawMetricsPanel(metricsRect);
            DrawBehaviorPanel(behaviorRect);
            DrawAbnormalityPanel(abnormalityRect);
            DrawTimelinePanel(timelineRect);
        }

        private Rect BuildPanelRect(float x, float y, float width, bool expanded, float contentHeight)
        {
            return new Rect(x, y, width, PanelHeaderHeight + (expanded ? contentHeight : 0.0f));
        }

        private float GetBottomUiReserve()
        {
#if UNITY_EDITOR
            return EditorGameViewBottomReserve;
#else
            return 0.0f;
#endif
        }

        private void DrawOverview(Rect rect)
        {
            RegisterUiRect(rect);
            DrawPanel(rect);

            string focusLabel = GetFocusLabel();
            string timeLabel = GetCurrentTimeLabel();
            string stateLabel = currentFrame != null && currentFrame.state != null ? HumanizeToken(currentFrame.state.state_label) : "n/a";

            Rect contentRect = Shrink(rect, 16.0f);
            GUI.Label(new Rect(contentRect.x, contentRect.y, contentRect.width, 36.0f), "Poultry Twin | Room 1", titleStyle);
            GUI.Label(new Rect(contentRect.x, contentRect.y + 38.0f, contentRect.width, 24.0f), "Minimum viable scientific prototype", smallStyle);

            Rect imageFrame = new Rect(contentRect.x, contentRect.y + 66.0f, contentRect.width, 126.0f);
            DrawSubPanel(imageFrame, new Color(0.98f, 0.98f, 0.98f, 0.34f), new Color(0.72f, 0.75f, 0.79f, 0.55f));
            DrawRoomMap(imageFrame);

            Rect summaryRect = new Rect(contentRect.x, imageFrame.yMax + 12.0f, contentRect.width, 150.0f);
            DrawSubPanel(summaryRect, new Color(0.98f, 0.98f, 0.98f, 0.26f), new Color(0.72f, 0.75f, 0.79f, 0.42f));
            Rect summaryContent = Shrink(summaryRect, 10.0f);
            GUI.Label(new Rect(summaryContent.x, summaryContent.y, summaryContent.width, 32.0f), focusLabel, sectionStyle);
            GUI.Label(new Rect(summaryContent.x, summaryContent.y + 30.0f, summaryContent.width, 24.0f), timeLabel, smallStyle);
            GUI.Label(new Rect(summaryContent.x, summaryContent.y + 56.0f, summaryContent.width, 26.0f), "Nominal flock: " + GetBirdCountLabel() + "    Visible: " + GetVisibleBirdProxyCountLabel(), bodyStyle);
            GUI.Label(new Rect(summaryContent.x, summaryContent.y + 84.0f, summaryContent.width, 42.0f), "State: " + stateLabel, smallStyle);

            GUI.Label(new Rect(contentRect.x, rect.yMax - 42.0f, contentRect.width, 36.0f), CompactDisclaimerText, smallStyle);
        }

        private void DrawRoomMap(Rect frameRect)
        {
            Rect innerRect = Shrink(frameRect, 8.0f);
            if (roomMapTexture == null)
            {
                GUI.Label(new Rect(innerRect.x, innerRect.y + 68.0f, innerRect.width, 20.0f), "Room 1 reference image unavailable", bodyStyle);
                return;
            }

            Rect imageRect = GetAspectFitRect(innerRect, roomMapTexture.width, roomMapTexture.height);
            GUI.DrawTexture(imageRect, roomMapTexture, ScaleMode.StretchToFill, false);
            DrawZoneMapOverlay(imageRect);
        }

        private void DrawZoneMapOverlay(Rect imageRect)
        {
            ZoneOverlayController overlayController = playbackController != null ? playbackController.OverlayController : null;
            PoultryTwinRoomLayout.ZoneProfile[] profiles = PoultryTwinRoomLayout.GetRoom1Profiles();

            for (int profileIndex = 0; profileIndex < profiles.Length; profileIndex++)
            {
                PoultryTwinRoomLayout.ZoneProfile profile = profiles[profileIndex];
                Rect[] mapRects = profile.GetNormalizedOverlayRects();
                for (int rectIndex = 0; rectIndex < mapRects.Length; rectIndex++)
                {
                    Rect mapRect = mapRects[rectIndex];
                    Rect drawRect = new Rect(
                        imageRect.x + (mapRect.x * imageRect.width),
                        imageRect.y + (mapRect.y * imageRect.height),
                        mapRect.width * imageRect.width,
                        mapRect.height * imageRect.height
                    );

                    bool isSelected = overlayController != null && overlayController.SelectedZoneId == profile.SemanticZoneId;
                    Color fill = profile.AccentColor;
                    fill.a = isSelected ? 0.18f : 0.07f;
                    DrawRectFill(drawRect, fill);
                    DrawRectOutline(drawRect, isSelected ? new Color(0.95f, 0.88f, 0.70f, 0.96f) : new Color(profile.AccentColor.r, profile.AccentColor.g, profile.AccentColor.b, 0.84f), isSelected ? 2.0f : 1.0f);

                    if (GUI.Button(drawRect, GUIContent.none, headerButtonStyle) && overlayController != null)
                    {
                        if (overlayController.SelectedZoneId == profile.SemanticZoneId)
                        {
                            overlayController.ClearSelection();
                        }
                        else
                        {
                            overlayController.SelectZoneById(profile.SemanticZoneId);
                        }
                    }
                }
            }

            if (overlayController != null && overlayController.HasSelectedZone)
            {
                GUI.Label(new Rect(imageRect.x + 8.0f, imageRect.y + 8.0f, imageRect.width - 16.0f, 24.0f), "Selected: " + overlayController.SelectedZoneDisplayName, smallStyle);
            }
            else
            {
                GUI.Label(new Rect(imageRect.x + 8.0f, imageRect.y + 8.0f, imageRect.width - 16.0f, 24.0f), "Click a zone here or in the 3D room", smallStyle);
            }
        }

        private void DrawFocusPanel(Rect rect)
        {
            string subtitle = TryGetFocusZoneFrame(out PoultryTwinZoneFrame zoneFrame)
                ? GetZonePanelSubtitle(zoneFrame)
                : "Default whole-room view";
            DrawFoldoutHeader(rect, "Scope", subtitle, ref focusExpanded);

            if (!focusExpanded)
            {
                return;
            }

            GUILayout.BeginArea(GetContentRect(rect));
            GUILayout.Label(GetFocusLabel(), sectionStyle);
            GUILayout.Space(6.0f);
            GUILayout.Label("Window: " + GetCurrentTimeLabel(), bodyStyle);

            if (zoneFrame != null)
            {
                GUILayout.Label("Data source: " + Safe(zoneFrame.source_display_name), bodyStyle);
                string zoneDataNote = GetZoneDataNote(zoneFrame);
                if (!string.IsNullOrEmpty(zoneDataNote))
                {
                    GUILayout.Label(zoneDataNote, smallStyle);
                }

                GUILayout.Space(6.0f);
                GUILayout.Label("Zone activity index: " + GetZoneActivityValue(zoneFrame).ToString("0.00"), bodyStyle);
                GUILayout.Label("Occupancy share: " + zoneFrame.occupancy_share.ToString("0.00"), bodyStyle);
                GUILayout.Label("Mean visible birds: " + zoneFrame.bird_count_mean.ToString("0.0") + " / " + GetBirdCountLabel(), bodyStyle);

                if (zoneFrame.behavior_summary != null)
                {
                    GUILayout.Label("Dominant behavior: " + HumanizeToken(zoneFrame.behavior_summary.dominant_behaviour), bodyStyle);
                }
            }
            else
            {
                GUILayout.Label("Bird count: " + GetBirdCountLabel(), bodyStyle);
                GUILayout.Label("Visible / tracked proxies: " + GetVisibleBirdProxyCountLabel(), bodyStyle);
                GUILayout.Label("Current state: " + GetStateLabel(), bodyStyle);
                GUILayout.Label("Current risk: " + GetRiskValue().ToString("0.00") + " (" + GetRiskLevelLabel() + ")", bodyStyle);

                PoultryTwinBehaviorSummary roomBehavior = currentFrame != null ? currentFrame.behavior_summary : null;
                if (roomBehavior != null)
                {
                    GUILayout.Label("Dominant behavior: " + HumanizeToken(roomBehavior.dominant_behaviour), bodyStyle);
                }
            }

            GUILayout.EndArea();
        }

        private void DrawMetricsPanel(Rect rect)
        {
            string subtitle = TryGetFocusZoneFrame(out _)
                ? "Activity, occupancy share, birds, confidence"
                : "Activity, occupancy, clustering, risk";
            DrawFoldoutHeader(rect, "Metrics", subtitle, ref metricsExpanded);

            if (!metricsExpanded)
            {
                return;
            }

            GUILayout.BeginArea(GetContentRect(rect));
            MetricItem[] metrics = GetMetricItems();
            for (int index = 0; index < metrics.Length; index++)
            {
                DrawMetricCard(metrics[index]);
                if (index < metrics.Length - 1)
                {
                    GUILayout.Space(8.0f);
                }
            }
            GUILayout.EndArea();
        }

        private void DrawBehaviorPanel(Rect rect)
        {
            PoultryTwinBehaviorSummary behavior = GetFocusBehaviorSummary();
            string subtitle = behavior != null && behavior.available
                ? "Dominant: " + HumanizeToken(behavior.dominant_behaviour)
                : "Behavior unavailable in this window";
            DrawFoldoutHeader(rect, "Behavior Mix", subtitle, ref behaviorExpanded);

            if (!behaviorExpanded)
            {
                return;
            }

            GUILayout.BeginArea(GetContentRect(rect));
            if (behavior == null || !behavior.available)
            {
                GUILayout.Label("Behavior proportions are unavailable for this window. This usually means the source video segment was night-time or had no usable behavior detections.", bodyStyle);
                GUILayout.Space(8.0f);
                GUILayout.Label("Try moving the timeline into a brighter daytime window to see active / feeding / drinking proportions.", smallStyle);
                GUILayout.EndArea();
                return;
            }

            GUILayout.Label("Dominant behavior: " + HumanizeToken(behavior.dominant_behaviour), sectionStyle);
            GUILayout.Space(4.0f);
            GUILayout.Label("Records: " + behavior.total_records + "    Unique tracks: " + behavior.unique_tracks + "    Mean birds: " + behavior.mean_detected_birds.ToString("0.0"), bodyStyle);
            GUILayout.Label("Observed seconds: " + behavior.observed_seconds + "    Mean confidence: " + behavior.mean_confidence.ToString("0.00"), bodyStyle);
            GUILayout.Space(10.0f);

            PoultryTwinBehaviorFraction[] mix = behavior.behavior_mix;
            if (mix == null || mix.Length == 0)
            {
                GUILayout.Label("No behavior fractions were stored for this window.", bodyStyle);
                GUILayout.EndArea();
                return;
            }

            PoultryTwinBehaviorFraction[] sortedMix = SortBehaviorMix(mix);
            for (int index = 0; index < sortedMix.Length; index++)
            {
                DrawBehaviorBar(sortedMix[index]);
            }

            GUILayout.EndArea();
        }

        private void DrawAbnormalityPanel(Rect rect)
        {
            string title = GetAbnormalityTitle();
            DrawFoldoutHeader(rect, "Abnormality", title, ref abnormalityExpanded);

            if (!abnormalityExpanded)
            {
                return;
            }

            GUILayout.BeginArea(GetContentRect(rect));
            GUILayout.Label(title, alertStyle);
            GUILayout.Space(8.0f);
            GUILayout.Label(GetAbnormalityDescription(), bodyStyle);
            GUILayout.EndArea();
        }

        private void DrawTimelinePanel(Rect rect)
        {
            string subtitle = dateBands.Count > 0
                ? dateBands[0].DisplayLabel + " to " + dateBands[dateBands.Count - 1].DisplayLabel
                : "Timeline unavailable";
            RegisterUiRect(rect);
            DrawPanel(rect);

            Rect headerRect = new Rect(rect.x + 12.0f, rect.y + 10.0f, rect.width - 24.0f, 56.0f);
            GUI.Label(new Rect(headerRect.x, headerRect.y + 2.0f, headerRect.width, 34.0f), "Timeline", foldoutTitleStyle);
            GUI.Label(new Rect(headerRect.x + 22.0f, headerRect.y + 34.0f, headerRect.width - 22.0f, 26.0f), subtitle, foldoutSubtitleStyle);
            DrawRectFill(new Rect(rect.x + 14.0f, rect.y + PanelHeaderHeight - 4.0f, rect.width - 28.0f, 1.0f), new Color(0.66f, 0.68f, 0.72f, 0.22f));

            Rect contentRect = GetContentRect(rect);
            GUI.BeginGroup(contentRect);

            float timelineWidth = contentRect.width;
            float barY = 4.0f;
            DrawDateBands(new Rect(0.0f, barY, timelineWidth, 28.0f));

            if (playbackController != null && playbackController.HasTimeline)
            {
                DrawTimelineScrubber(new Rect(0.0f, barY + 40.0f, timelineWidth, 18.0f));

                float detailY = barY + 62.0f;
                GUI.Label(
                    new Rect(0.0f, detailY, timelineWidth, 22.0f),
                    "Frame " + (playbackController.CurrentFrameIndex + 1) + " / " + playbackController.FrameCount + "    " + GetCurrentTimeLabel(),
                    bodyStyle
                );
                GUI.Label(
                    new Rect(0.0f, detailY + 24.0f, timelineWidth, 18.0f),
                    "10 s per window step at 1x | Space play/pause | Left/Right step | Up/Down speed | R reset",
                    smallStyle
                );
            }
            else
            {
                GUI.Label(new Rect(0.0f, barY + 62.0f, timelineWidth, 22.0f), "Timeline unavailable until JSON loads successfully.", bodyStyle);
            }

            GUI.EndGroup();
        }

        private void DrawTimelineScrubber(Rect rect)
        {
            int frameCount = playbackController != null ? playbackController.FrameCount : 0;
            if (frameCount <= 0)
            {
                return;
            }

            Event currentEvent = Event.current;
            if (currentEvent != null &&
                (currentEvent.type == EventType.MouseDown || currentEvent.type == EventType.MouseDrag) &&
                rect.Contains(currentEvent.mousePosition))
            {
                float normalizedMouseX = Mathf.Clamp01((currentEvent.mousePosition.x - rect.x) / Mathf.Max(1.0f, rect.width));
                int nextIndex = Mathf.RoundToInt(normalizedMouseX * Mathf.Max(0, frameCount - 1));
                if (nextIndex != playbackController.CurrentFrameIndex)
                {
                    playbackController.SetFrameIndex(nextIndex);
                }

                currentEvent.Use();
            }

            float progress = frameCount > 1
                ? Mathf.Clamp01((float)playbackController.CurrentFrameIndex / (frameCount - 1))
                : 0.0f;

            DrawRectFill(rect, new Color(0.80f, 0.82f, 0.84f, 0.76f));
            DrawRectOutline(rect, new Color(0.26f, 0.28f, 0.31f, 0.72f), 1.0f);
            Rect fillRect = new Rect(rect.x + 2.0f, rect.y + 2.0f, Mathf.Max(0.0f, (rect.width - 4.0f) * progress), rect.height - 4.0f);
            DrawRectFill(fillRect, new Color(0.44f, 0.58f, 0.70f, 0.86f));

            float handleX = rect.x + (rect.width * progress);
            DrawRectFill(new Rect(handleX - 2.0f, rect.y - 5.0f, 4.0f, rect.height + 10.0f), new Color(0.16f, 0.17f, 0.19f, 0.94f));
            DrawRectOutline(new Rect(handleX - 5.0f, rect.y - 4.0f, 10.0f, rect.height + 8.0f), new Color(1.0f, 1.0f, 1.0f, 0.76f), 1.0f);
        }

        private void DrawFoldoutHeader(Rect rect, string title, string subtitle, ref bool expanded)
        {
            RegisterUiRect(rect);
            DrawPanel(rect);

            Rect headerRect = new Rect(rect.x + 12.0f, rect.y + 10.0f, rect.width - 24.0f, 56.0f);
            if (GUI.Button(headerRect, GUIContent.none, headerButtonStyle))
            {
                expanded = !expanded;
            }
            if (Time.frameCount >= 0)
            {
                string modernFoldoutPrefix = expanded ? "v " : "> ";
                GUI.Label(new Rect(headerRect.x, headerRect.y + 2.0f, headerRect.width, 34.0f), modernFoldoutPrefix + title, foldoutTitleStyle);
                GUI.Label(new Rect(headerRect.x + 22.0f, headerRect.y + 34.0f, headerRect.width - 22.0f, 26.0f), subtitle, foldoutSubtitleStyle);
                DrawRectFill(new Rect(rect.x + 14.0f, rect.y + PanelHeaderHeight - 4.0f, rect.width - 28.0f, 1.0f), new Color(0.66f, 0.68f, 0.72f, 0.22f));
                return;
            }

            string foldoutPrefix = expanded ? "▼ " : "► ";
            GUI.Label(new Rect(headerRect.x + 10.0f, headerRect.y + 1.0f, headerRect.width - 20.0f, 18.0f), foldoutPrefix + title, foldoutTitleStyle);
            GUI.Label(new Rect(headerRect.x + 10.0f, headerRect.y + 19.0f, headerRect.width - 20.0f, 14.0f), subtitle, foldoutSubtitleStyle);
        }

        private void DrawMetricCard(MetricItem metric)
        {
            Rect rect = GUILayoutUtility.GetRect(10.0f, 72.0f, GUILayout.ExpandWidth(true));
            DrawSubPanel(rect, new Color(1.0f, 1.0f, 1.0f, 0.32f), new Color(0.72f, 0.74f, 0.78f, 0.42f));
            Rect inner = Shrink(rect, 10.0f);

            GUI.Label(new Rect(inner.x, inner.y, inner.width, 22.0f), metric.Label, metricLabelStyle);
            if (!metric.Available)
            {
                GUI.Label(new Rect(inner.x, inner.y + 20.0f, inner.width, 28.0f), "N/A", metricValueStyle);
                GUI.Label(new Rect(inner.x, inner.y + 48.0f, inner.width, 22.0f), metric.Description, smallStyle);
                return;
            }

            GUI.Label(new Rect(inner.x, inner.y + 18.0f, 86.0f, 28.0f), metric.Value.ToString("0.00"), metricValueStyle);
            DrawBar(new Rect(inner.x + 92.0f, inner.y + 24.0f, inner.width - 98.0f, 12.0f), metric.Value, metric.Label);
            GUI.Label(new Rect(inner.x, inner.y + 48.0f, inner.width, 22.0f), metric.Description, smallStyle);
        }

        private void DrawBehaviorBar(PoultryTwinBehaviorFraction fraction)
        {
            Rect rect = GUILayoutUtility.GetRect(10.0f, 40.0f, GUILayout.ExpandWidth(true));
            DrawSubPanel(rect, new Color(1.0f, 1.0f, 1.0f, 0.28f), new Color(0.74f, 0.76f, 0.80f, 0.34f));
            Rect inner = Shrink(rect, 8.0f);

            GUI.Label(new Rect(inner.x, inner.y, inner.width * 0.55f, 20.0f), Safe(fraction.label), smallStyle);
            GUI.Label(new Rect(inner.x + inner.width - 60.0f, inner.y, 60.0f, 20.0f), fraction.value.ToString("0%"), smallStyle);
            DrawBar(new Rect(inner.x, inner.y + 20.0f, inner.width, 10.0f), fraction.value, fraction.label);
        }

        private void DrawDateBands(Rect rect)
        {
            if (dateBands.Count == 0 || playbackController == null || playbackController.FrameCount == 0)
            {
                return;
            }

            float total = Mathf.Max(1.0f, playbackController.FrameCount);
            Color previous = GUI.color;

            for (int index = 0; index < dateBands.Count; index++)
            {
                DateBand band = dateBands[index];
                float x = rect.x + ((band.StartIndex / total) * rect.width);
                float width = (band.Count / total) * rect.width;
                Rect bandRect = new Rect(x, rect.y, Mathf.Max(16.0f, width - 3.0f), rect.height);
                DrawRectFill(bandRect, band.Color);
                DrawRectOutline(bandRect, new Color(1.0f, 1.0f, 1.0f, 0.16f), 1.0f);
                GUI.Label(new Rect(x + 8.0f, rect.y + 5.0f, Mathf.Max(80.0f, width - 16.0f), rect.height - 8.0f), band.DisplayLabel, smallStyle);
            }

            float progress = playbackController.FrameCount > 1
                ? (float)playbackController.CurrentFrameIndex / (playbackController.FrameCount - 1)
                : 0.0f;
            DrawRectFill(new Rect(rect.x + (progress * rect.width) - 1.5f, rect.y - 2.0f, 3.0f, rect.height + 4.0f), new Color(0.22f, 0.23f, 0.26f, 0.84f));
            GUI.color = previous;
        }

        private MetricItem[] GetMetricItems()
        {
            if (TryGetFocusZoneFrame(out PoultryTwinZoneFrame zoneFrame))
            {
                bool hasBehavior = zoneFrame.behavior_summary != null && zoneFrame.behavior_summary.available;
                return new[]
                {
                    new MetricItem
                    {
                        Label = "Activity",
                        Description = "Calibrated zone activity index",
                        Value = Mathf.Clamp01(GetZoneActivityValue(zoneFrame)),
                        Available = true,
                    },
                    new MetricItem
                    {
                        Label = "Occupancy",
                        Description = "Estimated share of visible birds",
                        Value = Mathf.Clamp01(zoneFrame.occupancy_share),
                        Available = true,
                    },
                    new MetricItem
                    {
                        Label = "Mean Birds",
                        Description = "Average visible birds in this zone",
                        Value = Mathf.Clamp01(zoneFrame.bird_count_mean / Mathf.Max(1.0f, GetBirdCountValue())),
                        Available = true,
                    },
                    new MetricItem
                    {
                        Label = "Confidence",
                        Description = "Behavior model confidence for this zone",
                        Value = hasBehavior ? Mathf.Clamp01(zoneFrame.behavior_summary.mean_confidence) : 0.0f,
                        Available = hasBehavior,
                    },
                };
            }

            float activity = 0.0f;
            float occupancy = 0.0f;
            float clustering = 0.0f;
            float risk = 0.0f;
            if (currentFrame != null && currentFrame.metrics != null)
            {
                activity = currentFrame.metrics.normalized_activity > 0.0f
                    ? currentFrame.metrics.normalized_activity
                    : currentFrame.metrics.activity_mean;
                occupancy = currentFrame.metrics.occupancy_imbalance_index;
                clustering = 1.0f - currentFrame.metrics.spatial_freedom_index;
            }

            if (currentFrame != null && currentFrame.welfare != null)
            {
                risk = currentFrame.welfare.risk_score;
            }

            return new[]
            {
                new MetricItem
                {
                    Label = "Activity",
                    Description = "Whole-room movement intensity",
                    Value = Mathf.Clamp01(activity),
                    Available = currentFrame != null && currentFrame.metrics != null,
                },
                new MetricItem
                {
                    Label = "Occupancy",
                    Description = "Distribution imbalance across zones",
                    Value = Mathf.Clamp01(occupancy),
                    Available = currentFrame != null && currentFrame.metrics != null,
                },
                new MetricItem
                {
                    Label = "Clustering",
                    Description = "Proxy from reduced spatial freedom",
                    Value = Mathf.Clamp01(clustering),
                    Available = currentFrame != null && currentFrame.metrics != null,
                },
                new MetricItem
                {
                    Label = "Risk",
                    Description = "Current welfare risk score",
                    Value = Mathf.Clamp01(risk),
                    Available = currentFrame != null && currentFrame.welfare != null,
                },
            };
        }

        private bool TryGetFocusZoneFrame(out PoultryTwinZoneFrame zoneFrame)
        {
            zoneFrame = null;
            ZoneOverlayController overlayController = playbackController != null ? playbackController.OverlayController : null;
            if (overlayController == null || !overlayController.HasSelectedZone)
            {
                return false;
            }

            return overlayController.TryGetSelectedZoneFrame(out zoneFrame);
        }

        private PoultryTwinBehaviorSummary GetFocusBehaviorSummary()
        {
            if (TryGetFocusZoneFrame(out PoultryTwinZoneFrame zoneFrame))
            {
                return zoneFrame.behavior_summary;
            }

            return currentFrame != null ? currentFrame.behavior_summary : null;
        }

        private string GetFocusLabel()
        {
            ZoneOverlayController overlayController = playbackController != null ? playbackController.OverlayController : null;
            if (overlayController != null && overlayController.HasSelectedZone)
            {
                return overlayController.SelectedZoneDisplayName;
            }

            return "Whole Room";
        }

        private string GetZonePanelSubtitle(PoultryTwinZoneFrame zoneFrame)
        {
            if (zoneFrame == null)
            {
                return "Selected zone";
            }

            if (string.IsNullOrEmpty(zoneFrame.source_zone_id))
            {
                return "Scene-only zone";
            }

            if (zoneFrame.is_proxy_zone)
            {
                return "Derived from " + Safe(zoneFrame.source_display_name);
            }

            return Safe(zoneFrame.source_display_name);
        }

        private string GetZoneDataNote(PoultryTwinZoneFrame zoneFrame)
        {
            if (zoneFrame == null)
            {
                return string.Empty;
            }

            if (string.IsNullOrEmpty(zoneFrame.source_zone_id))
            {
                return "This selection is part of the Unity scene layout only and is not currently linked to a semantic-zone export.";
            }

            if (zoneFrame.is_proxy_zone)
            {
                return "This selection is derived from an aggregated semantic-zone summary rather than a dedicated raw zone.";
            }

            return string.Empty;
        }

        private float GetZoneActivityValue(PoultryTwinZoneFrame zoneFrame)
        {
            if (zoneFrame == null)
            {
                return 0.0f;
            }

            ZoneOverlayController overlayController = playbackController != null ? playbackController.OverlayController : null;
            if (overlayController != null)
            {
                return overlayController.GetDisplayActivityValue(zoneFrame);
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

        private string GetAbnormalityTitle()
        {
            if (currentFrame == null)
            {
                return "Awaiting frame data";
            }

            if (TryGetFocusZoneFrame(out PoultryTwinZoneFrame zoneFrame))
            {
                float activity = GetZoneActivityValue(zoneFrame);
                if ((zoneFrame.zone_id == PoultryTwinRoomLayout.FeederZoneId || zoneFrame.zone_id == PoultryTwinRoomLayout.DrinkerZoneId) && activity < 0.28f)
                {
                    return "Poor feeder / drinker use";
                }

                if (zoneFrame.occupancy_share > 0.65f)
                {
                    return "Crowding proxy in selected zone";
                }
            }

            float roomActivity = currentFrame.metrics != null
                ? (currentFrame.metrics.normalized_activity > 0.0f ? currentFrame.metrics.normalized_activity : currentFrame.metrics.activity_mean)
                : 0.0f;
            float occupancy = currentFrame.metrics != null ? currentFrame.metrics.occupancy_imbalance_index : 0.0f;
            float clustering = currentFrame.metrics != null ? 1.0f - currentFrame.metrics.spatial_freedom_index : 0.0f;

            if (roomActivity < 0.18f)
            {
                return "Low activity example";
            }

            if (occupancy > 0.58f || clustering > 0.58f)
            {
                return "Crowding / clustering example";
            }

            return "Monitoring normal prototype state";
        }

        private string GetAbnormalityDescription()
        {
            string title = GetAbnormalityTitle();

            if (title == "Poor feeder / drinker use")
            {
                return "The current selected resource zone is showing low relative use. This is the type of abnormality card we can later hook to automated alerts and state predictions.";
            }

            if (title == "Crowding proxy in selected zone")
            {
                return "The selected zone is carrying a large share of visible birds. With richer tracking, this panel can evolve into a proper crowding alert.";
            }

            if (title == "Low activity example")
            {
                return "Whole-room movement intensity is currently low. This is a good candidate window for demonstrating a low-activity abnormality example.";
            }

            if (title == "Crowding / clustering example")
            {
                return "Occupancy imbalance and clustering are both elevated. This panel is ready to highlight crowding windows on the timeline.";
            }

            return "The prototype is currently in a normal monitoring state. Click a zone or scrub the timeline to inspect other windows.";
        }

        private string GetCurrentTimeLabel()
        {
            if (currentFrame == null)
            {
                return "n/a";
            }

            string start = FormatTimestamp(currentFrame.start_time);
            string end = FormatTimestamp(currentFrame.end_time);
            string date = !string.IsNullOrEmpty(currentFrame.local_date) ? FormatDateLabel(currentFrame.local_date) : "n/a";
            return date + "  " + start + " - " + end;
        }

        private string GetStateLabel()
        {
            return currentFrame != null && currentFrame.state != null
                ? HumanizeToken(currentFrame.state.state_label)
                : "n/a";
        }

        private float GetRiskValue()
        {
            return currentFrame != null && currentFrame.welfare != null ? currentFrame.welfare.risk_score : 0.0f;
        }

        private string GetRiskLevelLabel()
        {
            return currentFrame != null && currentFrame.welfare != null
                ? Safe(currentFrame.welfare.risk_level)
                : "n/a";
        }

        private float GetBirdCountValue()
        {
            if (currentFrame != null && currentFrame.bird_count > 0)
            {
                return currentFrame.bird_count;
            }

            return 30.0f;
        }

        private string GetBirdCountLabel()
        {
            return Mathf.RoundToInt(GetBirdCountValue()).ToString();
        }

        private string GetVisibleBirdProxyCountLabel()
        {
            if (currentFrame == null || currentFrame.birds == null)
            {
                return "0 / " + GetBirdCountLabel();
            }

            return currentFrame.birds.Length + " / " + GetBirdCountLabel();
        }

        private void RebuildDateBandsIfNeeded()
        {
            if (playbackController == null || playbackController.Data == null || playbackController.Data.timeline == null)
            {
                dateBands.Clear();
                cachedFrameCount = -1;
                return;
            }

            if (cachedFrameCount == playbackController.FrameCount && dateBands.Count > 0)
            {
                return;
            }

            dateBands.Clear();
            cachedFrameCount = playbackController.FrameCount;
            PoultryTwinTimelineFrame[] timeline = playbackController.Data.timeline;

            int startIndex = 0;
            while (startIndex < timeline.Length)
            {
                string rawDate = GetFrameDate(timeline[startIndex]);
                int endIndex = startIndex + 1;
                while (endIndex < timeline.Length && GetFrameDate(timeline[endIndex]) == rawDate)
                {
                    endIndex++;
                }

                dateBands.Add(
                    new DateBand
                    {
                        RawDate = rawDate,
                        DisplayLabel = FormatDateLabel(rawDate),
                        StartIndex = startIndex,
                        Count = endIndex - startIndex,
                        Color = GetDateBandColor(dateBands.Count),
                    }
                );
                startIndex = endIndex;
            }
        }

        private string GetFrameDate(PoultryTwinTimelineFrame frame)
        {
            if (frame == null)
            {
                return string.Empty;
            }

            if (!string.IsNullOrEmpty(frame.local_date))
            {
                return frame.local_date;
            }

            return !string.IsNullOrEmpty(frame.start_time) && frame.start_time.Length >= 10
                ? frame.start_time.Substring(0, 10)
                : string.Empty;
        }

        private Color GetDateBandColor(int index)
        {
            Color[] palette =
            {
                new Color(0.77f, 0.82f, 0.88f, 0.86f),
                new Color(0.78f, 0.86f, 0.79f, 0.86f),
                new Color(0.91f, 0.84f, 0.72f, 0.86f),
                new Color(0.89f, 0.78f, 0.78f, 0.86f),
            };

            return palette[index % palette.Length];
        }

        private PoultryTwinBehaviorFraction[] SortBehaviorMix(PoultryTwinBehaviorFraction[] mix)
        {
            PoultryTwinBehaviorFraction[] sorted = new PoultryTwinBehaviorFraction[mix.Length];
            Array.Copy(mix, sorted, mix.Length);
            Array.Sort(
                sorted,
                (left, right) =>
                {
                    float leftValue = left != null ? left.value : 0.0f;
                    float rightValue = right != null ? right.value : 0.0f;
                    return rightValue.CompareTo(leftValue);
                }
            );
            return sorted;
        }

        private void DrawBar(Rect rect, float value, string title)
        {
            Color previous = GUI.color;
            GUI.color = new Color(0.84f, 0.85f, 0.88f, 0.70f);
            GUI.DrawTexture(rect, solidTexture != null ? solidTexture : Texture2D.whiteTexture);
            GUI.color = MetricColor(title, value);
            GUI.DrawTexture(new Rect(rect.x + 2.0f, rect.y + 2.0f, Mathf.Max(0.0f, (rect.width - 4.0f) * Mathf.Clamp01(value)), rect.height - 4.0f), solidTexture != null ? solidTexture : Texture2D.whiteTexture);
            GUI.color = previous;
        }

        private Color MetricColor(string title, float value)
        {
            if (title == "Activity")
            {
                return Color.Lerp(new Color(0.46f, 0.56f, 0.62f), new Color(0.41f, 0.66f, 0.50f), value);
            }

            if (title == "Risk")
            {
                return Color.Lerp(new Color(0.70f, 0.71f, 0.74f), new Color(0.82f, 0.43f, 0.34f), value);
            }

            if (title == "Confidence")
            {
                return Color.Lerp(new Color(0.68f, 0.69f, 0.71f), new Color(0.80f, 0.69f, 0.39f), value);
            }

            return Color.Lerp(new Color(0.55f, 0.62f, 0.70f), new Color(0.86f, 0.72f, 0.44f), value);
        }

        private void DrawPanel(Rect rect)
        {
            DrawSubPanel(rect, new Color(0.94f, 0.94f, 0.94f, 0.58f), new Color(0.72f, 0.74f, 0.78f, 0.44f));
        }

        private void DrawSubPanel(Rect rect, Color fillColor, Color borderColor)
        {
            DrawRectFill(rect, fillColor);
            DrawRectOutline(rect, borderColor, 1.0f);
            DrawRectFill(new Rect(rect.x + 1.0f, rect.y + 1.0f, rect.width - 2.0f, 1.0f), new Color(1.0f, 1.0f, 1.0f, 0.18f));
        }

        private void RegisterUiRect(Rect rect)
        {
            interactiveRects.Add(rect);
        }

        private Rect GetContentRect(Rect rect)
        {
            return new Rect(rect.x + 14.0f, rect.y + PanelHeaderHeight + 8.0f, rect.width - 28.0f, rect.height - (PanelHeaderHeight + 16.0f));
        }

        private Rect Shrink(Rect rect, float padding)
        {
            return new Rect(rect.x + padding, rect.y + padding, rect.width - (padding * 2.0f), rect.height - (padding * 2.0f));
        }

        private void DrawRectFill(Rect rect, Color color)
        {
            if (solidTexture == null)
            {
                solidTexture = Texture2D.whiteTexture;
            }

            Color previous = GUI.color;
            GUI.color = color;
            GUI.DrawTexture(rect, solidTexture);
            GUI.color = previous;
        }

        private void DrawRectOutline(Rect rect, Color color, float thickness)
        {
            DrawRectFill(new Rect(rect.x, rect.y, rect.width, thickness), color);
            DrawRectFill(new Rect(rect.x, rect.yMax - thickness, rect.width, thickness), color);
            DrawRectFill(new Rect(rect.x, rect.y, thickness, rect.height), color);
            DrawRectFill(new Rect(rect.xMax - thickness, rect.y, thickness, rect.height), color);
        }

        private Rect GetAspectFitRect(Rect bounds, int textureWidth, int textureHeight)
        {
            if (textureWidth <= 0 || textureHeight <= 0)
            {
                return bounds;
            }

            float targetAspect = textureWidth / (float)textureHeight;
            float boundsAspect = bounds.width / bounds.height;
            if (boundsAspect > targetAspect)
            {
                float width = bounds.height * targetAspect;
                return new Rect(bounds.x + ((bounds.width - width) * 0.5f), bounds.y, width, bounds.height);
            }

            float height = bounds.width / targetAspect;
            return new Rect(bounds.x, bounds.y + ((bounds.height - height) * 0.5f), bounds.width, height);
        }

        private string HumanizeToken(string value)
        {
            if (string.IsNullOrEmpty(value))
            {
                return "n/a";
            }

            return value.Replace("_", " ");
        }

        private string FormatDateLabel(string rawDate)
        {
            if (DateTime.TryParse(rawDate, out DateTime parsed))
            {
                return parsed.ToString("MMM dd", CultureInfo.InvariantCulture);
            }

            return Safe(rawDate);
        }

        private string FormatTimestamp(string isoTime)
        {
            if (DateTimeOffset.TryParse(isoTime, out DateTimeOffset parsed))
            {
                return parsed.ToString("HH:mm:ss", CultureInfo.InvariantCulture);
            }

            return Safe(isoTime);
        }

        private void EnsureTextures()
        {
            if (solidTexture == null)
            {
                solidTexture = Texture2D.whiteTexture;
            }

            if (roomMapTexture != null)
            {
                return;
            }

            string imagePath = Path.Combine(Application.streamingAssetsPath, "room1_reference.png");
            if (!File.Exists(imagePath))
            {
                return;
            }

            byte[] bytes = File.ReadAllBytes(imagePath);
            roomMapTexture = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            roomMapTexture.name = "room1_reference.png";
            roomMapTexture.wrapMode = TextureWrapMode.Clamp;
            roomMapTexture.filterMode = FilterMode.Bilinear;
            roomMapTexture.LoadImage(bytes, false);
        }

        private void EnsureStyles()
        {
            if (panelStyle != null)
            {
                return;
            }

            panelStyle = new GUIStyle(GUI.skin.box);
            panelStyle.normal.textColor = new Color(0.18f, 0.19f, 0.22f);
            panelStyle.alignment = TextAnchor.MiddleCenter;
            panelStyle.border = new RectOffset(10, 10, 10, 10);

            headerButtonStyle = new GUIStyle(GUIStyle.none);

            titleStyle = new GUIStyle(GUI.skin.label);
            titleStyle.fontSize = 28;
            titleStyle.fontStyle = FontStyle.Bold;
            titleStyle.clipping = TextClipping.Overflow;
            titleStyle.normal.textColor = new Color(0.16f, 0.17f, 0.20f);

            sectionStyle = new GUIStyle(GUI.skin.label);
            sectionStyle.fontSize = 20;
            sectionStyle.fontStyle = FontStyle.Bold;
            sectionStyle.wordWrap = true;
            sectionStyle.clipping = TextClipping.Overflow;
            sectionStyle.normal.textColor = new Color(0.20f, 0.21f, 0.24f);

            bodyStyle = new GUIStyle(GUI.skin.label);
            bodyStyle.fontSize = 17;
            bodyStyle.wordWrap = true;
            bodyStyle.clipping = TextClipping.Overflow;
            bodyStyle.normal.textColor = new Color(0.24f, 0.25f, 0.28f);

            smallStyle = new GUIStyle(bodyStyle);
            smallStyle.fontSize = 13;
            smallStyle.clipping = TextClipping.Overflow;
            smallStyle.normal.textColor = new Color(0.40f, 0.42f, 0.46f);

            alertStyle = new GUIStyle(sectionStyle);
            alertStyle.fontSize = 20;
            alertStyle.clipping = TextClipping.Overflow;
            alertStyle.normal.textColor = new Color(0.76f, 0.38f, 0.29f);

            metricLabelStyle = new GUIStyle(GUI.skin.label);
            metricLabelStyle.fontSize = 16;
            metricLabelStyle.fontStyle = FontStyle.Bold;
            metricLabelStyle.clipping = TextClipping.Overflow;
            metricLabelStyle.normal.textColor = new Color(0.32f, 0.34f, 0.37f);

            metricValueStyle = new GUIStyle(GUI.skin.label);
            metricValueStyle.fontSize = 22;
            metricValueStyle.fontStyle = FontStyle.Bold;
            metricValueStyle.clipping = TextClipping.Overflow;
            metricValueStyle.normal.textColor = new Color(0.17f, 0.18f, 0.20f);

            foldoutTitleStyle = new GUIStyle(GUI.skin.label);
            foldoutTitleStyle.fontSize = 20;
            foldoutTitleStyle.fontStyle = FontStyle.Bold;
            foldoutTitleStyle.clipping = TextClipping.Overflow;
            foldoutTitleStyle.normal.textColor = new Color(0.17f, 0.18f, 0.20f);

            foldoutSubtitleStyle = new GUIStyle(GUI.skin.label);
            foldoutSubtitleStyle.fontSize = 13;
            foldoutSubtitleStyle.clipping = TextClipping.Overflow;
            foldoutSubtitleStyle.normal.textColor = new Color(0.46f, 0.48f, 0.52f);
        }

        private string Safe(string value)
        {
            return string.IsNullOrEmpty(value) ? "n/a" : value;
        }
    }
}
