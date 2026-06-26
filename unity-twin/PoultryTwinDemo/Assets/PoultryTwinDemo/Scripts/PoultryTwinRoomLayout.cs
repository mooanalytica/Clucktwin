using System;
using UnityEngine;

namespace PoultryTwinDemo
{
    public static class PoultryTwinRoomLayout
    {
        public const string RoomIdRoom1 = "room_1";
        public const string DrinkerZoneId = "drinker";
        public const string FeederZoneId = "feeder";
        public const string RestingZoneId = "resting_floor";
        public const string OpenMovementZoneId = "open_movement";

        public static readonly Vector2 RoomSize = new Vector2(10.8f, 16.4f);
        public static readonly Vector3 CameraPosition = new Vector3(-7.8f, 11.6f, -10.8f);
        public static readonly Vector3 CameraRotation = new Vector3(56.0f, 33.0f, 0.0f);
        public static readonly Vector3 CameraFocusPoint = new Vector3(0.0f, 0.65f, 0.35f);
        public const float CameraFieldOfView = 38.0f;

        private static readonly ZoneProfile[] Room1Profiles =
        {
            new ZoneProfile(
                DrinkerZoneId,
                "Drinker",
                new Color(0.94f, 0.30f, 0.24f),
                new[]
                {
                    new Rect(-3.2063f, 0.4404f, 1.0913f, 5.4819f),
                },
                new[] { "zone_A", "drinker_zone", "drinking_area" }
            ),
            new ZoneProfile(
                FeederZoneId,
                "Feeder",
                new Color(0.32f, 0.78f, 0.34f),
                new[]
                {
                    new Rect(-2.0981f, -1.6248f, 1.5469f, 5.0415f),
                },
                new[] { "zone_C", "feeder_zone", "feeding_zone", "feeding_area" }
            ),
            new ZoneProfile(
                RestingZoneId,
                "Resting / Floor",
                new Color(0.30f, 0.66f, 0.92f),
                new[]
                {
                    new Rect(-2.4244f, -5.9830f, 3.7294f, 4.1152f),
                    new Rect(1.3106f, -5.9830f, 1.6369f, 13.7730f),
                },
                new[] { "zone_B", "resting_zone", "resting_area", "resting_floor", "perching_area" }
            ),
            new ZoneProfile(
                OpenMovementZoneId,
                "Open Movement",
                new Color(0.98f, 0.75f, 0.24f),
                new[]
                {
                    new Rect(-3.2063f, 6.0133f, 1.1250f, 1.7463f),
                    new Rect(-2.0981f, 3.4470f, 3.3862f, 4.3126f),
                    new Rect(-0.5344f, -1.8222f, 1.8225f, 5.2693f),
                    new Rect(-3.2063f, -1.7767f, 1.0856f, 2.1563f),
                },
                new[] { "zone_D", "open_area", "general_area", "movement_area", "open_movement_zone" }
            ),
        };

        public static ZoneProfile[] GetRoom1Profiles()
        {
            ZoneProfile[] clone = new ZoneProfile[Room1Profiles.Length];
            Array.Copy(Room1Profiles, clone, Room1Profiles.Length);
            return clone;
        }

        public static bool TryMapExternalZoneId(string sourceZoneId, out string semanticZoneId)
        {
            semanticZoneId = sourceZoneId;
            if (string.IsNullOrEmpty(sourceZoneId))
            {
                return false;
            }

            for (int i = 0; i < Room1Profiles.Length; i++)
            {
                ZoneProfile profile = Room1Profiles[i];
                if (profile.SemanticZoneId == sourceZoneId)
                {
                    semanticZoneId = profile.SemanticZoneId;
                    return true;
                }

                string[] aliases = profile.Aliases;
                for (int aliasIndex = 0; aliasIndex < aliases.Length; aliasIndex++)
                {
                    if (string.Equals(aliases[aliasIndex], sourceZoneId, StringComparison.OrdinalIgnoreCase))
                    {
                        semanticZoneId = profile.SemanticZoneId;
                        return true;
                    }
                }
            }

            return false;
        }

        public static ZoneProfile GetProfileOrDefault(string zoneId)
        {
            for (int i = 0; i < Room1Profiles.Length; i++)
            {
                if (Room1Profiles[i].SemanticZoneId == zoneId)
                {
                    return Room1Profiles[i];
                }
            }

            return default;
        }

        public static Rect WorldRectToNormalizedOverlayRect(Rect worldRect)
        {
            float halfWidth = RoomSize.x * 0.5f;
            float halfDepth = RoomSize.y * 0.5f;

            float xMin = Mathf.Clamp01((worldRect.xMin + halfWidth) / RoomSize.x);
            float xMax = Mathf.Clamp01((worldRect.xMax + halfWidth) / RoomSize.x);
            float top = Mathf.Clamp01(1.0f - ((worldRect.yMax + halfDepth) / RoomSize.y));
            float bottom = Mathf.Clamp01(1.0f - ((worldRect.yMin + halfDepth) / RoomSize.y));
            return Rect.MinMaxRect(xMin, top, xMax, bottom);
        }

        public readonly struct ZoneProfile
        {
            public readonly string SemanticZoneId;
            public readonly string DisplayName;
            public readonly Color AccentColor;
            public readonly Rect[] WorldRects;
            public readonly string[] Aliases;

            public ZoneProfile(string semanticZoneId, string displayName, Color accentColor, Rect[] worldRects, string[] aliases)
            {
                SemanticZoneId = semanticZoneId;
                DisplayName = displayName;
                AccentColor = accentColor;
                WorldRects = worldRects;
                Aliases = aliases;
            }

            public Rect[] GetNormalizedOverlayRects()
            {
                Rect[] normalizedRects = new Rect[WorldRects.Length];
                for (int index = 0; index < WorldRects.Length; index++)
                {
                    normalizedRects[index] = WorldRectToNormalizedOverlayRect(WorldRects[index]);
                }

                return normalizedRects;
            }
        }
    }
}
