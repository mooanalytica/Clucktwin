using System;

namespace PoultryTwinDemo
{
    [Serializable]
    public class PoultryTwinTimelineFile
    {
        public PoultryTwinMetadata metadata;
        public PoultryTwinRoom[] rooms;
        public PoultryTwinTimelineFrame[] timeline;

        public int TimelineCount
        {
            get { return timeline != null ? timeline.Length : 0; }
        }

        public PoultryTwinRoom GetPrimaryRoom()
        {
            if (rooms == null || rooms.Length == 0)
            {
                return null;
            }

            string preferredRoomId = null;
            if (timeline != null && timeline.Length > 0 && timeline[0] != null)
            {
                preferredRoomId = timeline[0].room_id;
            }

            if (!string.IsNullOrEmpty(preferredRoomId))
            {
                for (int i = 0; i < rooms.Length; i++)
                {
                    if (rooms[i] != null && rooms[i].room_id == preferredRoomId)
                    {
                        return rooms[i];
                    }
                }
            }

            return rooms[0];
        }
    }

    [Serializable]
    public class PoultryTwinMetadata
    {
        public string schema_version;
        public string created_at;
        public string source_feature_table;
        public string model_type;
        public string notes;
    }

    [Serializable]
    public class PoultryTwinRoom
    {
        public string room_id;
        public string display_name;
        public PoultryTwinZoneDefinition[] zones;
    }

    [Serializable]
    public class PoultryTwinZoneDefinition
    {
        public string zone_id;
        public string display_name;
        public int row;
        public int col;
        public PoultryTwinPoint[] polygon;
    }

    [Serializable]
    public class PoultryTwinPoint
    {
        public float x;
        public float y;
    }

    [Serializable]
    public class PoultryTwinTimelineFrame
    {
        public int frame_index;
        public string window_id;
        public string room_id;
        public string room_label;
        public int bird_count;
        public string start_time;
        public string end_time;
        public string local_date;
        public PoultryTwinMetrics metrics;
        public PoultryTwinState state;
        public PoultryTwinWelfare welfare;
        public PoultryTwinZoneFrame[] zones;
        public PoultryTwinEventInfo @event;
        public PoultryTwinBehaviorSummary behavior_summary;
        public PoultryTwinDashboardContext dashboard_context;
        public PoultryTwinBirdFrame[] birds;
    }

    [Serializable]
    public class PoultryTwinMetrics
    {
        public float mobility_index;
        public float spatial_freedom_index;
        public float occupancy_imbalance_index;
        public float activity_mean;
        public float normalized_activity;
    }

    [Serializable]
    public class PoultryTwinState
    {
        public int state_id;
        public string state_label;
        public float state_probability;
    }

    [Serializable]
    public class PoultryTwinWelfare
    {
        public float risk_score;
        public string risk_level;
        public bool sustained_risk_flag;
    }

    [Serializable]
    public class PoultryTwinZoneFrame
    {
        public string zone_id;
        public string display_name;
        public string source_zone_id;
        public string source_display_name;
        public bool is_proxy_zone;
        public float activity;
        public float activity_norm;
        public float overlay_intensity;
        public float occupancy_share;
        public float bird_count_mean;
        public PoultryTwinBehaviorSummary behavior_summary;
    }

    [Serializable]
    public class PoultryTwinEventInfo
    {
        public string event_id;
        public string event_phase;
        public string event_type;
    }

    [Serializable]
    public class PoultryTwinBehaviorSummary
    {
        public bool available;
        public int total_records;
        public int unique_tracks;
        public int observed_seconds;
        public float mean_detected_birds;
        public float mean_confidence;
        public string dominant_behaviour;
        public PoultryTwinBehaviorFraction[] behavior_mix;
    }

    [Serializable]
    public class PoultryTwinBehaviorFraction
    {
        public string behavior_id;
        public string label;
        public float value;
        public int count;
    }

    [Serializable]
    public class PoultryTwinDashboardContext
    {
        public string data_zone_schema;
        public string timeline_date_label;
        public string source_video;
    }

    [Serializable]
    public class PoultryTwinBirdFrame
    {
        public string bird_id;
        public string track_id;
        public string zone_id;
        public string source_zone_id;
        public string behavior;
        public float confidence;
        public float x_norm;
        public float y_norm;
        public float world_x;
        public float world_z;
        public int observation_count;
    }
}
