using UnityEngine;

namespace PoultryTwinDemo
{
    public class PoultryTwinPlaybackController : MonoBehaviour
    {
        [SerializeField] private PoultryTwinJsonLoader jsonLoader;
        [SerializeField] private ZoneOverlayController zoneOverlayController;
        [SerializeField] private DemoHudController demoHudController;
        [SerializeField] private bool playOnStart = true;
        [SerializeField] private float secondsPerFrame = 10.0f;
        [SerializeField] private float minSpeed = 0.25f;
        [SerializeField] private float maxSpeed = 4.0f;
        [SerializeField] private float speedStep = 0.25f;

        private float playbackTimer;
        private string initializationStatus = "Waiting for initialization.";

        public bool IsPlaying { get; private set; }
        public int CurrentFrameIndex { get; private set; }
        public float CurrentSpeed { get; private set; } = 1.0f;

        public PoultryTwinTimelineFile Data
        {
            get { return jsonLoader != null ? jsonLoader.Data : null; }
        }

        public int FrameCount
        {
            get { return Data != null && Data.timeline != null ? Data.timeline.Length : 0; }
        }

        public bool HasTimeline
        {
            get { return FrameCount > 0; }
        }

        public string InitializationStatus
        {
            get { return initializationStatus; }
        }

        public string JsonPath
        {
            get { return jsonLoader != null ? jsonLoader.GetResolvedPath() : string.Empty; }
        }

        public string LoaderErrorMessage
        {
            get { return jsonLoader != null ? jsonLoader.LastErrorMessage : "JSON loader component is missing."; }
        }

        public string LoaderWarningMessage
        {
            get { return jsonLoader != null ? jsonLoader.LastWarningMessage : string.Empty; }
        }

        public string OverlayErrorMessage
        {
            get { return zoneOverlayController != null ? zoneOverlayController.LastErrorMessage : "Zone overlay controller is missing."; }
        }

        public string OverlayStatusMessage
        {
            get { return zoneOverlayController != null ? zoneOverlayController.StatusMessage : string.Empty; }
        }

        public bool HasVisibleZones
        {
            get { return zoneOverlayController != null && zoneOverlayController.HasZones; }
        }

        public ZoneOverlayController OverlayController
        {
            get { return zoneOverlayController; }
        }

        public float TimelineProgress01
        {
            get
            {
                if (!HasTimeline || FrameCount <= 1)
                {
                    return 0.0f;
                }

                return Mathf.Clamp01((float)CurrentFrameIndex / (FrameCount - 1));
            }
        }

        public float FrameDurationSeconds
        {
            get { return Mathf.Max(0.05f, secondsPerFrame); }
        }

        public float DisplayFrameDurationSeconds
        {
            get { return FrameDurationSeconds / Mathf.Max(0.01f, CurrentSpeed); }
        }

        public string CurrentAgeWindow
        {
            get
            {
                if (!HasTimeline)
                {
                    return "n/a";
                }

                float progress = TimelineProgress01;
                if (progress < 0.334f)
                {
                    return "Early";
                }

                if (progress < 0.667f)
                {
                    return "Middle";
                }

                return "Late";
            }
        }

        private void Start()
        {
            ResolveReferences();

            if (demoHudController != null)
            {
                demoHudController.Initialize(this);
            }

            if (jsonLoader == null)
            {
                initializationStatus = "JSON loader component is missing.";
                RefreshHud();
                return;
            }

            if (!jsonLoader.IsLoaded && !jsonLoader.TryLoad())
            {
                initializationStatus = string.IsNullOrEmpty(jsonLoader.LastErrorMessage)
                    ? "JSON load failed."
                    : jsonLoader.LastErrorMessage;
                RefreshHud();
                return;
            }

            if (!HasTimeline)
            {
                initializationStatus = "Timeline empty.";
                RefreshHud();
                return;
            }

            if (zoneOverlayController == null)
            {
                initializationStatus = "Zone overlay controller is missing.";
                RefreshHud();
                return;
            }

            zoneOverlayController.Initialize(Data);
            initializationStatus = zoneOverlayController.HasZones
                ? "Ready."
                : string.IsNullOrEmpty(zoneOverlayController.StatusMessage)
                    ? "No zones found."
                    : zoneOverlayController.StatusMessage;

            CurrentSpeed = Mathf.Clamp(CurrentSpeed, minSpeed, maxSpeed);
            IsPlaying = playOnStart;
            ApplyFrame(0, true);
            RefreshHud();
        }

        private void Update()
        {
            HandleKeyboardInput();

            if (!HasTimeline || !IsPlaying)
            {
                return;
            }

            float frameDuration = FrameDurationSeconds;
            playbackTimer += Time.deltaTime * CurrentSpeed;
            while (playbackTimer >= frameDuration && HasTimeline)
            {
                playbackTimer -= frameDuration;
                StepInternal(1, false, false);
            }
        }

        public void SetReferences(PoultryTwinJsonLoader loader, ZoneOverlayController overlayController, DemoHudController hudController)
        {
            jsonLoader = loader;
            zoneOverlayController = overlayController;
            demoHudController = hudController;
        }

        public PoultryTwinTimelineFrame GetCurrentFrame()
        {
            if (!HasTimeline)
            {
                return null;
            }

            int clampedIndex = Mathf.Clamp(CurrentFrameIndex, 0, FrameCount - 1);
            return Data.timeline[clampedIndex];
        }

        public void TogglePlayPause()
        {
            IsPlaying = !IsPlaying;
            playbackTimer = 0.0f;
            RefreshHud();
        }

        public void Step(int delta)
        {
            StepInternal(delta, true, true);
        }

        private void StepInternal(int delta, bool snapBirds)
        {
            StepInternal(delta, snapBirds, true);
        }

        private void StepInternal(int delta, bool snapBirds, bool resetTimer)
        {
            if (!HasTimeline)
            {
                return;
            }

            int nextIndex = CurrentFrameIndex + delta;
            if (nextIndex >= FrameCount)
            {
                nextIndex = 0;
            }
            else if (nextIndex < 0)
            {
                nextIndex = FrameCount - 1;
            }

            ApplyFrame(nextIndex, resetTimer, snapBirds);
        }

        public void ResetToStart()
        {
            if (!HasTimeline)
            {
                return;
            }

            ApplyFrame(0, true, true);
        }

        public void SetFrameIndex(int frameIndex)
        {
            if (!HasTimeline)
            {
                return;
            }

            ApplyFrame(Mathf.Clamp(frameIndex, 0, FrameCount - 1), true, true);
        }

        public void AdjustSpeed(int direction)
        {
            CurrentSpeed = Mathf.Clamp(CurrentSpeed + (direction * speedStep), minSpeed, maxSpeed);
            RefreshHud();
        }

        private void ApplyFrame(int frameIndex, bool resetTimer)
        {
            ApplyFrame(frameIndex, resetTimer, false);
        }

        private void ApplyFrame(int frameIndex, bool resetTimer, bool snapBirds)
        {
            if (!HasTimeline)
            {
                return;
            }

            CurrentFrameIndex = Mathf.Clamp(frameIndex, 0, FrameCount - 1);
            if (resetTimer)
            {
                playbackTimer = 0.0f;
            }

            PoultryTwinTimelineFrame frame = Data.timeline[CurrentFrameIndex];
            if (zoneOverlayController != null)
            {
                zoneOverlayController.ApplyFrame(frame, snapBirds ? 0.0f : DisplayFrameDurationSeconds);
            }

            if (demoHudController != null)
            {
                demoHudController.ShowFrame(frame);
            }
        }

        private void HandleKeyboardInput()
        {
            if (PoultryTwinInputAdapter.GetKeyDown(KeyCode.Space))
            {
                TogglePlayPause();
            }

            if (PoultryTwinInputAdapter.GetKeyDown(KeyCode.LeftArrow))
            {
                Step(-1);
            }

            if (PoultryTwinInputAdapter.GetKeyDown(KeyCode.RightArrow))
            {
                Step(1);
            }

            if (PoultryTwinInputAdapter.GetKeyDown(KeyCode.UpArrow))
            {
                AdjustSpeed(1);
            }

            if (PoultryTwinInputAdapter.GetKeyDown(KeyCode.DownArrow))
            {
                AdjustSpeed(-1);
            }

            if (PoultryTwinInputAdapter.GetKeyDown(KeyCode.R))
            {
                ResetToStart();
            }
        }

        private void ResolveReferences()
        {
            if (jsonLoader == null)
            {
                jsonLoader = GetComponent<PoultryTwinJsonLoader>();
            }

            if (zoneOverlayController == null)
            {
                zoneOverlayController = GetComponent<ZoneOverlayController>();
            }

            if (demoHudController == null)
            {
                demoHudController = GetComponent<DemoHudController>();
            }
        }

        private void RefreshHud()
        {
            if (demoHudController != null)
            {
                demoHudController.Refresh();
            }
        }
    }
}
