using UnityEngine;

namespace PoultryTwinDemo
{
    public class PoultryTwinCameraController : MonoBehaviour
    {
        [SerializeField] private float orbitRotateSpeed = 3.2f;
        [SerializeField] private float orbitZoomSpeed = 2.1f;
        [SerializeField] private float orbitPanSpeed = 0.012f;
        [SerializeField] private float orbitTruckSpeed = 5.5f;
        [SerializeField] private float moveSpeed = 8.0f;
        [SerializeField] private float fastMoveMultiplier = 2.4f;
        [SerializeField] private float verticalSpeed = 5.0f;
        [SerializeField] private float rotationSpeed = 95.0f;
        [SerializeField] private float minPitch = 18.0f;
        [SerializeField] private float maxPitch = 85.0f;
        [SerializeField] private float minOrbitDistance = 5.5f;
        [SerializeField] private float maxOrbitDistance = 24.0f;

        private Vector3 presentationPosition;
        private Vector3 presentationEulerAngles;
        private Vector3 presentationTarget;
        private Vector3 orbitTarget;
        private float orbitDistance;
        private float yaw;
        private float pitch;

        public static PoultryTwinCameraController ActiveInstance { get; private set; }

        public bool IsDebugMode { get; private set; }

        public bool IsNavigatingScene { get; private set; }

        private void Awake()
        {
            ActiveInstance = this;
            SyncAnglesFromTransform();
        }

        private void OnDestroy()
        {
            if (ActiveInstance == this)
            {
                ActiveInstance = null;
            }
        }

        private void Update()
        {
            HandleModeSwitches();

            if (IsDebugMode)
            {
                HandleFreeFlyNavigation();
            }
            else
            {
                HandleOrbitNavigation();
            }
        }

        public void SetPresentationView(Vector3 position, Vector3 eulerAngles)
        {
            SetPresentationView(position, eulerAngles, Vector3.zero);
        }

        public void SetPresentationView(Vector3 position, Vector3 eulerAngles, Vector3 focusPoint)
        {
            presentationPosition = position;
            presentationEulerAngles = eulerAngles;
            presentationTarget = EstimateLookTarget(position, eulerAngles, focusPoint);

            if (!IsDebugMode)
            {
                ResetToPresentationView();
            }
        }

        public void ResetToPresentationView()
        {
            IsDebugMode = false;
            transform.position = presentationPosition;
            transform.rotation = Quaternion.Euler(presentationEulerAngles);
            orbitTarget = presentationTarget;
            SyncAnglesFromTransform();
            orbitDistance = Mathf.Clamp(Vector3.Distance(transform.position, orbitTarget), minOrbitDistance, maxOrbitDistance);
        }

        private void HandleModeSwitches()
        {
            if (PoultryTwinInputAdapter.GetKeyDown(KeyCode.Tab))
            {
                IsDebugMode = !IsDebugMode;
                if (IsDebugMode)
                {
                    SyncAnglesFromTransform();
                }
                else
                {
                    orbitTarget = EstimateLookTarget(transform.position, transform.eulerAngles, presentationTarget);
                    SyncAnglesFromTransform();
                    orbitDistance = Mathf.Clamp(Vector3.Distance(transform.position, orbitTarget), minOrbitDistance, maxOrbitDistance);
                }
            }

            if (PoultryTwinInputAdapter.GetKeyDown(KeyCode.F))
            {
                ResetToPresentationView();
            }
        }

        private void HandleOrbitNavigation()
        {
            IsNavigatingScene = false;
            Vector2 mousePosition;
            bool hasMousePosition = PoultryTwinInputAdapter.TryGetMousePosition(out mousePosition);
            bool pointerOverHud = hasMousePosition && DemoHudController.ActiveInstance != null && DemoHudController.ActiveInstance.ContainsScreenPoint(mousePosition);

            if (hasMousePosition && !pointerOverHud)
            {
                if (PoultryTwinInputAdapter.GetMouseButton(1))
                {
                    Vector2 mouseDelta = PoultryTwinInputAdapter.GetMouseDelta();
                    yaw += mouseDelta.x * orbitRotateSpeed * 120.0f * Time.deltaTime;
                    pitch -= mouseDelta.y * orbitRotateSpeed * 120.0f * Time.deltaTime;
                    pitch = Mathf.Clamp(pitch, minPitch, maxPitch);
                    ApplyOrbitTransform();
                    IsNavigatingScene = true;
                }

                if (PoultryTwinInputAdapter.GetMouseButton(2))
                {
                    PanOrbitTarget();
                    ApplyOrbitTransform();
                    IsNavigatingScene = true;
                }

                float wheel = PoultryTwinInputAdapter.GetMouseScrollY();
                if (Mathf.Abs(wheel) > 0.001f)
                {
                    orbitDistance = Mathf.Clamp(orbitDistance - (wheel * orbitZoomSpeed), minOrbitDistance, maxOrbitDistance);
                    ApplyOrbitTransform();
                }
            }

            float horizontal = 0.0f;
            float vertical = 0.0f;
            if (PoultryTwinInputAdapter.GetKey(KeyCode.A))
            {
                horizontal -= 1.0f;
            }

            if (PoultryTwinInputAdapter.GetKey(KeyCode.D))
            {
                horizontal += 1.0f;
            }

            if (PoultryTwinInputAdapter.GetKey(KeyCode.W))
            {
                vertical += 1.0f;
            }

            if (PoultryTwinInputAdapter.GetKey(KeyCode.S))
            {
                vertical -= 1.0f;
            }

            if (Mathf.Abs(horizontal) > 0.001f || Mathf.Abs(vertical) > 0.001f)
            {
                Vector3 planarForward = Vector3.ProjectOnPlane(transform.forward, Vector3.up).normalized;
                Vector3 planarRight = Vector3.ProjectOnPlane(transform.right, Vector3.up).normalized;
                orbitTarget += ((planarRight * horizontal) + (planarForward * vertical)) * orbitTruckSpeed * Time.deltaTime;
                ApplyOrbitTransform();
            }
        }

        private void HandleFreeFlyNavigation()
        {
            Vector2 mousePosition;
            bool hasMousePosition = PoultryTwinInputAdapter.TryGetMousePosition(out mousePosition);
            bool pointerOverHud = hasMousePosition && DemoHudController.ActiveInstance != null && DemoHudController.ActiveInstance.ContainsScreenPoint(mousePosition);
            IsNavigatingScene = hasMousePosition && !pointerOverHud && (PoultryTwinInputAdapter.GetMouseButton(1) || PoultryTwinInputAdapter.GetMouseButton(2));

            HandleTranslation();
            if (hasMousePosition && !pointerOverHud)
            {
                HandleRotation();
                HandleZoom();
            }
        }

        private void PanOrbitTarget()
        {
            Vector2 mouseDelta = PoultryTwinInputAdapter.GetMouseDelta();
            float deltaX = mouseDelta.x;
            float deltaY = mouseDelta.y;
            Vector3 planarRight = Vector3.ProjectOnPlane(transform.right, Vector3.up).normalized;
            Vector3 planarUp = Vector3.ProjectOnPlane(transform.up, Vector3.up).normalized;
            if (planarUp.sqrMagnitude < 0.001f)
            {
                planarUp = -Vector3.ProjectOnPlane(transform.forward, Vector3.up).normalized;
            }

            float scale = orbitDistance * orbitPanSpeed;
            orbitTarget -= planarRight * deltaX * scale * 60.0f * Time.deltaTime;
            orbitTarget -= planarUp * deltaY * scale * 60.0f * Time.deltaTime;
        }

        private void ApplyOrbitTransform()
        {
            transform.rotation = Quaternion.Euler(pitch, yaw, 0.0f);
            transform.position = orbitTarget - (transform.forward * orbitDistance);
        }

        private void HandleTranslation()
        {
            Vector3 planarForward = Vector3.ProjectOnPlane(transform.forward, Vector3.up).normalized;
            Vector3 planarRight = Vector3.ProjectOnPlane(transform.right, Vector3.up).normalized;

            float horizontal = 0.0f;
            float vertical = 0.0f;
            if (PoultryTwinInputAdapter.GetKey(KeyCode.A))
            {
                horizontal -= 1.0f;
            }

            if (PoultryTwinInputAdapter.GetKey(KeyCode.D))
            {
                horizontal += 1.0f;
            }

            if (PoultryTwinInputAdapter.GetKey(KeyCode.W))
            {
                vertical += 1.0f;
            }

            if (PoultryTwinInputAdapter.GetKey(KeyCode.S))
            {
                vertical -= 1.0f;
            }

            float rise = 0.0f;
            if (PoultryTwinInputAdapter.GetKey(KeyCode.E))
            {
                rise += 1.0f;
            }

            if (PoultryTwinInputAdapter.GetKey(KeyCode.Q))
            {
                rise -= 1.0f;
            }

            Vector3 motion = (planarForward * vertical) + (planarRight * horizontal) + (Vector3.up * rise);
            if (motion.sqrMagnitude < 0.0001f)
            {
                return;
            }

            float speed = moveSpeed;
            if (PoultryTwinInputAdapter.GetKey(KeyCode.LeftShift) || PoultryTwinInputAdapter.GetKey(KeyCode.RightShift))
            {
                speed *= fastMoveMultiplier;
            }

            motion = motion.normalized;
            float verticalFactor = Mathf.Abs(rise) > 0.001f ? verticalSpeed / Mathf.Max(0.1f, moveSpeed) : 1.0f;
            transform.position += motion * speed * verticalFactor * Time.deltaTime;
        }

        private void HandleRotation()
        {
            if (!PoultryTwinInputAdapter.GetMouseButton(1))
            {
                return;
            }

            Vector2 mouseDelta = PoultryTwinInputAdapter.GetMouseDelta();
            yaw += mouseDelta.x * rotationSpeed * Time.deltaTime;
            pitch -= mouseDelta.y * rotationSpeed * Time.deltaTime;
            pitch = Mathf.Clamp(pitch, minPitch, maxPitch);
            transform.rotation = Quaternion.Euler(pitch, yaw, 0.0f);
        }

        private void HandleZoom()
        {
            float wheel = PoultryTwinInputAdapter.GetMouseScrollY();
            if (Mathf.Abs(wheel) < 0.001f)
            {
                return;
            }

            transform.position += transform.forward * wheel * orbitZoomSpeed;
        }

        private Vector3 EstimateLookTarget(Vector3 position, Vector3 eulerAngles, Vector3 fallbackTarget)
        {
            Vector3 safeFallback = fallbackTarget == Vector3.zero ? PoultryTwinRoomLayout.CameraFocusPoint : fallbackTarget;
            Quaternion rotation = Quaternion.Euler(eulerAngles);
            Ray ray = new Ray(position, rotation * Vector3.forward);
            Plane floorPlane = new Plane(Vector3.up, new Vector3(0.0f, safeFallback.y, 0.0f));
            float distance;
            if (floorPlane.Raycast(ray, out distance))
            {
                Vector3 hitPoint = ray.GetPoint(distance);
                if ((hitPoint - position).sqrMagnitude > 0.01f)
                {
                    return hitPoint;
                }
            }

            return safeFallback;
        }

        private void SyncAnglesFromTransform()
        {
            Vector3 euler = transform.eulerAngles;
            yaw = euler.y;
            pitch = NormalizePitch(euler.x);
        }

        private float NormalizePitch(float rawPitch)
        {
            if (rawPitch > 180.0f)
            {
                rawPitch -= 360.0f;
            }

            return Mathf.Clamp(rawPitch, minPitch, maxPitch);
        }
    }
}
