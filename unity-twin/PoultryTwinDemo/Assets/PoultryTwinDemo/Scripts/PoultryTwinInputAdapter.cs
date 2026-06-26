using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace PoultryTwinDemo
{
    public static class PoultryTwinInputAdapter
    {
        private const float MouseDeltaScale = 0.02f;
        private const float ScrollScale = 1.0f / 120.0f;

        public static bool GetKeyDown(KeyCode keyCode)
        {
#if ENABLE_INPUT_SYSTEM
            Key mappedKey;
            if (TryMapKey(keyCode, out mappedKey) && Keyboard.current != null)
            {
                return Keyboard.current[mappedKey].wasPressedThisFrame;
            }
            return false;
#endif
#if !ENABLE_INPUT_SYSTEM
            return Input.GetKeyDown(keyCode);
#endif
        }

        public static bool GetKey(KeyCode keyCode)
        {
#if ENABLE_INPUT_SYSTEM
            Key mappedKey;
            if (TryMapKey(keyCode, out mappedKey) && Keyboard.current != null)
            {
                return Keyboard.current[mappedKey].isPressed;
            }
            return false;
#endif
#if !ENABLE_INPUT_SYSTEM
            return Input.GetKey(keyCode);
#endif
        }

        public static bool GetMouseButtonDown(int button)
        {
#if ENABLE_INPUT_SYSTEM
            if (Mouse.current != null)
            {
                switch (button)
                {
                    case 0:
                        return Mouse.current.leftButton.wasPressedThisFrame;
                    case 1:
                        return Mouse.current.rightButton.wasPressedThisFrame;
                    case 2:
                        return Mouse.current.middleButton.wasPressedThisFrame;
                }
            }
            return false;
#endif
#if !ENABLE_INPUT_SYSTEM
            return Input.GetMouseButtonDown(button);
#endif
        }

        public static bool GetMouseButton(int button)
        {
#if ENABLE_INPUT_SYSTEM
            if (Mouse.current != null)
            {
                switch (button)
                {
                    case 0:
                        return Mouse.current.leftButton.isPressed;
                    case 1:
                        return Mouse.current.rightButton.isPressed;
                    case 2:
                        return Mouse.current.middleButton.isPressed;
                }
            }
            return false;
#endif
#if !ENABLE_INPUT_SYSTEM
            return Input.GetMouseButton(button);
#endif
        }

        public static Vector2 GetMousePosition()
        {
#if ENABLE_INPUT_SYSTEM
            if (Mouse.current != null)
            {
                return Mouse.current.position.ReadValue();
            }
            return Vector2.zero;
#endif
#if !ENABLE_INPUT_SYSTEM
            return Input.mousePosition;
#endif
        }

        public static bool TryGetMousePosition(out Vector2 position)
        {
            position = GetMousePosition();
            return IsValidScreenPoint(position);
        }

        public static bool IsValidScreenPoint(Vector2 screenPoint)
        {
            if (Screen.width <= 0 || Screen.height <= 0)
            {
                return false;
            }

            if (float.IsNaN(screenPoint.x) || float.IsNaN(screenPoint.y) ||
                float.IsInfinity(screenPoint.x) || float.IsInfinity(screenPoint.y))
            {
                return false;
            }

            return screenPoint.x >= 0.0f &&
                screenPoint.y >= 0.0f &&
                screenPoint.x <= Screen.width &&
                screenPoint.y <= Screen.height;
        }

        public static Vector2 GetMouseDelta()
        {
#if ENABLE_INPUT_SYSTEM
            if (Mouse.current != null)
            {
                return Mouse.current.delta.ReadValue() * MouseDeltaScale;
            }
            return Vector2.zero;
#endif
#if !ENABLE_INPUT_SYSTEM
            return new Vector2(Input.GetAxis("Mouse X"), Input.GetAxis("Mouse Y"));
#endif
        }

        public static float GetMouseScrollY()
        {
#if ENABLE_INPUT_SYSTEM
            if (Mouse.current != null)
            {
                return Mouse.current.scroll.ReadValue().y * ScrollScale;
            }
            return 0.0f;
#endif
#if !ENABLE_INPUT_SYSTEM
            return Input.mouseScrollDelta.y;
#endif
        }

#if ENABLE_INPUT_SYSTEM
        private static bool TryMapKey(KeyCode keyCode, out Key key)
        {
            switch (keyCode)
            {
                case KeyCode.Tab:
                    key = Key.Tab;
                    return true;
                case KeyCode.F:
                    key = Key.F;
                    return true;
                case KeyCode.A:
                    key = Key.A;
                    return true;
                case KeyCode.D:
                    key = Key.D;
                    return true;
                case KeyCode.W:
                    key = Key.W;
                    return true;
                case KeyCode.S:
                    key = Key.S;
                    return true;
                case KeyCode.E:
                    key = Key.E;
                    return true;
                case KeyCode.Q:
                    key = Key.Q;
                    return true;
                case KeyCode.LeftShift:
                    key = Key.LeftShift;
                    return true;
                case KeyCode.RightShift:
                    key = Key.RightShift;
                    return true;
                case KeyCode.Space:
                    key = Key.Space;
                    return true;
                case KeyCode.LeftArrow:
                    key = Key.LeftArrow;
                    return true;
                case KeyCode.RightArrow:
                    key = Key.RightArrow;
                    return true;
                case KeyCode.UpArrow:
                    key = Key.UpArrow;
                    return true;
                case KeyCode.DownArrow:
                    key = Key.DownArrow;
                    return true;
                case KeyCode.R:
                    key = Key.R;
                    return true;
                case KeyCode.Escape:
                    key = Key.Escape;
                    return true;
                default:
                    key = Key.None;
                    return false;
            }
        }
#endif
    }
}
