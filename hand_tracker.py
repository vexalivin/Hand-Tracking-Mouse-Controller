import os
import urllib.request

import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)
from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec, draw_landmarks

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# MIDDLE_FINGER_MCP (tracked for cursor)
LANDMARK_INDEX = 9
INDEX_TIP = 8                            # landmark index for index finger tip
BOX_FRACTION = 0.70                      # central region mapped to full screen
# weight of new position (lower = smoother / more lag)
SMOOTH_FACTOR = 0.15
THUMB_TIP = 4                            # landmark index for thumb tip
MIDDLE_TIP = 12                          # landmark index for middle finger tip
RING_TIP = 16                            # landmark index for ring finger tip
PINKY_TIP = 20                           # landmark index for pinky finger tip
CLICK_THRESHOLD = 45                    # pixel distance to trigger a click
# max z-diff for valid pinch (prevents side-view false triggers)
DEPTH_TOLERANCE = 0.05
# min pixel distance to actually move cursor
MOVEMENT_DEADZONE = 4
GRACE_LIMIT = 7                          # frames to extrapolate after hand loss

# Turn off PyAutoGUI's built-in pause & enable failsafe
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True


def ensure_model():
    if os.path.isfile(MODEL_PATH):
        return
    print("Downloading hand landmarker model (~15 MB) …")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def main():
    ensure_model()

    screen_w, screen_h = pyautogui.size()      # full-screen resolution

    # ---------------------------
    # 1. Webcam setup
    # ---------------------------
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ---------------------------
    # 2. MediaPipe HandLandmarker
    # ---------------------------
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_tracking_confidence=0.3,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # Smoothed cursor position (start at screen centre)
    smooth_x = screen_w / 2.0
    smooth_y = screen_h / 2.0

    # Gesture state — flag per gesture to prevent spam
    left_clicked = False
    right_clicked = False
    hold_triggered = False
    is_holding = False

    # Last cursor position applied (for deadzone check)
    prev_cursor_x = int(screen_w / 2.0)
    prev_cursor_y = int(screen_h / 2.0)

    # Velocity extrapolation state
    last_target_x = screen_w / 2.0
    last_target_y = screen_h / 2.0
    vel_x = 0.0
    vel_y = 0.0
    grace_counter = 0

    # Timestamp counter for VIDEO mode
    frame_timestamp = 0

    # Frame dimensions (640 x 480 from cap.set above)
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---------------------------
    # 3. Main loop
    # ---------------------------
    cv2.namedWindow("Hand Tracker", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Hand Tracker", cv2.WND_PROP_TOPMOST, 1)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame — exiting.")
                break

            # Mirror so the user sees a natural selfie view
            frame = cv2.flip(frame, 1)

            frame_timestamp += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, frame_timestamp)

            # ---------- Fixed bounding box ----------
            margin = (1.0 - BOX_FRACTION) / 2.0
            b_lft = margin
            b_rgt = 1.0 - margin
            b_top = margin
            b_bot = 1.0 - margin
            bwn = b_rgt - b_lft
            bhn = b_bot - b_top

            hand_detected = False
            raw_target_x = raw_target_y = None
            tracked_lm = None
            box_color = (255, 255, 255)

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    tracked_lm = hand_landmarks[LANDMARK_INDEX]
                    thumb = hand_landmarks[THUMB_TIP]
                    index = hand_landmarks[INDEX_TIP]
                    middle = hand_landmarks[MIDDLE_TIP]
                    ring = hand_landmarks[RING_TIP]
                    pinky = hand_landmarks[PINKY_TIP]

                    d_idx = ((thumb.x - index.x) * cam_w) ** 2 + \
                        ((thumb.y - index.y) * cam_h) ** 2
                    d_idx **= 0.5
                    d_mid = ((thumb.x - middle.x) * cam_w) ** 2 + \
                        ((thumb.y - middle.y) * cam_h) ** 2
                    d_mid **= 0.5
                    d_pinky = ((thumb.x - pinky.x) * cam_w) ** 2 + \
                        ((thumb.y - pinky.y) * cam_h) ** 2
                    d_pinky **= 0.5

                    z_ok_idx = abs(thumb.z - index.z) < DEPTH_TOLERANCE
                    z_ok_mid = abs(thumb.z - middle.z) < DEPTH_TOLERANCE
                    z_ok_pinky = abs(thumb.z - pinky.z) < DEPTH_TOLERANCE

                    if d_idx < CLICK_THRESHOLD and z_ok_idx:
                        if not left_clicked:
                            pyautogui.click()
                            left_clicked = True
                    else:
                        left_clicked = False

                    if not left_clicked and d_mid < CLICK_THRESHOLD and z_ok_mid:
                        if not right_clicked:
                            pyautogui.click(button='right')
                            right_clicked = True
                    else:
                        right_clicked = False

                    if not left_clicked and not right_clicked and d_pinky < CLICK_THRESHOLD and z_ok_pinky:
                        if not hold_triggered:
                            hold_triggered = True
                            if is_holding:
                                pyautogui.mouseUp()
                                is_holding = False
                            else:
                                pyautogui.mouseDown()
                                is_holding = True
                    else:
                        hold_triggered = False

                    gesture_text = "TRACKING"
                    if is_holding:
                        lm_color = (0, 0, 255)
                        conn_color = (0, 0, 255)
                        track_color = (0, 255, 0)
                        overlay_clr = (0, 255, 0)
                        box_color = (0, 165, 255)
                        gesture_text = "HOLDING..."
                    elif left_clicked:
                        lm_color = (0, 0, 255)
                        conn_color = (0, 0, 255)
                        track_color = (0, 255, 0)
                        overlay_clr = (0, 255, 0)
                        box_color = (0, 255, 0)
                        gesture_text = "LEFT CLICK"
                    elif right_clicked:
                        lm_color = (0, 0, 255)
                        conn_color = (0, 0, 255)
                        track_color = (0, 255, 0)
                        overlay_clr = (0, 255, 0)
                        box_color = (255, 0, 0)
                        gesture_text = "RIGHT CLICK"
                    else:
                        lm_color = (0, 255, 0)
                        conn_color = (0, 0, 255)
                        track_color = (255, 255, 0)
                        overlay_clr = (255, 255, 0)
                        box_color = (255, 255, 255)

                    draw_landmarks(frame, hand_landmarks, HandLandmarksConnections.HAND_CONNECTIONS,
                                   DrawingSpec(color=lm_color,
                                               thickness=2, circle_radius=2),
                                   DrawingSpec(color=conn_color, thickness=2))

                    for idx, lm in enumerate(hand_landmarks):
                        lx = int(lm.x * cam_w) + 6
                        ly = int(lm.y * cam_h) + 4
                        cv2.putText(frame, str(idx), (lx, ly),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                    for lm, color in [(thumb, (0, 0, 255)), (index, (0, 255, 0)),
                                      (middle, (0, 255, 0)), (pinky, (0, 255, 0))]:
                        cx = int(lm.x * cam_w)
                        cy = int(lm.y * cam_h)
                        cv2.circle(frame, (cx, cy), 6, color, -1)

                    nx = max(b_lft, min(tracked_lm.x, b_rgt))
                    ny = max(b_top, min(tracked_lm.y, b_bot))
                    raw_target_x = ((nx - b_lft) / bwn) * screen_w
                    raw_target_y = ((ny - b_top) / bhn) * screen_h
                    raw_target_x = max(0.0, min(raw_target_x, screen_w - 1))
                    raw_target_y = max(0.0, min(raw_target_y, screen_h - 1))

                    hand_detected = True
                    break

            cursor_valid = True
            if hand_detected:
                raw_vel_x = raw_target_x - last_target_x
                raw_vel_y = raw_target_y - last_target_y
                vel_x = vel_x * 0.7 + raw_vel_x * 0.3
                vel_y = vel_y * 0.7 + raw_vel_y * 0.3
                last_target_x = raw_target_x
                last_target_y = raw_target_y
                grace_counter = 0
                target_x, target_y = raw_target_x, raw_target_y
            elif grace_counter < GRACE_LIMIT:
                grace_counter += 1
                target_x = last_target_x + vel_x
                target_y = last_target_y + vel_y
                target_x = max(0.0, min(target_x, screen_w - 1))
                target_y = max(0.0, min(target_y, screen_h - 1))
                vel_x *= 0.92
                vel_y *= 0.92
            else:
                cursor_valid = False

            if cursor_valid:
                smooth_x = smooth_x * (1.0 - SMOOTH_FACTOR) + \
                    target_x * SMOOTH_FACTOR
                smooth_y = smooth_y * (1.0 - SMOOTH_FACTOR) + \
                    target_y * SMOOTH_FACTOR
                cursor_x = int(smooth_x)
                cursor_y = int(smooth_y)
                if hand_detected:
                    dz_dx = cursor_x - prev_cursor_x
                    dz_dy = cursor_y - prev_cursor_y
                    if (dz_dx * dz_dx + dz_dy * dz_dy) ** 0.5 >= MOVEMENT_DEADZONE:
                        pyautogui.moveTo(cursor_x, cursor_y, _pause=False)
                        prev_cursor_x, prev_cursor_y = cursor_x, cursor_y
                else:
                    pyautogui.moveTo(cursor_x, cursor_y, _pause=False)
                    prev_cursor_x, prev_cursor_y = cursor_x, cursor_y

            if hand_detected and tracked_lm is not None:
                p_cam_x = int(tracked_lm.x * cam_w)
                p_cam_y = int(tracked_lm.y * cam_h)
                cv2.circle(frame, (p_cam_x, p_cam_y), 10, track_color, -1)
                cv2.putText(frame, gesture_text, (p_cam_x + 12, p_cam_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, overlay_clr, 2)

            x1 = int(b_lft * cam_w)
            x2 = int(b_rgt * cam_w)
            y1 = int(b_top * cam_h)
            y2 = int(b_bot * cam_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            cv2.imshow("Hand Tracker", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q") or cv2.getWindowProperty("Hand Tracker", cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
