import cv2
import numpy as np
import threading
import time

from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec, draw_landmarks
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)
from mediapipe.tasks.python import BaseOptions
import mediapipe as mp
import keyboard

from mouse import (
    send_left_down, send_left_up,
    send_right_down, send_right_up,
    send_left_click, send_right_click,
    send_middle_click, move_mouse_to,
    _virtual_screen_size,
)
from config import (
    MODEL_PATH, BOX_FRACTION, SMOOTH_FACTOR,
    DEPTH_TOLERANCE, MOVEMENT_DEADZONE, GRACE_LIMIT,
)
from utils import ensure_model, list_available_cameras, rounded_rect, _gray_frame
from shared import _recording_active


def camera_thread(config, config_lock, feedback_queue):
    ensure_model()

    def _open_cam(idx):
        c = cv2.VideoCapture(max(0, idx))
        if c.isOpened():
            return c
        c.release()
        return None

    def _cam_index():
        with config_lock:
            return config.get("camera_index", 0)

    cap = _open_cam(_cam_index())
    current_cam_idx = _cam_index()
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap else 640
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap else 480

    cv2.namedWindow("Hand Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Tracker", cam_w, cam_h)
    cv2.setWindowProperty("Hand Tracker", cv2.WND_PROP_TOPMOST, 1)

    settings_btn_rect = {"x": 10, "y": 10, "w": 100, "h": 30}
    _clicked = [False]

    def _mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            r = settings_btn_rect
            if r["x"] <= x <= r["x"] + r["w"] and r["y"] <= y <= r["y"] + r["h"]:
                _clicked[0] = True

    cv2.setMouseCallback("Hand Tracker", _mouse_cb)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_tracking_confidence=0.3,
    )
    landmarker = HandLandmarker.create_from_options(options)

    screen_w, screen_h = _virtual_screen_size()
    if cap:
        cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    smooth_x = screen_w / 2.0
    smooth_y = screen_h / 2.0
    prev_cursor_x = int(screen_w / 2.0)
    prev_cursor_y = int(screen_h / 2.0)
    last_target_x = screen_w / 2.0
    last_target_y = screen_h / 2.0
    vel_x = 0.0
    vel_y = 0.0
    grace_counter = 0
    frame_timestamp = 0

    drag_state = {}
    one_shot_guard = {}
    held_keys = {}

    def get_rules():
        with config_lock:
            return list(config.get("rules", []))

    def is_shutdown():
        with config_lock:
            return config.get("shutdown", False)

    _scan_lock = threading.Lock()
    def _scan_cameras_async():
        if not _scan_lock.acquire(blocking=False):
            return
        def _scan():
            try:
                new_list = list_available_cameras()
                feedback_queue.put(("cam_list", new_list))
                with config_lock:
                    cur = config.get("camera_index", 0)
                    cur_valid = any(i == cur for i, _ in new_list)
                    if not cur_valid and new_list:
                        new_idx = new_list[0][0]
                        config["camera_index"] = new_idx
                        feedback_queue.put(("cam_list", new_list))
                if not cur_valid and new_list:
                    feedback_queue.put(("auto_select", new_idx))
            finally:
                _scan_lock.release()
        threading.Thread(target=_scan, daemon=True).start()

    try:
        while not is_shutdown():
            new_idx = _cam_index()
            if new_idx != current_cam_idx:
                if cap:
                    cap.release()
                cap = _open_cam(new_idx)
                current_cam_idx = new_idx
                if cap:
                    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cv2.resizeWindow("Hand Tracker", cam_w, cam_h)

            if cap is None:
                frame = _gray_frame(cam_w, cam_h, "No camera")
                has_frame = False
            else:
                success, frame = cap.read()
                if not success:
                    cap.release()
                    cap = None
                    frame = _gray_frame(cam_w, cam_h, "No camera")
                    has_frame = False
                    _scan_cameras_async()
                else:
                    has_frame = True

            if has_frame:
                frame = cv2.flip(frame, 1)
                frame_timestamp += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, frame_timestamp)

            margin = (1.0 - BOX_FRACTION) / 2.0
            b_lft = margin
            b_rgt = 1.0 - margin
            b_top = margin
            b_bot = 1.0 - margin
            bwn = b_rgt - b_lft
            bhn = b_bot - b_top

            hand_detected = False
            any_hand_detected = False
            raw_target_x = raw_target_y = None
            tracked_lm = None
            tracking_idx = None
            box_color = (255, 255, 255)
            overlay_clr = (255, 255, 0)
            gesture_text = "NO HAND"
            active_gesture = None

            with config_lock:
                track_cfg = config.get(
                    "tracking", {"enabled": True, "hand": "right", "landmark": 9})
                tracking_enabled = track_cfg.get("enabled", True)
                tracking_hand = track_cfg.get("hand", "right")
                tracking_idx = track_cfg.get(
                    "landmark", 9) if tracking_enabled else None
                click_hold_s = max(
                    0.001, track_cfg.get("click_hold_ms", 267) / 1000.0)

            if has_frame and result.hand_landmarks and result.handedness:
                any_hand_detected = True
                for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                    if hand_idx >= len(result.handedness) or not result.handedness[hand_idx]:
                        continue
                    raw_handedness = result.handedness[hand_idx][0].category_name.lower()
                    handedness = "left" if raw_handedness == "right" else "right"

                    if tracking_idx is not None and tracking_idx < len(hand_landmarks) and handedness == tracking_hand:
                        tracked_lm = hand_landmarks[tracking_idx]
                        hand_detected = True

                    rules = [r for r in get_rules() if r.get(
                        "hand", "right") == handedness]

                    for rule in rules:
                        rid = rule["id"]
                        lm_a = hand_landmarks[rule["landmark_a"]]
                        lm_b = hand_landmarks[rule["landmark_b"]]

                        dx = (lm_a.x - lm_b.x) * cam_w
                        dy = (lm_a.y - lm_b.y) * cam_h
                        dist = (dx * dx + dy * dy) ** 0.5
                        z_ok = abs(lm_a.z - lm_b.z) < DEPTH_TOLERANCE
                        ta = rule.get("threshold_a", rule.get("threshold", 45))
                        tb = rule.get("threshold_b", rule.get("threshold", 45))
                        active = dist < ta + tb and z_ok

                        action = rule["action"]

                        if _recording_active.is_set():
                            continue

                        if action == "Active Drag":
                            if rid not in drag_state:
                                drag_state[rid] = {
                                    "state": "idle", "start_time": 0}
                            ds = drag_state[rid]
                            _down, _up = send_left_down, send_left_up

                            if active:
                                if ds["state"] == "idle":
                                    _down()
                                    ds["state"] = "pending"
                                    ds["start_time"] = time.monotonic()
                                elif ds["state"] == "pending":
                                    if time.monotonic() - ds["start_time"] >= click_hold_s:
                                        ds["state"] = "dragging"
                            else:
                                if ds["state"] in ("pending", "dragging"):
                                    _up()
                                ds["state"] = "idle"

                            if ds["state"] != "idle":
                                active_gesture = f"LM {rule['landmark_a']}+{rule['landmark_b']}"

                        elif action in ("Left Click", "Right Click"):
                            if rid not in one_shot_guard:
                                one_shot_guard[rid] = False
                            if active and not one_shot_guard[rid]:
                                (send_left_click if action == "Left Click" else send_right_click)()
                                one_shot_guard[rid] = True
                                active_gesture = f"LM {rule['landmark_a']}+{rule['landmark_b']}"
                            elif not active:
                                one_shot_guard[rid] = False

                        elif action == "Middle Click":
                            if rid not in one_shot_guard:
                                one_shot_guard[rid] = False
                            if active and not one_shot_guard[rid]:
                                send_middle_click()
                                one_shot_guard[rid] = True
                                active_gesture = f"LM {rule['landmark_a']}+{rule['landmark_b']}"
                            elif not active:
                                one_shot_guard[rid] = False

                        elif action == "Custom Key...":
                            k = rule.get("custom_key", "")
                            if not k:
                                continue

                            if k.startswith("{") and k.endswith("}"):
                                btn = k[1:-1]
                                if btn == "middle":
                                    if rid not in one_shot_guard:
                                        one_shot_guard[rid] = False
                                    if active and not one_shot_guard[rid]:
                                        send_middle_click()
                                        one_shot_guard[rid] = True
                                        active_gesture = f"LM {rule['landmark_a']}+{rule['landmark_b']}"
                                    elif not active:
                                        one_shot_guard[rid] = False
                                else:
                                    if rid not in drag_state:
                                        drag_state[rid] = {
                                            "state": "idle", "start_time": 0}
                                    ds = drag_state[rid]
                                    if btn == "left":
                                        _down, _up = send_left_down, send_left_up
                                    else:
                                        _down, _up = send_right_down, send_right_up
                                    if active:
                                        if ds["state"] == "idle":
                                            _down()
                                            ds["state"] = "pending"
                                            ds["start_time"] = time.monotonic()
                                        elif ds["state"] == "pending":
                                            if time.monotonic() - ds["start_time"] >= click_hold_s:
                                                ds["state"] = "dragging"
                                    else:
                                        if ds["state"] in ("pending", "dragging"):
                                            _up()
                                        ds["state"] = "idle"
                                    if ds["state"] != "idle":
                                        active_gesture = f"LM {rule['landmark_a']}+{rule['landmark_b']}"
                                continue

                            parts = [p.strip() for p in k.lower().split("+")]

                            if rid not in held_keys:
                                held_keys[rid] = {"state": "idle", "start_time": 0,
                                                  "keys": [], "repeat_next": 0}
                            hk = held_keys[rid]

                            if active:
                                if hk["state"] == "idle":
                                    for p in parts:
                                        keyboard.press(p)
                                    hk["state"] = "pending"
                                    hk["start_time"] = time.monotonic()
                                    hk["keys"] = parts
                                    hk["repeat_next"] = 0
                                    active_gesture = f"LM {rule['landmark_a']}+{rule['landmark_b']}"
                                elif hk["state"] == "pending":
                                    if time.monotonic() - hk["start_time"] >= click_hold_s:
                                        hk["state"] = "active"
                                        hk["repeat_next"] = time.monotonic() + 0.5
                                        active_gesture = f"LM {rule['landmark_a']}+{rule['landmark_b']}"
                                elif hk["state"] == "active":
                                    if len(parts) == 1 and time.monotonic() >= hk["repeat_next"]:
                                        keyboard.release(parts[0])
                                        keyboard.press(parts[0])
                                        hk["repeat_next"] = time.monotonic() + 0.033
                                    active_gesture = f"LM {rule['landmark_a']}+{rule['landmark_b']}"
                            else:
                                if hk["state"] in ("pending", "active"):
                                    for p in reversed(hk["keys"]):
                                        keyboard.release(p)
                                hk["state"] = "idle"
                                hk["keys"] = []

                    dragging = any(
                        ds["state"] in ("pending", "dragging")
                        for ds in drag_state.values()
                    )
                    one_shot_fired = any(one_shot_guard.values())

                    if dragging:
                        lm_color = (0, 0, 255)
                        conn_color = (0, 0, 255)
                        track_color = (0, 255, 0)
                        overlay_clr = (0, 255, 0)
                        box_color = (0, 165, 255)
                        gesture_text = "DRAGGING"
                    elif one_shot_fired:
                        lm_color = (0, 0, 255)
                        conn_color = (0, 0, 255)
                        track_color = (0, 255, 0)
                        overlay_clr = (0, 255, 0)
                        box_color = (0, 255, 0)
                        gesture_text = "TRIGGER"
                    else:
                        lm_color = (0, 255, 0)
                        conn_color = (0, 0, 255)
                        track_color = (255, 255, 0)
                        overlay_clr = (255, 255, 0)
                        box_color = (255, 255, 255)
                        gesture_text = "TRACKING"

                    draw_landmarks(frame, hand_landmarks, HandLandmarksConnections.HAND_CONNECTIONS,
                                   DrawingSpec(color=lm_color,
                                               thickness=2, circle_radius=2),
                                   DrawingSpec(color=conn_color, thickness=2))

                    wrist = hand_landmarks[0]
                    wx = int(wrist.x * cam_w)
                    wy = int(wrist.y * cam_h)
                    clr = (0, 255, 0) if handedness == "right" else (255, 0, 0)
                    cv2.putText(frame, handedness.upper(), (wx - 10, wy + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)

                    for idx, lm in enumerate(hand_landmarks):
                        lx = int(lm.x * cam_w) + 6
                        ly = int(lm.y * cam_h) + 4
                        cv2.putText(frame, str(idx), (lx, ly),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                    drawn_lms = set()
                    for rule in rules:
                        for key in ("landmark_a", "landmark_b"):
                            lid = rule[key]
                            if lid in drawn_lms:
                                continue
                            drawn_lms.add(lid)
                            lm = hand_landmarks[lid]
                            cx = int(lm.x * cam_w)
                            cy = int(lm.y * cam_h)
                            tkey = "threshold_a" if key == "landmark_a" else "threshold_b"
                            tr = rule.get(tkey, rule.get("threshold", 45))
                            cv2.circle(frame, (cx, cy), tr,
                                       (0, 255, 255), 1, cv2.LINE_AA)
                            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

                    if hand_detected and tracked_lm is not None and handedness == tracking_hand:
                        nx = max(b_lft, min(tracked_lm.x, b_rgt))
                        ny = max(b_top, min(tracked_lm.y, b_bot))
                        raw_target_x = ((nx - b_lft) / bwn) * screen_w
                        raw_target_y = ((ny - b_top) / bhn) * screen_h
                        raw_target_x = max(
                            0.0, min(raw_target_x, screen_w - 1))
                        raw_target_y = max(
                            0.0, min(raw_target_y, screen_h - 1))

            feedback_queue.put(active_gesture)

            if not any_hand_detected:
                for ds in drag_state.values():
                    if ds["state"] in ("pending", "dragging"):
                        send_left_up()
                        send_right_up()
                    ds["state"] = "idle"
                for hk in held_keys.values():
                    if hk.get("state") in ("pending", "active") and hk.get("keys"):
                        for p in reversed(hk["keys"]):
                            keyboard.release(p)
                    hk["state"] = "idle"
                    hk["keys"] = []

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
                    if (dz_dx ** 2 + dz_dy ** 2) ** 0.5 >= MOVEMENT_DEADZONE:
                        move_mouse_to(cursor_x, cursor_y)
                        prev_cursor_x, prev_cursor_y = cursor_x, cursor_y
                else:
                    move_mouse_to(cursor_x, cursor_y)
                    prev_cursor_x, prev_cursor_y = cursor_x, cursor_y

            if hand_detected and tracked_lm is not None:
                p_cam_x = int(tracked_lm.x * cam_w)
                p_cam_y = int(tracked_lm.y * cam_h)
                cv2.circle(frame, (p_cam_x, p_cam_y), 10, track_color, -1)

            cv2.putText(frame, gesture_text, (10, cam_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay_clr, 2)

            x1 = int(b_lft * cam_w)
            x2 = int(b_rgt * cam_w)
            y1 = int(b_top * cam_h)
            y2 = int(b_bot * cam_h)
            rounded_rect(frame, (x1, y1), (x2, y2), box_color, 2, r=12)

            r = settings_btn_rect
            for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                cv2.putText(frame, "Settings", (r["x"] + 8 + dx, r["y"] + 21 + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(frame, "Settings", (r["x"] + 8, r["y"] + 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            if _clicked[0]:
                _clicked[0] = False
                feedback_queue.put(("toggle_settings",))

            cv2.imshow("Hand Tracker", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q") or cv2.getWindowProperty("Hand Tracker", cv2.WND_PROP_VISIBLE) < 1:
                with config_lock:
                    config["shutdown"] = True
                break

    finally:
        for hk in held_keys.values():
            if hk.get("state") in ("pending", "active") and hk.get("keys"):
                for p in reversed(hk["keys"]):
                    keyboard.release(p)
        send_left_up()
        send_right_up()
        feedback_queue.put(None)
        landmarker.close()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
