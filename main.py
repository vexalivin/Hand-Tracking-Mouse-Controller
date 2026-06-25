import keyboard
import numpy as np
from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec, draw_landmarks
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)
from mediapipe.tasks.python import BaseOptions
import sv_ttk
import pyautogui
import mediapipe as mp
import cv2
import ctypes
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk
import urllib.request

SCAN_TO_EN = {}

# Import error handling
_missing = []
for mod, pip_name in [("cv2", "opencv-python"), ("mediapipe", "mediapipe"),
                      ("pyautogui", "pyautogui"), ("sv_ttk", "sv-ttk"),
                      ("keyboard", "keyboard")]:
    try:
        __import__(mod)
    except ImportError:
        _missing.append(pip_name)
if _missing:
    print("Missing required packages:")
    for m in _missing:
        print(f"  pip install {m}")
    print("Or install everything at once:")
    print("  pip install -r requirements.txt")
    input("\nPress Enter to exit...")
    sys.exit(1)


# English key name mapping
keyboard._os_keyboard.init()
_sc2vk = keyboard._os_keyboard.scan_code_to_vk
_vks = keyboard._os_keyboard.official_virtual_keys
for _sc, _vk in _sc2vk.items():
    if _vk in _vks:
        _n = _vks[_vk][0].lower().replace("_", " ")
        if _n in ("left ctrl", "right ctrl"):
            _n = "ctrl"
        elif _n in ("left alt", "right alt", "left menu", "right menu"):
            _n = "alt"
        elif _n in ("left shift", "right shift"):
            _n = "shift"
        elif _n in ("left windows", "right windows"):
            _n = "win"
        elif _n == "spacebar":
            _n = "space"
        SCAN_TO_EN[_sc] = _n


MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
INPUT_MOUSE = 0


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", _MOUSEINPUT),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("data", _INPUT_UNION),
    ]


def _send_mouse(dwFlags):
    inp = _INPUT()
    inp.type = INPUT_MOUSE
    inp.data.mi.dwFlags = dwFlags
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def _send_mouse_abs(x, y, flags):
    sw = ctypes.windll.user32.GetSystemMetrics(0)
    sh = ctypes.windll.user32.GetSystemMetrics(1)
    inp = _INPUT()
    inp.type = INPUT_MOUSE
    inp.data.mi.dx = int(x * 65535 / (sw - 1))
    inp.data.mi.dy = int(y * 65535 / (sh - 1))
    inp.data.mi.dwFlags = flags | MOUSEEVENTF_ABSOLUTE
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def send_left_click():
    _send_mouse(MOUSEEVENTF_LEFTDOWN)
    _send_mouse(MOUSEEVENTF_LEFTUP)


def send_right_click():
    _send_mouse(MOUSEEVENTF_RIGHTDOWN)
    _send_mouse(MOUSEEVENTF_RIGHTUP)


def send_left_down():
    _send_mouse(MOUSEEVENTF_LEFTDOWN)


def send_left_up():
    _send_mouse(MOUSEEVENTF_LEFTUP)


def send_right_down():
    _send_mouse(MOUSEEVENTF_RIGHTDOWN)


def send_right_up():
    _send_mouse(MOUSEEVENTF_RIGHTUP)


def send_middle_click():
    _send_mouse(MOUSEEVENTF_MIDDLEDOWN)
    _send_mouse(MOUSEEVENTF_MIDDLEUP)


def move_mouse_to(x, y):
    _send_mouse_abs(x, y, MOUSEEVENTF_MOVE)


def rounded_rect(img, p1, p2, color, thickness=2, r=10):
    x1, y1 = p1
    x2, y2 = p2
    r = min(r, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2 - r, y2), (x1 + r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 0, 180,
                270, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 0, 270,
                360, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0,
                90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 0, 90,
                180, color, thickness, cv2.LINE_AA)


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
RULES_PATH = os.path.join(os.path.dirname(__file__), "gesture_rules.json")

BOX_FRACTION = 0.70
SMOOTH_FACTOR = 0.15
DEPTH_TOLERANCE = 0.05
MOVEMENT_DEADZONE = 4
GRACE_LIMIT = 7

_recording_active = threading.Event()

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

ACTION_OPTIONS = ["Left Click", "Right Click",
                  "Middle Click", "Custom Key..."]
LANDMARK_IDS = [str(i) for i in range(21)]


def list_available_cameras():
    available = []
    # Suppress OpenCV obsensor errors during probing
    cv2.redirectError(lambda *args: (0, ''))
    try:
        for i in range(10):
            cap = None
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    available.append((i, w, h))
            except Exception:
                pass
            finally:
                if cap:
                    cap.release()
    finally:
        cv2.redirectError(None)

    wmi_names = {}
    if available:
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance -Namespace root/cimv2 -ClassName Win32_PnPEntity | "
                 "Where-Object { $_.PNPClass -eq 'Camera' } | "
                 "Select-Object Name,PNPDeviceID | ConvertTo-Json"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                devices = json.loads(result.stdout)
                if not isinstance(devices, list):
                    devices = [devices]
                for d in devices:
                    name = d.get("Name", "")
                    if name:
                        name = name.replace("  ", " ").strip()
                        wmi_names[len(wmi_names)] = name
        except Exception:
            pass

    result = []
    for i, w, h in available:
        label = wmi_names.get(len(result), f"Camera {i}")
        result.append((i, f"[{i}] {label}"))
    return result


def ensure_model():
    if os.path.isfile(MODEL_PATH):
        return
    print("Downloading hand landmarker model (~15 MB) …")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


# Config persistence

def save_config(data, path=RULES_PATH):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_config(path=RULES_PATH):
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                for r in data:
                    r.setdefault("hand", "right")
                    if "threshold" in r and "threshold_a" not in r:
                        r["threshold_a"] = r["threshold_b"] = r["threshold"]
                return {"tracking": {"enabled": True, "hand": "right", "landmark": 9, "click_hold_ms": 267}, "rules": data}
            for r in data.get("rules", []):
                r.setdefault("hand", "right")
                if "threshold" in r and "threshold_a" not in r:
                    r["threshold_a"] = r["threshold_b"] = r["threshold"]
            data.setdefault("tracking", {}).setdefault("hand", "right")
            data.setdefault("camera_index", 0)
            # backward compat: convert frames to ms
            track = data.get("tracking", {})
            if "click_hold_frames" in track and "click_hold_ms" not in track:
                track["click_hold_ms"] = int(
                    track.pop("click_hold_frames") * 33.33)
            track.setdefault("click_hold_ms", 267)
            return data
    return None


def default_config():
    return {
        "tracking": {"enabled": True, "hand": "right", "landmark": 9, "click_hold_ms": 267},
        "rules": [
            {"id": 0, "hand": "right", "landmark_a": 4, "landmark_b": 8, "threshold_a": 45,
                "threshold_b": 45, "action": "Active Drag", "custom_key": ""},
            {"id": 1, "hand": "right", "landmark_a": 4, "landmark_b": 12, "threshold_a": 45,
                "threshold_b": 45, "action": "Right Click", "custom_key": ""},
        ],
    }


# Settings window

class SettingsWindow:
    def __init__(self, config, config_lock, feedback_queue, cam_list):
        self.config = config
        self.config_lock = config_lock
        self.feedback_queue = feedback_queue
        self.cam_list = cam_list

        self.root = tk.Tk()
        self.root.title("Hand Tracker Gesture Rules")
        self.root.attributes("-topmost", True)
        self.root.minsize(480, 250)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._blank_icon = tk.PhotoImage(width=1, height=1)
        self.root.tk.call("wm", "iconphoto", self.root._w, self._blank_icon)
        sv_ttk.set_theme("dark")

        self._visible = False
        self.root.withdraw()

        self._build_ui()
        self._rebuild_listbox()
        self.root.update_idletasks()
        self.root.geometry("")
        self._poll_feedback()

    def _rule_text(self, r):
        hand = r.get("hand", "right")[0].upper()
        a, b = r["landmark_a"], r["landmark_b"]
        ta = r.get("threshold_a", r.get("threshold", 45))
        tb = r.get("threshold_b", r.get("threshold", 45))
        act = r["action"]
        if act == "Custom Key..." and r.get("custom_key"):
            act = f"Key:{r['custom_key']}"
        return f"{hand} | {a}\u2192{b} | {ta:>2}+{tb:>2} | {act}"

    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=(10, 0))
        ttk.Label(top, text="Gesture Rules", font=(
            "Segoe UI", 11, "bold")).pack(side="left")
        ttk.Button(top, text="+ Add",
                   command=self._add_rule).pack(side="right", padx=(2, 0))
        ttk.Button(top, text="Edit", command=self._edit_selected).pack(
            side="right", padx=(2, 0))
        ttk.Button(top, text="\u2716", width=3,
                   command=self._delete_selected).pack(side="right")

        track_frame = ttk.Frame(self.root)
        track_frame.pack(fill="x", padx=10, pady=(5, 0))
        self.track_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(track_frame, text="Tracking", variable=self.track_enabled_var,
                        command=self._on_tracking_toggle).pack(side="left")
        self.track_hand_var = tk.StringVar(value="right")
        hand_cb = ttk.Combobox(track_frame, textvariable=self.track_hand_var, values=["left", "right"],
                               width=5, state="readonly")
        hand_cb.pack(side="left", padx=(5, 2))
        hand_cb.bind("<<ComboboxSelected>>",
                     lambda e: self._on_tracking_change())

        ttk.Label(track_frame, text="LM").pack(side="left", padx=(2, 2))
        self.track_lm_var = tk.StringVar(value="9")
        lm_cb = ttk.Combobox(track_frame, textvariable=self.track_lm_var,
                             values=LANDMARK_IDS, width=3, state="readonly")
        lm_cb.pack(side="left")
        lm_cb.bind("<<ComboboxSelected>>",
                   lambda e: self._on_tracking_change())

        ttk.Label(track_frame, text="  Hold delay (ms):").pack(
            side="left", padx=(5, 2))
        self.click_hold_var = tk.StringVar(value="267")
        hold_entry = ttk.Entry(track_frame, textvariable=self.click_hold_var,
                               width=5, justify="center")
        hold_entry.pack(side="left")
        hold_entry.bind(
            "<KeyRelease>", lambda e=None: self._on_click_hold_change())

        ttk.Label(track_frame, text="  Camera:").pack(side="left", padx=(5, 2))
        self.cam_var = tk.StringVar()
        cam_values = [lbl for _,
                      lbl in self.cam_list] if self.cam_list else ["0"]
        self.cam_cb = ttk.Combobox(track_frame, textvariable=self.cam_var,
                                   values=cam_values, width=30, state="readonly")
        self.cam_cb.pack(side="left")
        self.cam_cb.bind("<<ComboboxSelected>>", lambda e: self._on_camera_change())
        ttk.Button(track_frame, text="\u21bb", width=3,
                   command=self._scan_cameras).pack(side="left", padx=(2, 0))

        self.listbox = tk.Listbox(self.root, height=8, font=("Consolas", 10))
        self.listbox.pack(fill="both", expand=True, padx=10, pady=(5, 0))
        self.listbox.bind("<Double-Button-1>", lambda e: self._edit_selected())

        sep = ttk.Separator(self.root, orient="horizontal")
        sep.pack(fill="x", padx=10, pady=(5, 5))

        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(status_frame, text="Gesture Status:").pack(side="left")
        self.status_canvas = tk.Canvas(
            status_frame, width=16, height=16, highlightthickness=0)
        self.status_canvas.pack(side="left", padx=(5, 0))
        self.status_dot = self.status_canvas.create_oval(
            2, 2, 14, 14, fill="gray", outline="")
        self.status_label = ttk.Label(status_frame, text="---")
        self.status_label.pack(side="left", padx=(5, 0))

    def _rebuild_listbox(self):
        self.listbox.delete(0, "end")
        with self.config_lock:
            rules = list(self.config.get("rules", []))
            tracking = self.config.get("tracking", {})
            self.track_enabled_var.set(tracking.get("enabled", True))
            self.track_hand_var.set(tracking.get("hand", "right"))
            self.track_lm_var.set(str(tracking.get("landmark", 9)))
            self.click_hold_var.set(str(tracking.get("click_hold_ms", 267)))
            self.cam_var.set("")
            cur_idx = self.config.get("camera_index", 0)
            for i, lbl in self.cam_list:
                if i == cur_idx:
                    self.cam_var.set(lbl)
                    break
            if not self.cam_var.get() and self.cam_list:
                self.cam_var.set(self.cam_list[0][1])
            elif not self.cam_list:
                self.cam_var.set("0")
        for r in rules:
            self.listbox.insert("end", self._rule_text(r))

    def _get_selected_rule(self):
        sel = self.listbox.curselection()
        if not sel:
            return None
        idx = sel[0]
        with self.config_lock:
            rules = self.config.get("rules", [])
            if idx < len(rules):
                return rules[idx]
        return None

    def _on_tracking_toggle(self):
        with self.config_lock:
            self.config.setdefault("tracking", {})[
                "enabled"] = self.track_enabled_var.get()
        self._save_full()

    def _on_tracking_change(self):
        with self.config_lock:
            t = self.config.setdefault("tracking", {})
            t["hand"] = self.track_hand_var.get()
            t["landmark"] = int(self.track_lm_var.get())
        self._save_full()

    def _on_click_hold_change(self):
        raw = self.click_hold_var.get().strip()
        if raw.isdigit():
            val = int(raw)
        else:
            val = 267
            self.click_hold_var.set(str(val))
        with self.config_lock:
            self.config.setdefault("tracking", {})["click_hold_ms"] = val
        self._save_full()

    def _on_camera_change(self):
        label = self.cam_var.get()
        idx = 0
        for i, lbl in (self.cam_list or []):
            if lbl == label:
                idx = i
                break
        with self.config_lock:
            self.config["camera_index"] = idx
        self._save_full()

    def _scan_cameras(self):
        def scan():
            new_list = list_available_cameras()
            self.root.after(0, lambda: self._apply_cam_list(new_list))
        threading.Thread(target=scan, daemon=True).start()

    def _apply_cam_list(self, new_list):
        self.cam_list = new_list
        self._refresh_cam_dropdown()

    def _save_full(self):
        with self.config_lock:
            save_config({"tracking": self.config.get("tracking", {}), "rules": self.config["rules"],
                         "camera_index": self.config.get("camera_index", 0)})

    def _add_rule(self):
        with self.config_lock:
            rules = self.config["rules"]
            new_id = self.config["next_rule_id"]
            self.config["next_rule_id"] = new_id + 1
            rule = {"id": new_id, "hand": "right", "landmark_a": 4, "landmark_b": 8,
                    "threshold_a": 45, "threshold_b": 45, "action": "Left Click", "custom_key": ""}
            rules.append(rule)
        self._save_full()
        self._rebuild_listbox()
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set("end")

    def _delete_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        with self.config_lock:
            rules = self.config["rules"]
            if idx < len(rules):
                del rules[idx]
        self._save_full()
        self._rebuild_listbox()

    def _edit_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        with self.config_lock:
            rules = self.config["rules"]
            if idx >= len(rules):
                return
            rule = rules[idx]

        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Rule")
        dialog.attributes("-topmost", True)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog._icon = tk.PhotoImage(width=1, height=1)
        dialog.tk.call("wm", "iconphoto", dialog._w, dialog._icon)
        sv_ttk.set_theme("dark")

        frame = ttk.Frame(dialog, padding=10)
        frame.pack()

        row = 0
        ttk.Label(frame, text="Hand:").grid(
            row=row, column=0, sticky="e", padx=(0, 5))
        hand_var = tk.StringVar(value=rule.get("hand", "right"))
        ttk.Combobox(frame, textvariable=hand_var, values=["left", "right"],
                     width=8, state="readonly").grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(frame, text="Landmark A:").grid(
            row=row, column=0, sticky="e", padx=(0, 5))
        la_var = tk.StringVar(value=str(rule["landmark_a"]))
        ttk.Combobox(frame, textvariable=la_var, values=LANDMARK_IDS,
                     width=4, state="readonly").grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(frame, text="Landmark B:").grid(
            row=row, column=0, sticky="e", padx=(0, 5))
        lb_var = tk.StringVar(value=str(rule["landmark_b"]))
        ttk.Combobox(frame, textvariable=lb_var, values=LANDMARK_IDS,
                     width=4, state="readonly").grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(frame, text="Threshold A:").grid(
            row=row, column=0, sticky="e", padx=(0, 5))
        ta_var = tk.IntVar(value=rule.get(
            "threshold_a", rule.get("threshold", 45)))
        ttk.Scale(frame, from_=10, to=100, orient="horizontal",
                  variable=ta_var, length=150).grid(row=row, column=1, sticky="ew", padx=(0, 5))
        ta_label = ttk.Label(frame, text=str(ta_var.get()), width=3)
        ta_label.grid(row=row, column=2, sticky="w")
        ta_var.trace_add(
            "write", lambda *a: ta_label.config(text=str(ta_var.get())))
        row += 1

        ttk.Label(frame, text="Threshold B:").grid(
            row=row, column=0, sticky="e", padx=(0, 5))
        tb_var = tk.IntVar(value=rule.get(
            "threshold_b", rule.get("threshold", 45)))
        ttk.Scale(frame, from_=10, to=100, orient="horizontal",
                  variable=tb_var, length=150).grid(row=row, column=1, sticky="ew", padx=(0, 5))
        tb_label = ttk.Label(frame, text=str(tb_var.get()), width=3)
        tb_label.grid(row=row, column=2, sticky="w")
        tb_var.trace_add(
            "write", lambda *a: tb_label.config(text=str(tb_var.get())))
        row += 1

        ttk.Label(frame, text="Action:").grid(
            row=row, column=0, sticky="e", padx=(0, 5))
        action_var = tk.StringVar(value=rule["action"])
        action_combo = ttk.Combobox(frame, textvariable=action_var,
                                    values=ACTION_OPTIONS, width=16, state="readonly")
        action_combo.grid(row=row, column=1, columnspan=2, sticky="w")
        row += 1

        key_frame = ttk.Frame(frame)
        key_frame.grid(row=row, column=0, columnspan=3, pady=(5, 0))
        ttk.Label(key_frame, text="Key:").pack(side="left")

        key_label = ttk.Label(key_frame, text=rule.get("custom_key", "") or "(none)",
                              width=18, anchor="w", font=("Consolas", 10))
        key_label.pack(side="left", padx=(5, 0))

        recording = {"active": False}

        def update_display():
            parts = recording["keys"]
            if parts:
                text = "+".join(parts)
                recording["last_display"] = text
                if len(parts) > len(recording.get("peak_keys", [])):
                    recording["peak_keys"] = parts.copy()
            else:
                text = recording.get("last_display") or "... press keys ..."
            dialog.after(0, lambda t=text: key_label.configure(text=t))

        def start_recording():
            recording["active"] = True
            _recording_active.set()
            record_btn.configure(text="Stop", style="")
            key_label.configure(text="... press keys ...")
            recording["original"] = rule.get("custom_key", "")
            recording["captured"] = None
            recording["keys"] = []
            recording["last_display"] = None
            recording["peak_keys"] = []
            recording["hook"] = keyboard.hook(on_global_key, suppress=True)

        def stop_recording():
            recording["active"] = False
            _recording_active.clear()
            record_btn.configure(text="Record")
            if recording.get("hook"):
                keyboard.unhook(recording["hook"])
                recording["hook"] = None
            recording["keys"].clear()
            if recording["captured"] is None:
                peak = recording.get("peak_keys")
                if peak:
                    recording["captured"] = "+".join(peak)
                else:
                    recording["captured"] = recording.get("last_display")
            if recording["captured"] is not None:
                key_label.configure(text=recording["captured"])
            else:
                key_label.configure(text=recording.get("original") or "(none)")

        def key_name(event):
            return SCAN_TO_EN.get(event.scan_code, event.name.lower().replace("left ", "").replace("right ", ""))

        def on_global_key(event):
            if recording["captured"] is not None:
                return
            simple = key_name(event)

            if event.event_type == "down":
                if simple not in recording["keys"]:
                    recording["keys"].append(simple)
                update_display()
            elif event.event_type == "up":
                if simple in recording["keys"]:
                    recording["keys"].remove(simple)
                if recording["keys"]:
                    update_display()
                else:
                    peak = recording.get("peak_keys")
                    if peak:
                        combo = "+".join(peak)
                        recording["captured"] = combo
                        dialog.after(0, lambda: (
                            key_label.configure(text=combo), stop_recording()))

        record_btn = ttk.Button(key_frame, text="Record", width=8,
                                command=lambda: start_recording() if not recording["active"] else stop_recording())
        record_btn.pack(side="left", padx=(5, 0))

        def show_key(show):
            for w in (key_label, record_btn):
                w.pack_forget()
            if show:
                key_label.pack(side="left", padx=(5, 0))
                record_btn.pack(side="left", padx=(5, 0))

        show_key(action_var.get() == "Custom Key...")
        action_var.trace_add(
            "write", lambda *a: show_key(action_var.get() == "Custom Key..."))

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=(0, 10))

        def save():
            with self.config_lock:
                new_a = int(la_var.get())
                new_b = int(lb_var.get())
                new_ta = ta_var.get()
                new_tb = tb_var.get()
                for r in self.config["rules"]:
                    if r["id"] == rule["id"]:
                        r["hand"] = hand_var.get()
                        r["landmark_a"] = new_a
                        r["landmark_b"] = new_b
                        r["threshold_a"] = new_ta
                        r["threshold_b"] = new_tb
                        r["action"] = action_var.get()
                        r["custom_key"] = recording.get("captured") if recording.get("captured") is not None else (
                            recording.get("original") or r.get("custom_key", ""))
                        r.pop("threshold", None)  # remove old single-threshold
                    else:
                        # Sync threshold_a if another rule shares landmark_a's value
                        if r["landmark_a"] == new_a:
                            r["threshold_a"] = new_ta
                        if r["landmark_b"] == new_b:
                            r["threshold_b"] = new_tb
                        # If sharing the same landmark value in opposite slot, sync that too
                        if r["landmark_a"] == new_b:
                            r["threshold_a"] = new_tb
                        if r["landmark_b"] == new_a:
                            r["threshold_b"] = new_ta
            self._save_full()
            self._rebuild_listbox()
            if recording["active"]:
                stop_recording()
            dialog.destroy()

        ttk.Button(btn_frame, text="Save", command=save).pack(
            side="left", padx=5)

        def cancel():
            if recording["active"]:
                stop_recording()
            dialog.destroy()
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        ttk.Button(btn_frame, text="Cancel", command=cancel).pack(
            side="left", padx=5)

    def _poll_feedback(self):
        with self.config_lock:
            if self.config.get("shutdown"):
                self.root.destroy()
                return
        try:
            while True:
                msg = self.feedback_queue.get_nowait()
                if msg is None:
                    self.status_canvas.itemconfig(self.status_dot, fill="gray")
                    self.status_label.config(text="---")
                elif isinstance(msg, tuple) and msg[0] == "toggle_settings":
                    self._toggle_settings()
                elif isinstance(msg, tuple) and msg[0] == "cam_list":
                    self.cam_list = msg[1]
                    self._refresh_cam_dropdown()
                elif isinstance(msg, tuple) and msg[0] == "auto_select":
                    self._auto_select_camera(msg[1])
                else:
                    self.status_canvas.itemconfig(
                        self.status_dot, fill="#00cc00")
                    self.status_label.config(text=msg)
                    self.root.after(150, lambda: self.status_canvas.itemconfig(
                        self.status_dot, fill="gray"))
        except queue.Empty:
            pass
        self.root.after(50, self._poll_feedback)

    def _refresh_cam_dropdown(self):
        values = [lbl for _, lbl in self.cam_list] if self.cam_list else ["0"]
        self.cam_cb.configure(values=values)
        current = self.cam_var.get()
        if current not in values:
            self.cam_var.set(values[0] if values else "0")

    def _toggle_settings(self):
        if self._visible:
            self.root.withdraw()
            self._visible = False
        else:
            self.root.deiconify()
            self.root.lift()
            self._visible = True

    def on_close(self):
        self.root.withdraw()
        self._visible = False

    def _auto_select_camera(self, idx):
        for i, lbl in (self.cam_list or []):
            if i == idx:
                self.cam_var.set(lbl)
                self._on_camera_change()
                return


# Camera / engine thread

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

    screen_w, screen_h = pyautogui.size()
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

    def _gray_frame(w, h, text):
        gray = np.full((h, w, 3), (60, 60, 60), dtype=np.uint8)
        cv2.putText(gray, text, (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        return gray

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
            raw_target_x = raw_target_y = None
            tracked_lm = None
            tracking_idx = None
            box_color = (255, 255, 255)
            overlay_clr = (255, 255, 0)
            gesture_text = "NO HAND"
            active_gesture = None

            # Read tracking config
            with config_lock:
                track_cfg = config.get(
                    "tracking", {"enabled": True, "hand": "right", "landmark": 9})
                tracking_enabled = track_cfg.get("enabled", True)
                tracking_hand = track_cfg.get("hand", "right")
                tracking_idx = track_cfg.get(
                    "landmark", 9) if tracking_enabled else None
                click_hold = max(
                    1, int(track_cfg.get("click_hold_ms", 267) / 33.33))

            if has_frame and result.hand_landmarks and result.handedness:
                for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                    if hand_idx >= len(result.handedness) or not result.handedness[hand_idx]:
                        continue
                    raw_handedness = result.handedness[hand_idx][0].category_name.lower(
                    )
                    # Frame is mirrored, so swap to match physical hand
                    handedness = "left" if raw_handedness == "right" else "right"

                    # Tracking — use configured hand only
                    if tracking_idx is not None and tracking_idx < len(hand_landmarks) and handedness == tracking_hand:
                        tracked_lm = hand_landmarks[tracking_idx]
                        hand_detected = True

                    # Filter rules for this hand
                    rules = [r for r in get_rules() if r.get(
                        "hand", "right") == handedness]
                    all_trigger_lms = set()
                    for rule in rules:
                        all_trigger_lms.add(rule["landmark_a"])
                        all_trigger_lms.add(rule["landmark_b"])

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

                        if action in ("Active Drag", "Left Click", "Right Click"):
                            if rid not in drag_state:
                                drag_state[rid] = {
                                    "state": "idle", "start_frame": 0}
                            ds = drag_state[rid]

                            if action == "Active Drag":
                                _down, _up, _click = send_left_down, send_left_up, send_left_click
                            elif action == "Left Click":
                                _down, _up, _click = send_left_down, send_left_up, send_left_click
                            else:
                                _down, _up, _click = send_right_down, send_right_up, send_right_click

                            if active:
                                if ds["state"] == "idle":
                                    _down()
                                    ds["state"] = "pending"
                                    ds["start_frame"] = frame_timestamp
                                elif ds["state"] == "pending":
                                    elapsed = frame_timestamp - \
                                        ds["start_frame"]
                                    if elapsed >= click_hold:
                                        ds["state"] = "dragging"
                            else:
                                if ds["state"] == "pending":
                                    _up()
                                elif ds["state"] == "dragging":
                                    _up()
                                ds["state"] = "idle"

                            if ds["state"] != "idle":
                                active_gesture = f"LM {rule['landmark_a']}+{rule['landmark_b']}"

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

                            # Handle mouse-click custom keys
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
                                            "state": "idle", "start_frame": 0}
                                    ds = drag_state[rid]
                                    if btn == "left":
                                        _down, _up, _click = send_left_down, send_left_up, send_left_click
                                    else:
                                        _down, _up, _click = send_right_down, send_right_up, send_right_click
                                    if active:
                                        if ds["state"] == "idle":
                                            _down()
                                            ds["state"] = "pending"
                                            ds["start_frame"] = frame_timestamp
                                        elif ds["state"] == "pending":
                                            if frame_timestamp - ds["start_frame"] >= click_hold:
                                                ds["state"] = "dragging"
                                    else:
                                        if ds["state"] == "pending":
                                            _up()
                                        elif ds["state"] == "dragging":
                                            _up()
                                        ds["state"] = "idle"
                                    if ds["state"] != "idle":
                                        active_gesture = f"LM {rule['landmark_a']}+{rule['landmark_b']}"
                                continue

                            parts = [p.strip() for p in k.lower().split("+")]

                            if rid not in held_keys:
                                held_keys[rid] = {"state": "idle", "start_frame": 0,
                                                  "keys": [], "repeat_next": 0}
                            hk = held_keys[rid]

                            if active:
                                if hk["state"] == "idle":
                                    for p in parts:
                                        keyboard.press(p)
                                    hk["state"] = "pending"
                                    hk["start_frame"] = frame_timestamp
                                    hk["keys"] = parts
                                    hk["repeat_next"] = 0
                                    active_gesture = f"LM {rule['landmark_a']}+{rule['landmark_b']}"
                                elif hk["state"] == "pending":
                                    if frame_timestamp - hk["start_frame"] >= click_hold:
                                        hk["state"] = "active"
                                        hk["repeat_next"] = frame_timestamp + 15
                                        active_gesture = f"LM {rule['landmark_a']}+{rule['landmark_b']}"
                                elif hk["state"] == "active":
                                    if len(parts) == 1 and frame_timestamp >= hk["repeat_next"]:
                                        keyboard.release(parts[0])
                                        keyboard.press(parts[0])
                                        hk["repeat_next"] = frame_timestamp + 1
                                    active_gesture = f"LM {rule['landmark_a']}+{rule['landmark_b']}"
                            else:
                                if hk["state"] in ("pending", "active"):
                                    for p in reversed(hk["keys"]):
                                        keyboard.release(p)
                                hk["state"] = "idle"
                                hk["keys"] = []

                    # Determine visual style
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

                    # Label handedness near the wrist
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

            if not hand_detected:
                for ds in drag_state.values():
                    if ds["state"] == "pending":
                        ds["state"] = "idle"
                    elif ds["state"] == "dragging":
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

            # Settings button
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
        feedback_queue.put(None)
        landmarker.close()
        if cap:
            cap.release()
        cv2.destroyAllWindows()


def main():
    data = load_config()
    if data is None:
        data = default_config()
        save_config(data)

    rules = data.get("rules", default_config()["rules"])
    next_id = max(r["id"] for r in rules) + 1 if rules else 0

    config = {
        "shutdown": False,
        "tracking": data.get("tracking", {"enabled": True, "landmark": 9}),
        "camera_index": data.get("camera_index", 0),
        "rules": rules,
        "next_rule_id": next_id,
    }
    config_lock = threading.Lock()
    feedback_queue = queue.Queue()

    cam_list = list_available_cameras()
    cam_idx = data.get("camera_index", 0)
    if not any(i == cam_idx for i, _ in cam_list):
        cam_idx = cam_list[0][0] if cam_list else 0

    config["camera_index"] = cam_idx

    cam_thread = threading.Thread(
        target=camera_thread, args=(
            config, config_lock, feedback_queue), daemon=True
    )
    cam_thread.start()

    SettingsWindow(config, config_lock, feedback_queue, cam_list)
    tk.mainloop()


if __name__ == "__main__":
    main()
