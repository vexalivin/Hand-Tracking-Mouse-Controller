import json
import os


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

ACTION_OPTIONS = ["Left Click", "Right Click",
                  "Middle Click", "Active Drag", "Custom Key..."]
LANDMARK_IDS = [str(i) for i in range(21)]


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
