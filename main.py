import sys

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

import queue
import threading
import tkinter as tk

from config import default_config, load_config, save_config
from engine import camera_thread
from settings_window import SettingsWindow
from utils import list_available_cameras


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
