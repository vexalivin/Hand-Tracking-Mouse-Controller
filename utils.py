import cv2
import json
import numpy as np
import os
import subprocess
import urllib.request

from config import MODEL_PATH, MODEL_URL


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


def _gray_frame(w, h, text):
    gray = np.full((h, w, 3), (60, 60, 60), dtype=np.uint8)
    cv2.putText(gray, text, (w // 2 - 80, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    return gray


def list_available_cameras():
    available = []
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
    print("Downloading hand landmarker model (~15 MB) \u2026")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")
