# Hand Tracking Mouse Controller

Controls your PC mouse using hand gestures via webcam. Built with OpenCV,
MediaPipe Tasks API, and ctypes `SendInput` for mouse/keyboard injection.

## Requirements

- Python 3.8+
- Webcam

## Install

```bash
pip install -r requirements.txt
```

Run the script:

```bash
py main.py
```

> On first run, the MediaPipe hand model (~15 MB) is downloaded automatically.

## Features

- **Cursor tracking** — Move your hand within the virtual bounding box (central
  70% of frame) to control the mouse cursor. Configurable per hand and landmark.
- **Gesture rules engine** — Create, edit, and delete custom gesture rules via
  the always-on-top settings window.
  - Each rule binds a **hand** (L/R), a **landmark pair** (0–20), two
    **distance thresholds**, and an **action** (Left Click, Right Click, Middle
    Click, Active Drag, Custom Key...).
  - Trigger-point coloured dots show which landmarks are monitored, with
    individual threshold circles (overlap = trigger).
  - Per-landmark threshold sync — rules sharing the same landmark automatically
    stay in sync when you adjust a threshold.
- **Click-hold with repeat delay** — The first action (click / key press)
  triggers **instantly**. Hold past the configurable delay (in ms) to
  start dragging (mouse) or auto-repeating (keyboard). Adjustable live from
  the settings window.
- **Custom key binding** — Record any key combination (e.g. Ctrl+Shift+A) or
  mouse button (`{left}`, `{right}`, `{middle}`) via the Record button in the
  edit dialog. Recording uses system-wide global hooks and auto-stops when all
  keys are released, saving the peak combination.
- **EMA smoothing** — Eliminates jitter from camera noise.
- **Deadzone** — Cursor locks solid when hand is still.
- **Velocity extrapolation** — Cursor keeps moving for up to 7 frames if the
  hand is briefly lost.
- **Depth validation** — Rejects false triggers when the hand is turned
  sideways.
- **Skeleton overlay** — Hand landmark skeleton with index numbers and
  handedness label near the wrist.
- **Live camera switching** — Dropdown lists all available cameras (enumeration
  via PowerShell WMI, with native resolution shown). Switch mid-stream without
  restarting.
- **Mirror display** — Frame is mirrored so your movements feel natural.

## Controls

- Close the camera window to stop tracking.

## How Gesture Rules Work

A rule fires when the **distance** between the two chosen landmarks drops below
the sum of **threshold A + threshold B**. Each landmark has its own threshold
circle radius; when the circles overlap, the gesture triggers.

| Action | Behaviour |
|---|---|
| Left Click / Right Click / Active Drag | Instant click on trigger; hold past delay → drag |
| Middle Click | One-shot on trigger |
| Custom Key... (keyboard) | Instant key press on trigger; hold past delay → auto-repeat |
| Custom Key... `{left}`/`{right}` | Instant mouse-down on trigger; hold past delay → drag |
| Custom Key... `{middle}` | One-shot middle click on trigger |

## Files

| File | Purpose |
|---|---|
| `main.py` | Main script |
| `hand_landmarker.task` | MediaPipe model (auto-downloaded) |
| `gesture_rules.json` | User-defined rules (auto-created, gitignored) |
| `requirements.txt` | Python dependencies |

## Platform

Windows only. The program uses `ctypes.windll.user32.SendInput` for mouse
injection and the `keyboard` library's Windows scan-code layer. macOS and
Linux are not supported.
