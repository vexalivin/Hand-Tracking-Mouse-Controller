# Hand Tracking Mouse Controller

Controls your PC mouse using hand gestures via webcam. Built with OpenCV, MediaPipe Tasks API, and PyAutoGUI.

## Requirements

- Python 3.8+
- Webcam

## Install

```bash
pip install -r requirements.txt
```

Run the script:

```bash
py hand_tracker.py
```

> On first run, the MediaPipe hand model (~15 MB) is downloaded automatically.

## Gesture Layout

| Gesture | Fingers | Action |
|---|---|---|
| Cursor movement | Middle MCP (landmark 9) | Smooth mouse tracking |
| Left Click | Thumb (4) + Index (8) | `pyautogui.click()` |
| Right Click | Thumb (4) + Middle (12) | `pyautogui.click(button='right')` |
| Hold / Drag | Thumb (4) + Pinky (20) | Toggle `mouseDown` / `mouseUp` |

## Features

- **EMA smoothing** — eliminates jitter from camera noise
- **Deadzone** — cursor locks solid when hand is still
- **Velocity extrapolation** — cursor keeps moving for up to 7 frames if the hand is briefly lost
- **Depth validation** — rejects false pinch triggers when the hand is turned sideways (doesnt work sometimes)
- **Dynamic bounding box** — central 70% of the frame maps to the full screen
- **Visual feedback** — skeleton overlay, colored trigger-point markers, gesture status text

## Controls

- Close the window to quit.

## Files

| File | Purpose |
|---|---|
| `hand_tracker.py` | Main script |
| `hand_landmarker.task` | MediaPipe model (auto-downloaded) |
| `requirements.txt` | Python dependencies |
