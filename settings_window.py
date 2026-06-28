import keyboard
import queue
import threading
import tkinter as tk
from tkinter import ttk

import sv_ttk

from config import ACTION_OPTIONS, LANDMARK_IDS, save_config
from shared import _recording_active
from utils import list_available_cameras


SCAN_TO_EN = {}

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
                        r.pop("threshold", None)
                    else:
                        if r["landmark_a"] == new_a:
                            r["threshold_a"] = new_ta
                        if r["landmark_b"] == new_b:
                            r["threshold_b"] = new_tb
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
