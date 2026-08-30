#!/usr/bin/env python3
"""
Lane ROI Calibrator — click-to-mark tool for beamng_pilot's lane-detection ROI.

Mark three things on a screenshot of YOUR OWN driving POV:
  1. LEFT lane line   (click a few points along it, near -> far)
  2. RIGHT lane line  (click a few points along it, near -> far)
  3. HOOD / dash line (click a couple points along the top edge of your
                        hood/dash — same idea as the yellow line you drew)

It writes lane_roi_config.json next to this script. beamng_pilot_v16.py
loads that file at startup (see the integration snippet you were given)
and uses it instead of the hardcoded ROI_POLY_*/ROI_TOP_CROP constants —
so the search area matches YOUR camera FOV instead of a generic trapezoid.

With no arguments, this automatically scans for the "BeamNG.tech" /
"BeamNG.drive" window (polls for up to 30s, same as the pilot script) and
grabs its client area — title bar and window border excluded, so the
captured frame is pixel-for-pixel what the pilot script itself sees.

Usage:
    python lane_roi_calibrator.py                 # auto-scans for the game window
    python lane_roi_calibrator.py screenshot.png   # or calibrate from a saved image

Controls:
    left click   - add a point to the line/stage you're currently marking
    n            - confirm this stage, move to the next one
    z            - undo last point
    r            - clear all points in the current stage and restart it
    g            - re-scan the window for a fresh frame (live-scan mode only)
    s            - save (only once all 3 stages have >= 2 points each)
    q / Esc      - quit without saving
"""
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

import cv2
import numpy as np
import json
import sys
import os
import time

try:
    import mss
    import pygetwindow as gw
    HAS_WINDOW_LIBS = True
except ImportError:
    HAS_WINDOW_LIBS = False

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lane_roi_config.json")

STAGES = ["LEFT lane line", "RIGHT lane line", "HOOD / dash cutoff line"]
COLORS = [(0, 0, 255), (255, 100, 0), (0, 255, 255)]   # BGR: red, blue, yellow
INSTRUCTIONS = [
    "Click a few points ALONG the left lane line, near car -> far away",
    "Click a few points ALONG the right lane line, near car -> far away",
    "Click 2+ points along the top edge of your hood/dash (left -> right)",
]


def find_game_window():
    """Identical to beamng_pilot_v16.py's find_game_window() — crops out the
    title bar + window border so the captured region is exactly the game's
    client area, same as what the pilot script processes."""
    try:
        windows = gw.getWindowsWithTitle("BeamNG")
        for win in windows:
            if "BeamNG.drive" in win.title or "BeamNG.tech" in win.title:
                if win.left < -30000 or win.top < -30000:
                    continue
                border_x, title_bar = 9, 32
                x = win.left + border_x
                y = win.top + title_bar
                w = win.width - (border_x * 2)
                h = win.height - title_bar - border_x
                if w <= 0 or h <= 0:
                    continue
                return {"top": y, "left": x, "width": w, "height": h}
        return None
    except Exception:
        return None


def grab_window():
    print(">> Scanning for BeamNG window...")
    monitor = None
    for _ in range(30):
        monitor = find_game_window()
        if monitor:
            print(f">> Window found: {monitor}")
            break
        time.sleep(1)
    if monitor is None:
        print("[ERROR] BeamNG window not found after 30s. Is the game running and visible")
        print("        (not minimized)? You can also pass a saved screenshot instead:")
        print("        python lane_roi_calibrator.py screenshot.png")
        sys.exit(1)
    with mss.mss() as sct:
        shot = np.array(sct.grab(monitor))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def load_image():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] couldn't read image: {path}")
            sys.exit(1)
        print(f"[INFO] loaded {path}")
        return img

    if not HAS_WINDOW_LIBS:
        print("[ERROR] mss/pygetwindow aren't available, and no image path was given.")
        print("        Usage: python lane_roi_calibrator.py [screenshot.png]")
        sys.exit(1)

    return grab_window()


def save_config(points, w, h):
    left, right, hood = points

    def fx(p): return round(p[0] / w, 4)
    def fy(p): return round(p[1] / h, 4)

    left_bottom  = max(left,  key=lambda p: p[1])   # nearest to car
    left_top     = min(left,  key=lambda p: p[1])   # farthest marked point
    right_bottom = max(right, key=lambda p: p[1])
    right_top    = min(right, key=lambda p: p[1])

    # Camera pitched high -> stop trusting the road at whichever lane line's
    # farthest marked point is HIGHEST on screen (smallest y), instead of
    # searching all the way to the true horizon (bridges/guardrails/overpasses
    # up there get picked up as fake lane lines otherwise).
    roi_top_crop = min(fy(left_top), fy(right_top))

    # Hood/dash cutoff -> the highest (smallest y) point you clicked on it,
    # so no part of the hood/dash can ever end up inside the ROI.
    roi_bottom_crop = min(fy(p) for p in hood)

    config = {
        "roi_top_crop":               roi_top_crop,
        "roi_bottom_crop":            roi_bottom_crop,
        "roi_poly_bottom_left_frac":  fx(left_bottom),
        "roi_poly_top_left_frac":     fx(left_top),
        "roi_poly_bottom_right_frac": fx(right_bottom),
        "roi_poly_top_right_frac":    fx(right_top),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[OK] saved -> {OUTPUT_PATH}")
    for k, v in config.items():
        print(f"   {k:28s} = {v}")
    print("\nDrop this file next to beamng_pilot_v16.py (same folder) and it")
    print("will be picked up automatically on the next run.")


def main():
    img = load_image()
    h, w = img.shape[:2]
    print(f"[INFO] image size: {w}x{h}")
    print(f"\n[1/3] {INSTRUCTIONS[0]}")
    print("      'n' next stage | 'z' undo | 'r' restart stage | 's' save | 'q' quit\n")

    points = [[], [], []]   # left, right, hood
    stage = 0

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and stage < 3:
            points[stage].append((x, y))

    win_name = "Lane ROI Calibrator"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_click)

    while True:
        disp = img.copy()

        for s in range(3):
            pts, col = points[s], COLORS[s]
            for i in range(1, len(pts)):
                cv2.line(disp, pts[i - 1], pts[i], col, 3)
            for p in pts:
                cv2.circle(disp, p, 5, col, -1)

        cv2.rectangle(disp, (0, 0), (w, 46), (0, 0, 0), -1)
        if stage < 3:
            cv2.putText(disp, f"[{stage + 1}/3] Mark: {STAGES[stage]}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(disp, INSTRUCTIONS[stage],
                        (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        else:
            cv2.putText(disp, "All 3 marked -- press 's' to save, 'r'/'z' to fix a stage",
                        (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(disp, "'g' re-scan window", (w - 190, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow(win_name, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('g') and HAS_WINDOW_LIBS and len(sys.argv) <= 1:
            img = grab_window()
            h, w = img.shape[:2]
            print("[INFO] re-scanned window — points kept, adjust as needed.")
            continue

        if key == ord('n') and stage < 3:
            if len(points[stage]) < 2:
                print(f"[WARN] need at least 2 points for {STAGES[stage]} first.")
            else:
                stage += 1
                if stage < 3:
                    print(f"\n[{stage + 1}/3] {INSTRUCTIONS[stage]}")

        elif key == ord('z'):
            active = min(stage, 2)
            if points[active]:
                points[active].pop()

        elif key == ord('r'):
            active = min(stage, 2)
            points[active] = []
            print(f"[INFO] cleared {STAGES[active]} — mark it again.")

        elif key == ord('s'):
            if any(len(p) < 2 for p in points):
                print("[WARN] all three (left / right / hood) need >= 2 points before saving.")
                continue
            save_config(points, w, h)
            break

        elif key in (ord('q'), 27):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
