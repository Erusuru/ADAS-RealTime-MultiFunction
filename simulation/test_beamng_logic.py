"""
Sanity tests for the pure control-logic in beamng_pilot_v12_roadseg-detr.py —
the steering controller, the segmentation lane-deviation fit, and the
lane-change state machine. No BeamNG session, GPU, or game window needed:
the four game-interface packages (mss, keyboard, pydirectinput,
pygetwindow) are stubbed out purely so the file can be imported standalone —
nothing below actually calls them. Needs cv2 + numpy (already required to
run the real script) but not torch/ultralytics/beamngpy — those already
degrade gracefully via the try/except ImportError blocks at the top of the
script itself.

Run:  python3 test_beamng_logic.py
Exits 0 if everything passes, 1 otherwise — safe to wire into a pre-flight
check before a play session, or re-run any time you retune a constant.
"""
import sys
import time
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import cv2

for _name in ("mss", "keyboard", "pydirectinput", "pygetwindow"):
    sys.modules.setdefault(_name, MagicMock())

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("beamng_pilot", _HERE / "beamng_pilot_v12_roadseg-detr.py")
bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bp)

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  OK   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}  {detail}")


# ─────────────────────────────────────────────────────────────────────────
print("\n[1] Steering sign: lane center right of frame-center -> positive (right) steer")
s_right = bp.LaneAssist().calculate_analog_steer(deviation=100.0, engaged=True, speed_kmh=30.0)
check("deviation +100px -> steer > 0 (RIGHT)", s_right > 0, f"got {s_right}")

s_left = bp.LaneAssist().calculate_analog_steer(deviation=-100.0, engaged=True, speed_kmh=30.0)
check("deviation -100px -> steer < 0 (LEFT)", s_left < 0, f"got {s_left}")

check("deadzone: small deviation -> exactly 0",
      bp.LaneAssist().calculate_analog_steer(5.0, True, 30.0) == 0.0)
check("disengaged -> always 0",
      bp.LaneAssist().calculate_analog_steer(200.0, False, 30.0) == 0.0)

# ─────────────────────────────────────────────────────────────────────────
print("\n[2] Steering D-term damps a shrinking deviation more than a growing one")
la_grow = bp.LaneAssist()
la_grow.calculate_analog_steer(50.0, True, 30.0)
s_growing = la_grow.calculate_analog_steer(100.0, True, 30.0)

la_shrink = bp.LaneAssist()
la_shrink.calculate_analog_steer(150.0, True, 30.0)
s_shrinking = la_shrink.calculate_analog_steer(100.0, True, 30.0)

check("growing-deviation steer > shrinking-deviation steer at the same instantaneous dev",
      s_growing > s_shrinking, f"growing={s_growing} shrinking={s_shrinking}")

# ─────────────────────────────────────────────────────────────────────────
print("\n[3] Speed-adaptive gain: same deviation, gentler steer at higher speed")
s_slow = bp.LaneAssist().calculate_analog_steer(100.0, True, speed_kmh=10.0)
s_fast = bp.LaneAssist().calculate_analog_steer(100.0, True, speed_kmh=150.0)
check("steer(10 km/h) > steer(150 km/h) for the same deviation", s_slow > s_fast, f"slow={s_slow} fast={s_fast}")
check("gain never drops to 0 even at high speed", s_fast > 0, f"got {s_fast}")

# ─────────────────────────────────────────────────────────────────────────
print("\n[4] compute_seg_deviation: needs BOTH lane boundaries, ignores whole-road blobs")
w, h = 640, 360
blank = np.zeros((h, w), dtype=np.uint8)
dev, valid = bp.compute_seg_deviation(blank, w, h)
check("empty mask -> invalid", not valid)

# A solid whole-road blob is what the OLD buggy da_mask-centroid version
# consumed. The fixed version fits lane-boundary lines, so a filled blob
# with no distinct edges shouldn't produce a confident reading.
road_blob = blank.copy()
road_blob[int(h * 0.55):h, int(w * 0.05):int(w * 0.95)] = 255
dev_blob, valid_blob = bp.compute_seg_deviation(road_blob, w, h)
check("solid whole-road blob (old bug's input) -> no confident deviation",
      not valid_blob, f"dev={dev_blob} valid={valid_blob}")

two_lines = blank.copy()
cv2.line(two_lines, (int(w * 0.35), h), (int(w * 0.45), int(h * 0.55)), 255, 4)
cv2.line(two_lines, (int(w * 0.65), h), (int(w * 0.55), int(h * 0.55)), 255, 4)
dev_center, valid_center = bp.compute_seg_deviation(two_lines, w, h)
check("two centered lane lines -> valid", valid_center, f"valid={valid_center}")
if valid_center:
    check("two centered lane lines -> deviation near 0", abs(dev_center) < 25, f"got {dev_center:.1f}px")

shifted = blank.copy()
cv2.line(shifted, (int(w * 0.55), h), (int(w * 0.65), int(h * 0.55)), 255, 4)
cv2.line(shifted, (int(w * 0.85), h), (int(w * 0.75), int(h * 0.55)), 255, 4)
dev_shift, valid_shift = bp.compute_seg_deviation(shifted, w, h)
check("lane shifted right in-frame -> positive deviation (matches hough_dev's sign convention)",
      valid_shift and dev_shift > 0, f"valid={valid_shift} dev={dev_shift}")

# ─────────────────────────────────────────────────────────────────────────
print("\n[5] shift_polygon_for_lane_change: geometry sanity")
cx = w // 2
hw_b, hw_t = int(w * bp.EGO_LANE_HALF_W_BOTTOM), int(w * bp.EGO_LANE_HALF_W_TOP)
y_bot, y_top = h, int(h * 0.58)
ego_poly = np.array([(cx - hw_b, y_bot), (cx + hw_b, y_bot), (cx + hw_t, y_top), (cx - hw_t, y_top)], dtype=np.int32)

left_corridor = bp.shift_polygon_for_lane_change(ego_poly, -1, w)
right_corridor = bp.shift_polygon_for_lane_change(ego_poly, +1, w)
check("left corridor sits strictly left of the ego polygon",
      left_corridor[0][0] < ego_poly[0][0], f"{left_corridor[0][0]} vs {ego_poly[0][0]}")
check("right corridor sits strictly right of the ego polygon",
      right_corridor[1][0] > ego_poly[1][0], f"{right_corridor[1][0]} vs {ego_poly[1][0]}")
check("neither corridor overlaps the ego polygon's own footprint",
      left_corridor[1][0] <= ego_poly[0][0] and right_corridor[0][0] >= ego_poly[1][0],
      f"left_corridor right-edge={left_corridor[1][0]} vs ego left-edge={ego_poly[0][0]}; "
      f"right_corridor left-edge={right_corridor[0][0]} vs ego right-edge={ego_poly[1][0]}")

# ─────────────────────────────────────────────────────────────────────────
print("\n[6] target_lane_is_drivable")
da_full, da_empty = np.full((h, w), 255, dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
check("fully-drivable da_mask -> True", bp.target_lane_is_drivable(da_full, left_corridor, w, h))
check("empty da_mask -> False", not bp.target_lane_is_drivable(da_empty, left_corridor, w, h))
check("da_mask=None -> permissive True (matches is_off_road's own convention)",
      bp.target_lane_is_drivable(None, left_corridor, w, h))

# ─────────────────────────────────────────────────────────────────────────
print("\n[7] LaneChangeController: full lifecycle with a clear gap")
lc = bp.LaneChangeController()
check("starts IDLE", lc.state == lc.IDLE)

lc.request(-1)
d = lc.begin_frame(steering_ready=True, speed_kmh=40.0, off_road=False, current_max_threat=0)
check("valid request while ready -> direction set the same frame", d == -1, f"got {d}")
check("state -> CHECKING", lc.state == lc.CHECKING)

off, status = lc.end_frame(w, True, False, 0, target_lane_has_vehicle=False, target_lane_drivable=True)
check("still CHECKING right after the first clear reading (needs continuous time, not one sample)",
      lc.state == lc.CHECKING, f"state={lc.state}")
check("offset is 0 while still CHECKING (no lateral movement before committing)", off == 0.0, f"got {off}")

# Simulate the confirmation window having elapsed without an actual sleep.
lc.clear_since = time.time() - (bp.LANE_CHANGE_CHECK_SECONDS + 0.1)
lc._last_t = time.time() - 0.05
off, status = lc.end_frame(w, True, False, 0, target_lane_has_vehicle=False, target_lane_drivable=True)
check("commits to EXECUTING once continuously clear for LANE_CHANGE_CHECK_SECONDS",
      lc.state == lc.EXECUTING, f"state={lc.state}")

n = 0
while lc.state == lc.EXECUTING and n < 500:
    lc._last_t = time.time() - 0.033   # simulate a realistic ~30fps frame gap
    off, status = lc.end_frame(w, True, False, 0, target_lane_has_vehicle=False, target_lane_drivable=True)
    n += 1
check("EXECUTING eventually completes and returns to IDLE", lc.state == lc.IDLE, f"state={lc.state} after {n} steps")
check("direction resets to 0 on completion", lc.direction == 0)
check("final offset is exactly 0 once idle again", off == 0.0, f"got {off}")

# ─────────────────────────────────────────────────────────────────────────
print("\n[8] LaneChangeController: aborts if a vehicle appears mid-execution")
lc2 = bp.LaneChangeController()
lc2.request(1)
lc2.begin_frame(True, 40.0, False, 0)
lc2.clear_since = time.time() - (bp.LANE_CHANGE_CHECK_SECONDS + 0.1)
lc2._last_t = time.time() - 0.05
lc2.end_frame(w, True, False, 0, False, True)
check("committed to EXECUTING", lc2.state == lc2.EXECUTING)
for _ in range(5):
    lc2._last_t = time.time() - 0.033
    lc2.end_frame(w, True, False, 0, False, True)
check("made some progress before the abort trigger", lc2.progress > 0.0)
lc2.end_frame(w, True, False, 0, target_lane_has_vehicle=True, target_lane_drivable=True)
check("a vehicle appearing mid-EXECUTING triggers ABORTING", lc2.state == lc2.ABORTING, f"state={lc2.state}")
n = 0
off_abort = None
while lc2.state == lc2.ABORTING and n < 500:
    lc2._last_t = time.time() - 0.033
    off_abort, _ = lc2.end_frame(w, True, False, 0, True, True)
    n += 1
check("ABORTING eventually returns fully to IDLE at offset 0",
      lc2.state == lc2.IDLE and off_abort == 0.0, f"state={lc2.state} offset={off_abort}")

# ─────────────────────────────────────────────────────────────────────────
print("\n[9] LaneChangeController: refuses to start when it isn't actually safe to")
for label, kw in [
    ("too slow", dict(steering_ready=True, speed_kmh=5.0, off_road=False, current_max_threat=0)),
    ("off-road", dict(steering_ready=True, speed_kmh=40.0, off_road=True, current_max_threat=0)),
    ("active AEB threat", dict(steering_ready=True, speed_kmh=40.0, off_road=False, current_max_threat=2)),
    ("steering not ready", dict(steering_ready=False, speed_kmh=40.0, off_road=False, current_max_threat=0)),
]:
    lcx = bp.LaneChangeController()
    lcx.request(-1)
    lcx.begin_frame(**kw)
    check(f"{label} -> request ignored, stays IDLE", lcx.state == lcx.IDLE, f"state={lcx.state}")

# ─────────────────────────────────────────────────────────────────────────
print("\n[10] LaneChangeController: gap that never clears -> gives up, returns to IDLE")
lc7 = bp.LaneChangeController()
lc7.request(1)
lc7.begin_frame(True, 40.0, False, 0)
lc7.checking_since = time.time() - (bp.LANE_CHANGE_CHECK_TIMEOUT_S + 0.5)
lc7._last_t = time.time() - 0.05
lc7.end_frame(w, True, False, 0, target_lane_has_vehicle=True, target_lane_drivable=True)
check("timeout while never clear -> gives up to IDLE (not stuck forever)", lc7.state == lc7.IDLE, f"state={lc7.state}")

print(f"\n{'=' * 60}\n{passed} passed, {failed} failed\n{'=' * 60}")
sys.exit(1 if failed else 0)
