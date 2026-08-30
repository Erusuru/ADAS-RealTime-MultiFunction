"""
Vector-Based Forward Collision Warning (FCW) Physics Engine
============================================================
Author: Ramazan Ertugrul Aydogan
Affiliation: South-West University Neofit Rilski, Blagoevgrad, Bulgaria

Implements PROCEDURE VECTOR_BASED_FCW_LOGIC from Section II.C of the MDPI Report:
  - Historical Trajectory Tracking (10-frame Deque)
  - Velocity Vector Calculation (vx, vy, approach_speed_ms)
  - Future Path Projection (1.0s ahead)
  - Ego Lane Polygon Intersection Test (PointInPolygon)
  - Lateral Threat Detection (T-Bone & Cut-In logic)
  - Two-Stage Alert Thresholds:
      Stage 1: Distance < 8.0 m
      Stage 2: Distance <= 60.0 m AND Time-to-Collision (TTC) < 2.5 s
"""

import time
import numpy as np
import cv2
from collections import deque
from typing import Dict, Tuple, List, Optional


class VectorFCWPhysicsEngine:
    def __init__(self, history_len: int = 10, lateral_speed_theta: float = 2.0, time_horizon_s: float = 1.0):
        """
        Initializes the deterministic Vector-Based FCW engine.
        """
        self.history_len = history_len
        self.lateral_speed_theta = lateral_speed_theta
        self.time_horizon_s = time_horizon_s

        # Dict mapping object_id -> deque of (cx, cy, width, timestamp, distance_m)
        self.track_history: Dict[int, deque] = {}

    def update_track(self, object_id: int, cx: float, cy: float, width_px: float, distance_m: float, timestamp: Optional[float] = None):
        """Records a new detection for a tracked object."""
        if timestamp is None:
            timestamp = time.time()

        if object_id not in self.track_history:
            self.track_history[object_id] = deque(maxlen=self.history_len)

        self.track_history[object_id].append({
            'cx': cx,
            'cy': cy,
            'w': width_px,
            'dist': distance_m,
            't': timestamp
        })

    def purge_lost_tracks(self, active_ids: List[int]):
        """Removes tracked entries no longer detected."""
        active_set = set(active_ids)
        for obj_id in list(self.track_history.keys()):
            if obj_id not in active_set:
                del self.track_history[obj_id]

    def evaluate_threat(self, object_id: int, frame_shape: Tuple[int, int], ego_polygon: Optional[np.ndarray]) -> Dict:
        """
        Evaluates collision threat for a tracked object according to PROCEDURE VECTOR_BASED_FCW_LOGIC.
        """
        h, w = frame_shape[:2]
        history = self.track_history.get(object_id, None)

        result = {
            'object_id': object_id,
            'alert_level': 'SAFE',      # 'SAFE', 'STAGE1_PROXIMITY', 'STAGE2_TTC', 'TBONE_CUTIN'
            'ttc_s': None,
            'approach_speed_ms': 0.0,
            'vx': 0.0,
            'vy': 0.0,
            'future_cx': None,
            'future_cy': None,
            'will_hit_ego': False,
            'is_cut_in': False,
            'distance_m': history[-1]['dist'] if history else 0.0
        }

        if history is None or len(history) < 3:
            return result

        first = history[0]
        latest = history[-1]
        dt = latest['t'] - first['t']

        if dt <= 0.001:
            return result

        # Velocity in pixels/sec
        vx = (latest['cx'] - first['cx']) / dt
        vy = (latest['cy'] - first['cy']) / dt
        result['vx'] = vx
        result['vy'] = vy

        # Approach speed in meters/sec (positive = closing in)
        approach_speed_ms = (first['dist'] - latest['dist']) / dt
        result['approach_speed_ms'] = approach_speed_ms

        # Compute TTC
        if approach_speed_ms > 0.5:
            ttc = latest['dist'] / approach_speed_ms
            result['ttc_s'] = ttc
        else:
            ttc = float('inf')

        # Project future centroid position
        future_cx = latest['cx'] + (vx * self.time_horizon_s)
        future_cy = latest['cy'] + (vy * self.time_horizon_s)
        result['future_cx'] = future_cx
        result['future_cy'] = future_cy

        # Ego lane polygon test
        will_hit = False
        if ego_polygon is not None and len(ego_polygon) >= 3:
            # pointPolygonTest returns >= 0 if inside or on edge
            dist_test = cv2.pointPolygonTest(ego_polygon.astype(np.float32), (float(future_cx), float(future_cy)), False)
            will_hit = (dist_test >= 0)
        else:
            # Fallback default corridor: center 40% of lower half
            will_hit = (w * 0.30 <= future_cx <= w * 0.70) and (future_cy >= h * 0.50)

        result['will_hit_ego'] = will_hit

        # T-Bone / Cut-In Logic
        # Objects in the middle vertical band (30-80% height) with high lateral velocity toward center
        in_middle_band = (h * 0.30 <= latest['cy'] <= h * 0.80)
        is_moving_towards_center = (latest['cx'] < w / 2 and vx > 0) or (latest['cx'] > w / 2 and vx < 0)
        is_cut_in = (
            in_middle_band and
            abs(vx) > abs(vy) * 1.5 and
            abs(vx) > (self.lateral_speed_theta * 20.0) and
            is_moving_towards_center
        )
        result['is_cut_in'] = is_cut_in

        # Two-stage alert assessment
        new_dist = latest['dist']
        stage1 = (new_dist < 8.0)
        stage2 = (new_dist <= 60.0 and ttc < 2.5)

        if will_hit or is_cut_in:
            if is_cut_in:
                result['alert_level'] = 'TBONE_CUTIN'
            elif stage1:
                result['alert_level'] = 'STAGE1_PROXIMITY'
            elif stage2:
                result['alert_level'] = 'STAGE2_TTC'

        return result
