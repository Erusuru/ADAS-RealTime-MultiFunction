"""
Integrated Forward ADAS Perception Pipeline
===========================================
Author: Ramazan Ertugrul Aydogan
Affiliation: South-West University Neofit Rilski, Blagoevgrad, Bulgaria

Integrates Lane Departure Warning, YOLO Object Detection, Vector FCW, and Pinhole Distance Estimation.
"""

import cv2
import numpy as np
import time
from typing import Tuple, List, Dict
from algorithms.lane_departure_warning import LaneDepartureWarning
from algorithms.vector_fcw_physics import VectorFCWPhysicsEngine
from algorithms.monocular_distance import MonocularDistanceEstimator


class ForwardADASPipeline:
    def __init__(self, yolo_model_path: str = "yolov8n.pt", conf_thresh: float = 0.40):
        self.ldw = LaneDepartureWarning()
        self.fcw = VectorFCWPhysicsEngine()
        self.distance_estimator = MonocularDistanceEstimator()
        self.conf_thresh = conf_thresh

        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(yolo_model_path)
        except Exception:
            print("[WARN] YOLO model could not be loaded. Running geometric fallback.")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        h, w = frame.shape[:2]
        now = time.time()

        # 1. Lane Detection & Ego Corridor
        overlay, ldw_state, lateral_offset, ego_polygon = self.ldw.process_frame(frame)

        # 2. Object Detection
        active_threats = []
        detected_objects = []

        if self.model is not None:
            results = self.model(frame, verbose=False, conf=self.conf_thresh)
            active_ids = []

            for r in results:
                boxes = r.boxes
                for idx, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    if cls_name not in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person']:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    bbox_w = x2 - x1

                    dist_m = self.distance_estimator.estimate_distance(cls_name, bbox_w)
                    obj_id = int(box.id[0]) if box.id is not None else idx

                    active_ids.append(obj_id)
                    self.fcw.update_track(obj_id, cx, cy, bbox_w, dist_m, now)
                    threat = self.fcw.evaluate_threat(obj_id, (h, w), ego_polygon)

                    # Visual Rendering
                    is_threat = (threat['alert_level'] != 'SAFE')
                    box_color = (0, 0, 255) if is_threat else (0, 255, 0)
                    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2 if not is_threat else 3)

                    label = f"{cls_name} {dist_m:.1f}m"
                    if threat['ttc_s'] is not None and threat['ttc_s'] < 10.0:
                        label += f" | TTC:{threat['ttc_s']:.1f}s"
                    if is_threat:
                        label += f" [{threat['alert_level']}]"
                        active_threats.append(threat)

                    cv2.putText(overlay, label, (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            self.fcw.purge_lost_tracks(active_ids)

        # 3. Collision Imminent HUD Banner
        overall_threat = "SAFE"
        if active_threats:
            overall_threat = "COLLISION_IMMINENT"
            # Red top banner
            cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 255), -1)
            cv2.putText(overlay, "RED - EMERGENCY BRAKE - COLLISION IMMINENT", (int(w * 0.1), 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

        pipeline_summary = {
            'ldw_state': ldw_state,
            'lateral_offset_px': lateral_offset,
            'threat_level': overall_threat,
            'active_threats': active_threats
        }

        return overlay, pipeline_summary
