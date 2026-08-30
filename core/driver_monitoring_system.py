"""
Driver Monitoring System (DMS) & Eye Aspect Ratio (EAR) Tracker
================================================================
Author: Ramazan Ertugrul Aydogan
Affiliation: South-West University Neofit Rilski, Blagoevgrad, Bulgaria

Implements the facial landmark drowsiness detection pipeline from Section II.E:
  EAR = (||p1 - p5|| + ||p2 - p4||) / (2 * ||p0 - p3||)

Triggers Drowsy Alert if bilateral averaged EAR < 0.22 for > 1.5 consecutive seconds.
"""

import time
import numpy as np
import cv2
from typing import Tuple, Optional


class DriverMonitoringSystem:
    # 6 landmark indices per eye in MediaPipe Face Mesh
    LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]   # [p0, p1, p2, p3, p4, p5]
    RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380] # [p0, p1, p2, p3, p4, p5]

    def __init__(self, ear_threshold: float = 0.22, drowsy_duration_s: float = 1.5):
        self.ear_threshold = ear_threshold
        self.drowsy_duration_s = drowsy_duration_s

        self.closure_start_time: Optional[float] = None
        self.is_drowsy: bool = False
        self.mp_face_mesh = None

        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except ImportError:
            print("[WARN] MediaPipe not installed. DMS will run in fallback simulation mode.")

    def compute_ear(self, landmarks, eye_indices, w: int, h: int) -> float:
        """Computes Eye Aspect Ratio from 6 landmarks."""
        coords = np.array([[landmarks[idx].x * w, landmarks[idx].y * h] for idx in eye_indices])
        p0, p1, p2, p3, p4, p5 = coords

        v1 = np.linalg.norm(p1 - p5)
        v2 = np.linalg.norm(p2 - p4)
        h_dist = np.linalg.norm(p0 - p3)

        if h_dist == 0:
            return 0.0
        return float((v1 + v2) / (2.0 * h_dist))

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """
        Processes driver-facing camera frame and evaluates drowsiness state.
        """
        h, w = frame.shape[:2]
        avg_ear = 0.30
        now = time.time()

        if self.mp_face_mesh is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                left_ear = self.compute_ear(lms, self.LEFT_EYE_LANDMARKS, w, h)
                right_ear = self.compute_ear(lms, self.RIGHT_EYE_LANDMARKS, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

        # Evaluate drowsiness threshold
        if avg_ear < self.ear_threshold:
            if self.closure_start_time is None:
                self.closure_start_time = now
            elif (now - self.closure_start_time) >= self.drowsy_duration_s:
                self.is_drowsy = True
        else:
            self.closure_start_time = None
            self.is_drowsy = False

        overlay = frame.copy()
        status_text = f"EAR: {avg_ear:.2f} | STATUS: {'DROWSY! ALARM!' if self.is_drowsy else 'ATTENTIVE'}"
        color = (0, 0, 255) if self.is_drowsy else (0, 255, 0)
        cv2.putText(overlay, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return overlay, self.is_drowsy, avg_ear
