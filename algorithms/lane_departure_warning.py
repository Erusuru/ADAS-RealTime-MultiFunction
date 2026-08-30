"""
Lane Departure Warning (LDW) Algorithm
======================================
Author: Ramazan Ertugrul Aydogan
Affiliation: South-West University Neofit Rilski, Blagoevgrad, Bulgaria

Implements the classical computer vision lane detection pipeline described in:
"Real-Time, Multi-Function ADAS Application", Section II.C.

Pipeline Stages:
  1. Gamma Correction and CLAHE Contrast Enhancement
  2. HLS Color Space Filtering (White and Yellow Lane Isolation)
  3. Gaussian Noise Suppression and Canny Edge Detection
  4. Trapezoidal Region of Interest (ROI) Masking
  5. Probabilistic Hough Line Transform (cv2.HoughLinesP)
  6. Slope Grouping and Exponential Moving Average (EMA, alpha = 0.8) Smoothing
  7. Lateral Deviation Calculation (>5% screen width triggers alert)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class LaneDepartureWarning:
    def __init__(self, ema_alpha: float = 0.80, deviation_threshold_pct: float = 0.05):
        """
        Initializes the Lane Departure Warning subsystem.
        """
        self.ema_alpha = ema_alpha
        self.deviation_threshold_pct = deviation_threshold_pct
        self.smoothed_left: Optional[Tuple[float, float]] = None
        self.smoothed_right: Optional[Tuple[float, float]] = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def apply_gamma_clahe(self, frame: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """Applies gamma correction and CLAHE contrast enhancement."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        gamma_corrected = cv2.LUT(frame, table)

        lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        cl = self.clahe.apply(l_channel)
        merged = cv2.merge((cl, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def filter_lane_colors(self, frame: np.ndarray) -> np.ndarray:
        """Isolates white and yellow lane markings in HLS color space."""
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, 180, 0], dtype=np.uint8)
        upper_white = np.array([180, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hls, lower_white, upper_white)

        lower_yellow = np.array([15, 38, 100], dtype=np.uint8)
        upper_yellow = np.array([35, 204, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        return cv2.bitwise_and(frame, frame, mask=combined_mask)

    def extract_edges(self, color_filtered: np.ndarray) -> np.ndarray:
        """Grayscale conversion, Gaussian blur, and Canny edge detection."""
        gray = cv2.cvtColor(color_filtered, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, threshold1=50, threshold2=150)

    def get_trapezoidal_roi_mask(self, h: int, w: int) -> np.ndarray:
        """Generates trapezoidal Region of Interest polygon mask."""
        mask = np.zeros((h, w), dtype=np.uint8)
        roi_polygon = np.array([
            [int(w * 0.05), int(h * 0.95)],
            [int(w * 0.40), int(h * 0.58)],
            [int(w * 0.60), int(h * 0.58)],
            [int(w * 0.95), int(h * 0.95)]
        ], np.int32)
        cv2.fillPoly(mask, [roi_polygon], 255)
        return mask

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str, float, Optional[np.ndarray]]:
        """
        Executes the complete LDW pipeline on an input frame.
        """
        h, w = frame.shape[:2]
        enhanced = self.apply_gamma_clahe(frame)
        color_filtered = self.filter_lane_colors(enhanced)
        edges = self.extract_edges(color_filtered)
        roi_mask = self.get_trapezoidal_roi_mask(h, w)
        masked_edges = cv2.bitwise_and(edges, roi_mask)

        lines = cv2.HoughLinesP(
            masked_edges, rho=1, theta=np.pi / 180, threshold=25, minLineLength=30, maxLineGap=20
        )
        left_pts, right_pts = [], []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if -1.5 < slope < -0.3:
                    left_pts.extend([(x1, y1), (x2, y2)])
                elif 0.3 < slope < 1.5:
                    right_pts.extend([(x1, y1), (x2, y2)])

        left_line = self._fit_line(left_pts)
        right_line = self._fit_line(right_pts)

        if left_line is not None:
            self.smoothed_left = left_line if self.smoothed_left is None else (
                self.ema_alpha * np.array(self.smoothed_left) + (1 - self.ema_alpha) * np.array(left_line)
            )
        if right_line is not None:
            self.smoothed_right = right_line if self.smoothed_right is None else (
                self.ema_alpha * np.array(self.smoothed_right) + (1 - self.ema_alpha) * np.array(right_line)
            )

        overlay = frame.copy()
        ego_polygon = None
        warning_state = "SAFE"
        lateral_offset = 0.0

        if self.smoothed_left is not None and self.smoothed_right is not None:
            y_bottom = int(h * 0.95)
            y_top = int(h * 0.58)

            xl_bot = int((y_bottom - self.smoothed_left[1]) / self.smoothed_left[0]) if self.smoothed_left[0] != 0 else int(w * 0.2)
            xl_top = int((y_top - self.smoothed_left[1]) / self.smoothed_left[0]) if self.smoothed_left[0] != 0 else int(w * 0.4)
            xr_bot = int((y_bottom - self.smoothed_right[1]) / self.smoothed_right[0]) if self.smoothed_right[0] != 0 else int(w * 0.8)
            xr_top = int((y_top - self.smoothed_right[1]) / self.smoothed_right[0]) if self.smoothed_right[0] != 0 else int(w * 0.6)

            lane_center = (xl_bot + xr_bot) / 2.0
            vehicle_center = w / 2.0
            lateral_offset = vehicle_center - lane_center
            threshold_px = w * self.deviation_threshold_pct

            if lateral_offset > threshold_px:
                warning_state = "DEVIATION_RIGHT"
            elif lateral_offset < -threshold_px:
                warning_state = "DEVIATION_LEFT"

            line_color = (0, 0, 255) if warning_state != "SAFE" else (0, 255, 0)
            cv2.line(overlay, (xl_bot, y_bottom), (xl_top, y_top), line_color, 4)
            cv2.line(overlay, (xr_bot, y_bottom), (xr_top, y_top), line_color, 4)

            ego_polygon = np.array([[xl_bot, y_bottom], [xl_top, y_top], [xr_top, y_top], [xr_bot, y_bottom]], np.int32)
            poly_overlay = overlay.copy()
            corridor_color = (0, 80, 255) if warning_state != "SAFE" else (0, 180, 0)
            cv2.fillPoly(poly_overlay, [ego_polygon], corridor_color)
            cv2.addWeighted(poly_overlay, 0.25, overlay, 0.75, 0, overlay)
        else:
            warning_state = "NO_LANES"

        return overlay, warning_state, lateral_offset, ego_polygon

    def _fit_line(self, points: List[Tuple[int, int]]) -> Optional[Tuple[float, float]]:
        if len(points) < 2:
            return None
        pts = np.array(points)
        x = pts[:, 0]
        y = pts[:, 1]
        try:
            poly = np.polyfit(x, y, 1)
            return float(poly[0]), float(poly[1])
        except Exception:
            return None
