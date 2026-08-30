"""
Monocular Pinhole Distance Estimation
=====================================
Author: Ramazan Ertugrul Aydogan
Affiliation: South-West University Neofit Rilski, Blagoevgrad, Bulgaria

Implements Pinhole Camera Model distance estimation from Section II.C:
  Distance = (Focal_Length * Real_Width) / Pixel_Width

Known-width Reference Table:
  - Car: 2.0 m
  - Truck: 2.5 m
  - Bus: 3.0 m
  - Pedestrian: 0.5 m
  - Default: 2.0 m
"""

from typing import Dict


class MonocularDistanceEstimator:
    # Standard metric widths according to MDPI specification
    REFERENCE_WIDTHS: Dict[str, float] = {
        'car': 2.0,
        'truck': 2.5,
        'bus': 3.0,
        'motorcycle': 0.8,
        'bicycle': 0.6,
        'biker': 0.8,
        'pedestrian': 0.5,
        'person': 0.5,
        'default': 2.0
    }

    def __init__(self, focal_length_px: float = 1200.0):
        """
        Initializes the distance estimator.
        Args:
            focal_length_px: Calibrated camera focal length in pixels.
        """
        self.focal_length_px = focal_length_px

    def estimate_distance(self, obj_class: str, bbox_width_px: float) -> float:
        """
        Computes metric distance from camera plane to target object.
        """
        if bbox_width_px <= 1.0:
            return 100.0  # Max default distance

        real_width = self.REFERENCE_WIDTHS.get(obj_class.lower(), self.REFERENCE_WIDTHS['default'])
        distance = (self.focal_length_px * real_width) / bbox_width_px
        return float(np.clip(distance, 0.5, 120.0)) if 'np' in globals() else max(0.5, min(120.0, distance))
