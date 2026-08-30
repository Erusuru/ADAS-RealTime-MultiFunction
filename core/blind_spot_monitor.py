"""
Blind Spot Ultrasonic Detection Pipeline
=========================================
Author: Ramazan Ertugrul Aydogan
Affiliation: South-West University Neofit Rilski, Blagoevgrad, Bulgaria

Implements rear-flank ultrasonic monitoring from Section II.D:
  - Continuously polls JSN-SR04T waterproof ultrasonic sensors on left/right rear flanks.
  - Danger zone threshold: <= 3.0 meters triggers side-mirror warning LED.
"""

import time
from typing import Dict


class BlindSpotMonitor:
    def __init__(self, danger_threshold_m: float = 3.0):
        self.danger_threshold_m = danger_threshold_m

    def evaluate_sensors(self, left_distance_m: float, right_distance_m: float) -> Dict[str, bool]:
        """
        Evaluates ultrasonic distance readings and returns side-mirror warning LED triggers.
        """
        return {
            'left_mirror_led': (0.1 <= left_distance_m <= self.danger_threshold_m),
            'right_mirror_led': (0.1 <= right_distance_m <= self.danger_threshold_m)
        }
