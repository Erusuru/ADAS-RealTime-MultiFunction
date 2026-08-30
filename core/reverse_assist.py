"""
Reverse Assist and Ultrasonic Parking Subsystem
===============================================
Author: Ramazan Ertugrul Aydogan
Affiliation: South-West University Neofit Rilski, Blagoevgrad, Bulgaria

Implements Reverse Assist Pipeline from Section II.C:
  - Launches low-latency rear camera feed via ffplay subprocess upon reverse engagement.
  - JSN-SR04T ultrasonic distance measurement with proportional buzzer beep modulation.
"""

import subprocess
import time
from typing import Optional


class ReverseAssistPipeline:
    def __init__(self, rear_camera_index: int = 1):
        self.rear_camera_index = rear_camera_index
        self.ffplay_process: Optional[subprocess.Popen] = None

    def start_camera_stream(self):
        """Launches ffplay subprocess for lowest-latency display."""
        if self.ffplay_process is None:
            cmd = ["ffplay", "-fast", "-nodisp", f"/dev/video{self.rear_camera_index}"]
            try:
                self.ffplay_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    def stop_camera_stream(self):
        """Terminates ffplay subprocess upon exiting reverse."""
        if self.ffplay_process is not None:
            self.ffplay_process.terminate()
            self.ffplay_process = None

    def compute_buzzer_interval_ms(self, obstacle_distance_m: float) -> Optional[int]:
        """
        Modulates buzzer beep interval inversely proportional to obstacle distance.
        """
        if obstacle_distance_m > 2.0:
            return None  # Silent
        elif obstacle_distance_m <= 0.30:
            return 0     # Continuous tone
        else:
            # Linear scaling from 500ms (at 2m) to 50ms (at 0.3m)
            pct = (obstacle_distance_m - 0.30) / (2.0 - 0.30)
            return int(50 + (pct * 450))
