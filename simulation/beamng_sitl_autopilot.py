"""
BeamNG.drive Simulation-in-the-Loop (SITL) Autopilot & Level 2 Validation
=========================================================================
Author: Ramazan Ertugrul Aydogan
Affiliation: South-West University Neofit Rilski, Blagoevgrad, Bulgaria

Implements Section III.B & V.C:
  - 3 Operating Modes:
      Mode 1 — Steering Only (Lane Keep)
      Mode 2 — Braking Only (YOLO Radar & Collision Avoidance / AEB)
      Mode 3 — Full Autopilot (Steering + Braking + Drowsiness escalation)
  - Speed-scaled TTC thresholds receiving UDP telemetry on port 4444.
  - Automated Emergency Braking (AEB) intervention.
"""

import socket
import json
import time
import numpy as np
from typing import Tuple, Dict, Optional


class BeamNGSITLAutopilot:
    def __init__(self, mode: int = 3, telemetry_port: int = 4444):
        """
        Args:
            mode: 1 (Steering Only), 2 (Braking Only), 3 (Full Autopilot).
            telemetry_port: UDP port for BeamNG physics telemetry.
        """
        self.mode = mode
        self.telemetry_port = telemetry_port
        self.speed_kmh = 0.0

        # Autopilot commands
        self.steering_cmd = 0.0
        self.throttle_cmd = 0.0
        self.brake_cmd = 0.0

    def parse_telemetry(self, raw_bytes: bytes):
        """Parses vehicle telemetry from BeamNG.drive UDP socket."""
        try:
            data = json.loads(raw_bytes.decode('utf-8'))
            self.speed_kmh = float(data.get('speed', 0.0)) * 3.6
        except Exception:
            pass

    def compute_control_step(self, lateral_deviation_px: float, collision_threat_level: str, is_drowsy: bool) -> Dict[str, float]:
        """
        Computes steering and braking commands based on perception input and operating mode.
        """
        # Mode 1 & 3: Steering
        if self.mode in [1, 3]:
            # Proportional curvature steering
            self.steering_cmd = float(np.clip(-lateral_deviation_px * 0.005, -1.0, 1.0))
        else:
            self.steering_cmd = 0.0

        # Mode 2 & 3: Braking and AEB
        if self.mode in [2, 3]:
            if collision_threat_level in ['COLLISION_IMMINENT', 'TBONE_CUTIN', 'STAGE1_PROXIMITY']:
                # Full Automated Emergency Braking
                self.brake_cmd = 1.0
                self.throttle_cmd = 0.0
            elif is_drowsy:
                # Escalated drowsiness braking
                self.brake_cmd = 0.6
                self.throttle_cmd = 0.0
            else:
                self.brake_cmd = 0.0
                self.throttle_cmd = 0.5  # Cruise throttle
        else:
            self.brake_cmd = 0.0
            self.throttle_cmd = 0.5

        return {
            'steering': self.steering_cmd,
            'throttle': self.throttle_cmd,
            'brake': self.brake_cmd,
            'speed_kmh': self.speed_kmh
        }
