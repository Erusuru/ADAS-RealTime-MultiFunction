"""
Intelligent Speed Adaptation (ISA) Module
==========================================
Author: Ramazan Ertugrul Aydogan
Affiliation: South-West University Neofit Rilski, Blagoevgrad, Bulgaria

Implements dual-mode Intelligent Speed Adaptation described in Section IV.C:
  1. Advisory Mode: Compares detected speed limit sign against vehicle speed and triggers warnings.
  2. SITL Active Control Mode: Overwrites vehicle cruise speed in simulation for automated deceleration.
"""

from typing import Optional, Tuple


class IntelligentSpeedAdaptation:
    def __init__(self, overspeed_tolerance_kmh: float = 5.0):
        self.overspeed_tolerance_kmh = overspeed_tolerance_kmh
        self.current_speed_limit_kmh: Optional[int] = None
        self.last_detected_time: float = 0.0

    def update_speed_limit(self, speed_limit_kmh: int):
        """Updates active regulatory speed limit from TSR detection."""
        self.current_speed_limit_kmh = speed_limit_kmh

    def evaluate_speed(self, current_vehicle_speed_kmh: float) -> Tuple[str, Optional[float]]:
        """
        Evaluates vehicle speed compliance.

        Returns:
            advisory_state: 'COMPLIANT', 'OVERSPEED_WARNING', or 'NO_LIMIT_SET'.
            target_safe_speed: Recommended maximum speed in km/h.
        """
        if self.current_speed_limit_kmh is None:
            return "NO_LIMIT_SET", None

        excess = current_vehicle_speed_kmh - self.current_speed_limit_kmh
        if excess > self.overspeed_tolerance_kmh:
            return "OVERSPEED_WARNING", float(self.current_speed_limit_kmh)
        return "COMPLIANT", float(self.current_speed_limit_kmh)
