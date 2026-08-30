"""
Intelligent Automatic Headlight Control with Hysteresis
=======================================================
Author: Ramazan Ertugrul Aydogan
Affiliation: South-West University Neofit Rilski, Blagoevgrad, Bulgaria

Implements LDR light sensing and relay control from Section II.C:
  - Turn-ON Threshold: Ambient light < 30% for > 2 consecutive seconds.
  - Turn-OFF Threshold: Ambient light > 50% for > 2 consecutive seconds.
  - Controls 12V automotive relay via optocoupler isolation.
"""

import time
from typing import Tuple


class AutoHeadlightController:
    def __init__(self, on_threshold_pct: float = 30.0, off_threshold_pct: float = 50.0, debounce_s: float = 2.0):
        self.on_threshold = on_threshold_pct
        self.off_threshold = off_threshold_pct
        self.debounce_s = debounce_s

        self.headlights_active: bool = False
        self.condition_start_time: Optional[float] = None
        self.pending_state: Optional[bool] = None

    def update(self, ambient_light_pct: float) -> Tuple[bool, str]:
        """
        Updates light sensor reading and evaluates headlight relay state.

        Returns:
            headlights_active: True if headlights are currently turned on.
            status_msg: Description of controller status.
        """
        now = time.time()

        if not self.headlights_active:
            if ambient_light_pct < self.on_threshold:
                if self.pending_state is not True:
                    self.pending_state = True
                    self.condition_start_time = now
                elif (now - self.condition_start_time) >= self.debounce_s:
                    self.headlights_active = True
                    self.pending_state = None
                    return True, "HEADLIGHTS_ACTIVATED_DARK"
            else:
                self.pending_state = None
        else:
            if ambient_light_pct > self.off_threshold:
                if self.pending_state is not False:
                    self.pending_state = False
                    self.condition_start_time = now
                elif (now - self.condition_start_time) >= self.debounce_s:
                    self.headlights_active = False
                    self.pending_state = None
                    return False, "HEADLIGHTS_DEACTIVATED_BRIGHT"
            else:
                self.pending_state = None

        state_str = "ON" if self.headlights_active else "OFF"
        return self.headlights_active, f"HEADLIGHTS_{state_str}_STABLE"
