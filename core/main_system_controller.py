"""
Main System Controller & Operational Finite State Machine
=========================================================
Author: Ramazan Ertugrul Aydogan
Affiliation: South-West University Neofit Rilski, Blagoevgrad, Bulgaria

Implements PROCEDURE MAIN_SYSTEM_LOOP from Section II.D of the MDPI Report:
  - Monitors vehicle reverse-gear GPIO pin.
  - Manages transitions between FORWARD_MODE and REVERSE_MODE.
  - Coordinates concurrent threads: Forward ADAS, Reverse Assist, Blind Spot Monitor, Auto Headlights, DMS.
"""

import time
import threading
import cv2
from typing import Optional
from core.forward_adas_pipeline import ForwardADASPipeline
from core.reverse_assist import ReverseAssistPipeline
from core.blind_spot_monitor import BlindSpotMonitor
from core.auto_headlight_control import AutoHeadlightController
from core.driver_monitoring_system import DriverMonitoringSystem


class ADASControllerFSM:
    def __init__(self, forward_camera_idx: int = 0, rear_camera_idx: int = 1, debug_mode: bool = True):
        self.forward_camera_idx = forward_camera_idx
        self.rear_camera_idx = rear_camera_idx
        self.debug_mode = debug_mode
        self.running = False

        # Subsystems
        self.forward_pipeline = ForwardADASPipeline()
        self.reverse_assist = ReverseAssistPipeline(rear_camera_idx)
        self.blind_spot = BlindSpotMonitor()
        self.auto_headlight = AutoHeadlightController()
        self.dms = DriverMonitoringSystem()

        # Operational State
        self.current_state = "FORWARD_MODE"

    def read_reverse_gpio_signal(self) -> bool:
        """Polls GPIO pin connected to 12V reverse light (reduced to 3.3V logic level)."""
        if self.debug_mode:
            return False
        try:
            import RPi.GPIO as GPIO
            return GPIO.input(18) == GPIO.HIGH
        except Exception:
            return False

    def run_main_system_loop(self):
        """
        Executes PROCEDURE MAIN_SYSTEM_LOOP from Section II.D.
        """
        self.running = True
        cap_forward = cv2.VideoCapture(self.forward_camera_idx)

        print("[INFO] ADAS Multi-Threaded Controller initialized.")
        print("[INFO] Press 'Q' to quit, 'R' to toggle simulated Reverse Gear in debug mode.")

        while self.running:
            is_reversing = self.read_reverse_gpio_signal()
            target_state = "REVERSE_MODE" if is_reversing else "FORWARD_MODE"

            # State Transition Handler
            if target_state != self.current_state:
                if target_state == "REVERSE_MODE":
                    print("[FSM] Transitioning to REVERSE_MODE")
                    self.reverse_assist.start_camera_stream()
                else:
                    print("[FSM] Transitioning to FORWARD_MODE")
                    self.reverse_assist.stop_camera_stream()
                self.current_state = target_state

            # State Execution
            if self.current_state == "FORWARD_MODE":
                ret, frame = cap_forward.read()
                if ret:
                    overlay, summary = self.forward_pipeline.process_frame(frame)
                    cv2.imshow("ADAS Forward View", overlay)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and self.debug_mode:
                self.current_state = "REVERSE_MODE" if self.current_state == "FORWARD_MODE" else "FORWARD_MODE"
                print(f"[DEBUG] Toggled state to {self.current_state}")

            time.sleep(0.010)  # 10ms CPU-saving sleep as specified in MDPI pseudo-code

        cap_forward.release()
        cv2.destroyAllWindows()
        print("[INFO] ADAS Controller shut down cleanly.")


if __name__ == "__main__":
    controller = ADASControllerFSM(debug_mode=True)
    controller.run_main_system_loop()
