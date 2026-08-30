"""
BeamNG UDP Telemetry Receiver & Screen Capture Node
===================================================
Author: Ramazan Ertugrul Aydogan
Affiliation: South-West University Neofit Rilski, Blagoevgrad, Bulgaria
"""

import socket
import threading
import time
from typing import Optional


class TelemetryReceiver:
    def __init__(self, port: int = 4444):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", self.port))
        self.sock.settimeout(0.5)
        self.latest_data = None
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(2048)
                self.latest_data = data
            except socket.timeout:
                pass

    def stop(self):
        self.running = False
        self.sock.close()
