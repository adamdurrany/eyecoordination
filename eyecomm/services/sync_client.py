"""Background HTTP sync client for the patient application."""
from __future__ import annotations

import queue
import threading
import time
import uuid
import requests

from eyecomm.services.settings import load_settings

DEFAULT_TILES = [
    {"id": "YES", "label": "YES", "phrase": "Yes", "action": "WORD"},
    {"id": "NO", "label": "NO", "phrase": "No", "action": "WORD"},
    {"id": "WATER", "label": "WATER", "phrase": "I need water", "action": "WORD"},
    {"id": "HELP", "label": "HELP", "phrase": "I need help", "action": "URGENT"},
    {"id": "TIRED", "label": "TIRED", "phrase": "I am tired", "action": "WORD"},
    {"id": "SPEAK", "label": "SPEAK / SEND", "phrase": "", "action": "SEND"},
    {"id": "DELETE", "label": "DELETE", "phrase": "", "action": "DELETE"},
    {"id": "CLEAR", "label": "CLEAR", "phrase": "", "action": "CLEAR"},
]

class SyncClient:
    """Poll caregiver-configured board and send patient requests off the camera loop."""

    def __init__(self):
        self.settings = load_settings()
        self.base_url = self.settings["server_url"]
        self.patient_id = self.settings["patient_id"]
        self.interval = max(2.0, float(self.settings.get("poll_interval_seconds", 2.0)))
        self._stop = threading.Event()
        self._outbox = queue.Queue()
        self._board_thread = None
        self._send_thread = None
        self._board_callback = None
        self.last_error = None

    def get_board(self, fallback=True):
        try:
            response = requests.get(f"{self.base_url}/api/board", timeout=1.5)
            response.raise_for_status()
            payload = response.json()
            tiles = payload.get("tiles", [])
            if len(tiles) == 8:
                self.last_error = None
                return tiles
        except Exception as exc:
            self.last_error = str(exc)
        return DEFAULT_TILES if fallback else None

    def start(self, board_callback=None):
        self._board_callback = board_callback
        self._board_thread = threading.Thread(
            target=self._poll_board, name="EyeComm-Board-Sync", daemon=True
        )
        self._send_thread = threading.Thread(
            target=self._send_worker, name="EyeComm-Request-Sender", daemon=True
        )
        self._board_thread.start()
        self._send_thread.start()

    def _poll_board(self):
        while not self._stop.is_set():
            tiles = self.get_board(fallback=False)
            if tiles is not None and self._board_callback:
                try:
                    self._board_callback(tiles)
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def send_request(self, message, priority="normal"):
        payload = {
            "request_id": str(uuid.uuid4()),
            "patient_id": self.patient_id,
            "message": str(message).strip(),
            "priority": priority,
        }
        if payload["message"]:
            self._outbox.put(payload)
            return "Request queued for caregiver."
        return "No message to send."

    def _send_worker(self):
        while not self._stop.is_set():
            try:
                payload = self._outbox.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                response = requests.post(
                    f"{self.base_url}/api/requests",
                    json=payload,
                    timeout=3.0
                )
                response.raise_for_status()
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
                # Retain the request for a later retry while the client is running.
                if not self._stop.is_set():
                    self._stop.wait(2.0)
                    self._outbox.put(payload)
            finally:
                self._outbox.task_done()

    def close(self):
        self._stop.set()
