"""Queued, non-blocking offline text-to-speech service for EyeComm."""
from __future__ import annotations

import queue
import threading
from typing import Optional


class SpeechService:
    """Own one pyttsx3 engine on a worker thread and serialize speech requests."""

    def __init__(self, rate: int = 155, volume: float = 1.0,
                 voice_id: Optional[str] = None):
        self.rate = int(rate)
        self.volume = max(0.0, min(1.0, float(volume)))
        self.voice_id = voice_id
        self._queue = queue.Queue()
        self._ready = threading.Event()
        self._closed = False
        self._error = None
        self._thread = threading.Thread(
            target=self._worker, name="EyeComm-TTS", daemon=True
        )
        self._thread.start()
        self._ready.wait(timeout=12)

    @property
    def available(self) -> bool:
        return self._ready.is_set() and self._error is None and not self._closed

    @property
    def error(self):
        return self._error

    def _worker(self):
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            if self.voice_id:
                engine.setProperty("voice", self.voice_id)
            self._ready.set()

            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        break
                    engine.say(item)
                    engine.runAndWait()
                except Exception as exc:
                    self._error = f"{type(exc).__name__}: {exc}"
                finally:
                    self._queue.task_done()
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._ready.set()
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass

    def speak(self, text: str) -> str:
        text = " ".join(str(text).split())
        if not text:
            return "Nothing to speak."
        if self._closed:
            return "Speech service is closed."
        if not self._ready.wait(timeout=0.1):
            return "Speech engine is still starting."
        if self._error:
            return f"Speech unavailable: {self._error}"
        self._queue.put(text)
        return f"Queued speech: {text}"

    def close(self, timeout: float = 4.0):
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
