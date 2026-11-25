"""Reliable repeated text-to-speech queue for EyeComm on Windows.

Uses pyttsx3's external event-loop mode. The worker owns the SAPI5 engine
and pumps its events while each utterance is active, then accepts the next
queued word without calling runAndWait() repeatedly.
"""
from __future__ import annotations

import ctypes
import os
import queue
import threading
import time
from typing import Optional


class SpeechService:
    def __init__(self, rate: int = 160, volume: float = 1.0):
        self.rate = int(rate)
        self.volume = max(0.0, min(1.0, float(volume)))
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._ready = threading.Event()
        self._closed = False
        self._error: Optional[str] = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._worker,
            name="EyeComm-Speech",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=15)
        if not self._ready.is_set():
            self._error = "Speech engine initialization timed out."

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def available(self) -> bool:
        return self._ready.is_set() and self._error is None and not self._closed

    def speak(self, text: str) -> bool:
        cleaned = " ".join(str(text).split())
        if not cleaned or self._closed or self._error:
            return False
        if not self._ready.is_set():
            return False
        self._queue.put(cleaned)
        return True

    def _worker(self):
        engine = None
        com_initialized = False
        loop_started = False

        try:
            if os.name == "nt":
                ole32 = ctypes.windll.ole32
                result = ole32.CoInitializeEx(None, 0x2)  # STA
                if result >= 0:
                    com_initialized = True
                elif result != -2147417850:  # RPC_E_CHANGED_MODE
                    raise OSError(f"COM initialization failed: HRESULT {result}")

            import pyttsx3
            engine = pyttsx3.init(driverName="sapi5" if os.name == "nt" else None)
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)

            voices = engine.getProperty("voices")
            if not voices:
                raise RuntimeError("No text-to-speech voices are installed.")

            # External event loop: this thread explicitly pumps events for
            # every utterance instead of re-entering runAndWait repeatedly.
            engine.startLoop(False)
            loop_started = True

        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._ready.set()

        if engine is None:
            if com_initialized:
                ctypes.windll.ole32.CoUninitialize()
            return

        try:
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        break

                    engine.say(item)

                    # Pump the SAPI5 event loop until this word has finished.
                    while engine.isBusy():
                        engine.iterate()
                        time.sleep(0.01)

                    # A final pump delivers completion/word callbacks.
                    engine.iterate()

                except Exception as exc:
                    self._error = f"{type(exc).__name__}: {exc}"
                    # Reinitialization is safer than continuing with a
                    # potentially broken SAPI5 driver state.
                    break
                finally:
                    self._queue.task_done()

        finally:
            if loop_started:
                try:
                    engine.endLoop()
                except Exception:
                    pass
            try:
                engine.stop()
            except Exception:
                pass
            if com_initialized:
                ctypes.windll.ole32.CoUninitialize()

    def close(self, timeout: float = 5.0):
        if self._closed:
            return
        self._closed = True
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=timeout)
