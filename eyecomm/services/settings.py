"""Runtime settings. Set EYECOMM_SERVER_URL to the caregiver/server PC address."""
from __future__ import annotations
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"
DEFAULTS = {
    "server_url": "http://127.0.0.1:8765",
    "patient_id": "patient-001",
    "poll_interval_seconds": 2.0
}

def load_settings():
    settings = dict(DEFAULTS)
    try:
        if SETTINGS_FILE.exists():
            settings.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    settings["server_url"] = os.getenv(
        "EYECOMM_SERVER_URL", settings["server_url"]
    ).rstrip("/")
    settings["patient_id"] = os.getenv(
        "EYECOMM_PATIENT_ID", settings["patient_id"]
    )
    return settings
