"""Load and validate EyeComm's editable communication board configuration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parent / "communication_words.json"

DEFAULT_ITEMS = [
    {"label": "YES", "spoken_phrase": "Yes", "category": "Basic", "action": "speak"},
    {"label": "NO", "spoken_phrase": "No", "category": "Basic", "action": "speak"},
    {"label": "WATER", "spoken_phrase": "I would like some water", "category": "Needs", "action": "speak"},
    {"label": "HELP", "spoken_phrase": "Help, I need assistance", "category": "Emergency", "action": "speak"},
    {"label": "TIRED", "spoken_phrase": "I am tired", "category": "Physical", "action": "speak"},
    {"label": "CLEAR", "spoken_phrase": "", "category": "Controls", "action": "clear"},
]

def _validate_item(item: Any) -> dict:
    if not isinstance(item, dict):
        raise ValueError("Each board item must be a JSON object.")

    label = str(item.get("label", "")).strip().upper()
    phrase = str(item.get("spoken_phrase", "")).strip()
    category = str(item.get("category", "General")).strip() or "General"
    action = str(item.get("action", "speak")).strip().lower()

    if not label:
        raise ValueError("Every communication button needs a label.")
    if len(label) > 18:
        raise ValueError(f"Button label is too long: {label}")
    if action not in {"speak", "clear"}:
        raise ValueError(f"Unsupported action '{action}' for {label}.")
    if action == "speak" and not phrase:
        raise ValueError(f"Button '{label}' needs a spoken_phrase.")
    if action == "clear":
        phrase = ""

    return {
        "label": label,
        "spoken_phrase": phrase,
        "category": category,
        "action": action,
    }

def load_board(path: Path | str = CONFIG_PATH) -> list[dict]:
    """Load board items from JSON; fall back safely if file is missing/invalid."""
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list) or not items:
            raise ValueError("Board configuration must contain a non-empty 'items' list.")
        validated = [_validate_item(item) for item in items]
        if len(validated) > 6:
            raise ValueError("The current patient screen supports up to 6 buttons.")
        labels = [item["label"] for item in validated]
        if len(labels) != len(set(labels)):
            raise ValueError("Communication button labels must be unique.")
        return validated
    except FileNotFoundError:
        save_board(DEFAULT_ITEMS, config_path)
        return [dict(item) for item in DEFAULT_ITEMS]
    except Exception as exc:
        print(f"EyeComm board configuration warning: {exc}")
        print("Using the built-in default communication board.")
        return [dict(item) for item in DEFAULT_ITEMS]

def save_board(items: list[dict], path: Path | str = CONFIG_PATH) -> None:
    """Validate and save board items. Intended for caregiver customisation later."""
    validated = [_validate_item(item) for item in items]
    if not validated:
        raise ValueError("The communication board cannot be empty.")
    if len(validated) > 6:
        raise ValueError("The current patient screen supports up to 6 buttons.")
    labels = [item["label"] for item in validated]
    if len(labels) != len(set(labels)):
        raise ValueError("Communication button labels must be unique.")

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"items": validated}, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
