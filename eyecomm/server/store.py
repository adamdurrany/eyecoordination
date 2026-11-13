"""Small SQLite store for the local EyeComm caregiver prototype."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DB_PATH = DATA / "eyecomm.sqlite3"
BOARD_PATH = DATA / "board.json"

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
FIXED_ACTIONS = {
    "YES": "WORD", "NO": "WORD", "WATER": "WORD", "HELP": "URGENT",
    "TIRED": "WORD", "SPEAK": "SEND", "DELETE": "DELETE", "CLEAR": "CLEAR"
}

def init_db():
    DATA.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                message TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                created_at TEXT NOT NULL,
                acknowledged_by TEXT,
                acknowledged_at TEXT,
                resolved_at TEXT
            )
        """)
    if not BOARD_PATH.exists():
        save_board(DEFAULT_TILES)

def get_board():
    init_db()
    try:
        data = json.loads(BOARD_PATH.read_text(encoding="utf-8"))
        if validate_tiles(data.get("tiles")):
            return data
    except (OSError, ValueError, TypeError):
        pass
    return {"version": 1, "tiles": DEFAULT_TILES}

def validate_tiles(tiles):
    if not isinstance(tiles, list) or len(tiles) != 8:
        return False
    ids = [t.get("id") for t in tiles if isinstance(t, dict)]
    if set(ids) != set(FIXED_ACTIONS) or len(ids) != 8:
        return False
    for tile in tiles:
        if not isinstance(tile.get("label"), str) or not tile["label"].strip():
            return False
        if len(tile["label"]) > 24:
            return False
        if not isinstance(tile.get("phrase", ""), str) or len(tile.get("phrase", "")) > 100:
            return False
        if tile.get("action") != FIXED_ACTIONS[tile["id"]]:
            return False
    return True

def save_board(tiles):
    if not validate_tiles(tiles):
        raise ValueError("Board must contain the eight original action IDs with valid labels.")
    DATA.mkdir(parents=True, exist_ok=True)
    temp = BOARD_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps({"version": 1, "tiles": tiles}, indent=2), encoding="utf-8")
    temp.replace(BOARD_PATH)

def create_request(payload):
    init_db()
    created = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            INSERT OR IGNORE INTO requests
            (request_id, patient_id, message, priority, status, created_at)
            VALUES (?, ?, ?, ?, 'PENDING', ?)
        """, (
            payload["request_id"], payload["patient_id"],
            payload["message"], payload["priority"], created
        ))
    return get_request(payload["request_id"])

def list_requests(limit=100):
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM requests ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]

def get_request(request_id):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT * FROM requests WHERE request_id=?", (request_id,)
        ).fetchone()
    return dict(row) if row else None

def update_status(request_id, status, caregiver="Caregiver"):
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as con:
        if status == "ACKNOWLEDGED":
            con.execute("""
                UPDATE requests SET status=?, acknowledged_by=?, acknowledged_at=?
                WHERE request_id=?
            """, (status, caregiver, now, request_id))
        elif status == "RESOLVED":
            con.execute("""
                UPDATE requests SET status=?, resolved_at=? WHERE request_id=?
            """, (status, now, request_id))
    return get_request(request_id)
