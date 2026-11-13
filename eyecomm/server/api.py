"""FastAPI local-network API for patient/caregiver synchronisation.

Prototype note: this initial LAN server has no authentication or TLS.
Do not expose it directly to the public internet.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from eyecomm.server.store import (
    init_db, get_board, save_board, validate_tiles,
    create_request, list_requests, update_status
)

app = FastAPI(title="EyeComm Local Caregiver Server", version="0.1.0")

class Tile(BaseModel):
    id: str
    label: str = Field(min_length=1, max_length=24)
    phrase: str = Field(default="", max_length=100)
    action: str

class BoardUpdate(BaseModel):
    tiles: list[Tile]

class PatientRequest(BaseModel):
    request_id: str
    patient_id: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=500)
    priority: Literal["normal", "urgent"] = "normal"

class StatusUpdate(BaseModel):
    caregiver: str = Field(default="Caregiver", max_length=80)

@app.on_event("startup")
def startup():
    init_db()

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "EyeComm"}

@app.get("/api/board")
def read_board():
    return get_board()

@app.put("/api/board")
def update_board(body: BoardUpdate):
    tiles = [tile.model_dump() if hasattr(tile, "model_dump") else tile.dict()
             for tile in body.tiles]
    if not validate_tiles(tiles):
        raise HTTPException(status_code=400, detail="Invalid board configuration.")
    save_board(tiles)
    return {"saved": True, "tiles": tiles}

@app.post("/api/requests")
def post_request(body: PatientRequest):
    data = body.model_dump() if hasattr(body, "model_dump") else body.dict()
    result = create_request(data)
    return {"accepted": True, "request": result}

@app.get("/api/requests")
def read_requests(limit: int = 100):
    return {"requests": list_requests(max(1, min(limit, 200)))}

@app.post("/api/requests/{request_id}/ack")
def acknowledge(request_id: str, body: StatusUpdate):
    result = update_status(request_id, "ACKNOWLEDGED", body.caregiver)
    if not result:
        raise HTTPException(status_code=404, detail="Request not found.")
    return result

@app.post("/api/requests/{request_id}/resolve")
def resolve(request_id: str):
    result = update_status(request_id, "RESOLVED")
    if not result:
        raise HTTPException(status_code=404, detail="Request not found.")
    return result
