from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import get_settings
from .inference import ISLInferenceService
from .utils.postprocessing import now_ms
from .utils.preprocessing import decode_image_bytes, decode_image_payload

settings = get_settings()
service = ISLInferenceService(settings)
project_root = Path(__file__).resolve().parents[1]
dataset_roots = [project_root / "archive" / split for split in ("Test", "Validation", "Train")]

app = FastAPI(
    title="ISL YOLO Realtime API",
    version="0.1.0",
    description="FastAPI backend for realtime Indian Sign Language recognition.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Base64PredictRequest(BaseModel):
    image: str = Field(..., description="data:image/jpeg;base64,... or raw base64")
    session_id: str = Field(default="rest-default")
    ts: Optional[int] = None


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "ISL YOLO Realtime API",
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        **service.model_status(),
    }


def _safe_label(label: str) -> str:
    cleaned = label.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Label cannot be empty")
    if any(ch in cleaned for ch in ("/", "\\", "..")):
        raise HTTPException(status_code=400, detail="Invalid label")
    return cleaned


def _resolve_sign_sample_path(label: str) -> Optional[Path]:
    candidates = [label, label.upper(), label.lower()]
    for root in dataset_roots:
        for candidate in candidates:
            class_dir = root / candidate
            if not class_dir.exists() or not class_dir.is_dir():
                continue
            images = sorted(
                p
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
            if images:
                return images[0]
    return None


@app.get("/sign-sample/{label}")
def get_sign_sample(label: str) -> FileResponse:
    safe_label = _safe_label(label)
    image_path = _resolve_sign_sample_path(safe_label)
    if image_path is None:
        raise HTTPException(status_code=404, detail=f"No sign sample found for label '{safe_label}'")
    return FileResponse(path=image_path)


@app.post("/predict")
async def predict_file(file: UploadFile = File(...), session_id: str = "rest-default") -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected image upload")

    data = await file.read()
    try:
        frame = decode_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = await run_in_threadpool(service.process_frame, frame, session_id, None)
    return {"type": "prediction", **result}


@app.post("/predict/base64")
async def predict_base64(payload: Base64PredictRequest) -> dict[str, Any]:
    try:
        frame = decode_image_payload(payload.image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = await run_in_threadpool(service.process_frame, frame, payload.session_id, payload.ts)
    return {"type": "prediction", **result}


@app.websocket("/ws")
async def websocket_inference(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(uuid4())

    await websocket.send_json(
        {
            "type": "status",
            "status": "connected",
            "session_id": session_id,
            **service.model_status(),
        }
    )

    try:
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")

            if msg_type == "frame":
                image_payload = message.get("image")
                if not isinstance(image_payload, str):
                    await websocket.send_json({"type": "error", "error": "Missing or invalid 'image' field"})
                    continue

                try:
                    frame = decode_image_payload(image_payload)
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "error": str(exc)})
                    continue

                client_ts = message.get("ts")
                result = await run_in_threadpool(service.process_frame, frame, session_id, client_ts)

                network_latency_ms: Optional[int] = None
                if isinstance(client_ts, (int, float)):
                    network_latency_ms = max(0, now_ms() - int(client_ts))

                await websocket.send_json(
                    {
                        "type": "prediction",
                        "network_latency_ms": network_latency_ms,
                        **result,
                    }
                )
                continue

            if msg_type == "control":
                action = str(message.get("action", "")).strip()
                control_response = await run_in_threadpool(service.handle_control, session_id, action)
                await websocket.send_json({"type": "control", "action": action, **control_response})
                continue

            if msg_type == "ping":
                await websocket.send_json({"type": "pong", "ts": now_ms()})
                continue

            await websocket.send_json({"type": "error", "error": f"Unsupported message type '{msg_type}'"})

    except WebSocketDisconnect:
        service.remove_session(session_id)
    except Exception as exc:  # noqa: BLE001
        service.remove_session(session_id)
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        finally:
            await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host=settings.host, port=settings.port, reload=True)
