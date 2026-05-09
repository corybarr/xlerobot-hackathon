"""WebSocket stream for MolmoAct2 inference logs."""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.inference import get_molmo_log_hub

router = APIRouter()


@router.websocket("/ws/inference/molmo/logs")
async def websocket_molmo_logs(websocket: WebSocket):
    await websocket.accept()
    hub = get_molmo_log_hub()
    hub.set_event_loop(asyncio.get_running_loop())
    q = hub.subscribe()
    try:
        while True:
            line = await q.get()
            await websocket.send_text(line)
    except WebSocketDisconnect:
        pass
    finally:
        hub.unsubscribe(q)
