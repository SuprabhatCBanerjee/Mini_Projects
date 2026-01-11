from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.realtime.ws_manager import manager

from app.core.logger import get_logger
logger = get_logger("WS")


router = APIRouter()

@router.websocket("/ws/{candidate_id}")
async def websocket_endpoint(websocket: WebSocket, candidate_id: str):
    logger.info(f"[WS] Client connected for {candidate_id}")
    await manager.connect(candidate_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected for {candidate_id}")
        manager.disconnect(candidate_id, websocket)
