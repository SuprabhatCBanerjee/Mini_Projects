from typing import Dict, List
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, candidate_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.setdefault(candidate_id, []).append(websocket)

    def disconnect(self, candidate_id: str, websocket: WebSocket):
        self.active_connections[candidate_id].remove(websocket)

    async def broadcast(self, candidate_id: str, message: dict):
        for ws in self.active_connections.get(candidate_id, []):
            await ws.send_json(message)


manager = ConnectionManager()
