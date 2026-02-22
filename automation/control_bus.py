from __future__ import annotations

import asyncio
import json
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed
from websockets.server import WebSocketServerProtocol

from automation.schema import make_id, validate, wrap_cmd


class ControlBusServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 18792):
        self.host = host
        self.port = port
        self.clients: dict[str, WebSocketServerProtocol] = {}
        self._server: websockets.server.Serve | None = None
        self._ack_waiters: dict[str, asyncio.Future] = {}
        self._heartbeat_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await websockets.serve(self._handle_client, self.host, self.port)
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def send_cmd(self, client_id: str, cmd: dict[str, Any], timeout_s: float = 10.0) -> dict[str, Any]:
        ws = self.clients.get(client_id)
        if ws is None:
            raise RuntimeError(f"Client not connected: {client_id}")
        msg_id = cmd.get("id") or make_id()
        cmd["id"] = msg_id
        future = asyncio.get_running_loop().create_future()
        self._ack_waiters[msg_id] = future
        try:
            await asyncio.wait_for(ws.send(json.dumps(cmd)), timeout=min(5.0, timeout_s))
            return await asyncio.wait_for(future, timeout=timeout_s)
        except (ConnectionClosed, asyncio.TimeoutError) as exc:
            self.clients.pop(client_id, None)
            raise RuntimeError(f"Control bus send/ack failed for {client_id}: {exc}") from exc
        finally:
            self._ack_waiters.pop(msg_id, None)

    async def send_dom_ping(self, client_id: str) -> dict[str, Any]:
        return await self.send_cmd(client_id, wrap_cmd("dom.ping", {}))

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(5)
            for client_id in list(self.clients.keys()):
                ws = self.clients.get(client_id)
                if ws is None:
                    continue
                try:
                    await ws.send(
                        json.dumps(
                            {
                                "v": 1,
                                "type": "heartbeat",
                                "id": make_id(),
                                "name": "heartbeat",
                                "payload": {"ts": asyncio.get_running_loop().time()},
                            }
                        )
                    )
                except Exception:
                    self.clients.pop(client_id, None)

    async def _handle_client(self, websocket: WebSocketServerProtocol) -> None:
        client_id = ""
        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not validate(msg):
                    continue

                msg_type = msg.get("type")
                if msg_type == "hello":
                    client_id = str(msg.get("payload", {}).get("client_id") or f"ext_{id(websocket)}")
                    async with self._lock:
                        self.clients[client_id] = websocket
                    await websocket.send(
                        json.dumps(
                            {
                                "v": 1,
                                "type": "hello_ack",
                                "id": msg["id"],
                                "name": "hello_ack",
                                "payload": {"client_id": client_id},
                            }
                        )
                    )
                    continue

                if msg_type == "heartbeat_ack":
                    continue

                if msg_type == "cmd_ack":
                    waiter = self._ack_waiters.get(msg.get("id"))
                    if waiter is not None and not waiter.done():
                        waiter.set_result(msg)
                    continue

                if msg_type == "event":
                    # Events are consumed by the desktop app; no-op here.
                    continue
        except ConnectionClosed:
            pass
        finally:
            if client_id:
                self.clients.pop(client_id, None)
