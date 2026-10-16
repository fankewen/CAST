"""Reliable length-framed localhost transport for camera JPEGs."""
from __future__ import annotations

import socket
import struct
import threading
import time


FRAME_HEADER = struct.Struct("<dI")
MAX_FRAME_BYTES = 16 * 1024 * 1024


class FrameStreamSender:
    """可靠的长度前缀本地传输。

    重要：建立连接一律放到后台线程，建连不会阻塞 send()；发送本身受 socket 超时约束。
    某些环境（如装了企业安全软件）会把发往未监听端口的 SYN 静默丢弃，
    此时 TCP connect 只能等到超时（实测 200ms）。若每帧都重连，采集帧率
    会被拖垮到 5fps 左右；这里改成异步连接 + 丢帧，接收端起来后自动接上。
    """

    def __init__(self, host: str, port: int, connect_retry: float = 0.5) -> None:
        self.address = (host, int(port))
        self.sock: socket.socket | None = None
        self.lock = threading.Lock()
        self.connect_retry = float(connect_retry)
        self._connecting = False
        self._closed = False
        self._last_attempt = 0.0

    def _connect_async(self) -> None:
        """后台建连接；调用方需已持有 self.lock。"""
        def worker() -> None:
            try:
                sock = socket.create_connection(self.address, timeout=1.0)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(1.0)
            except OSError:
                sock = None
            with self.lock:
                self._connecting = False
                if sock is not None:
                    if self.sock is None and not self._closed:
                        self.sock = sock
                    else:
                        try:
                            sock.close()
                        except OSError:
                            pass

        self._connecting = True
        self._last_attempt = time.monotonic()
        threading.Thread(target=worker, daemon=True, name="frame-connect").start()

    def send(self, timestamp_ms: float, payload: bytes) -> bool:
        if not 0 < len(payload) <= MAX_FRAME_BYTES:
            raise ValueError(f"invalid local frame size: {len(payload)}")
        packet = FRAME_HEADER.pack(float(timestamp_ms), len(payload)) + payload
        with self.lock:
            if self._closed:
                return False
            if self.sock is None:
                # 接收端不在：本帧直接丢弃，后台去连，不阻塞采集
                if (not self._connecting
                        and time.monotonic() - self._last_attempt >= self.connect_retry):
                    self._connect_async()
                return False
            try:
                self.sock.sendall(packet)
                return True
            except OSError:
                self.sock.close()
                self.sock = None
                return False

    def close(self) -> None:
        with self.lock:
            self._closed = True
            sock, self.sock = self.sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def receive_exact(connection: socket.socket, size: int, stop_event) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size and not stop_event.is_set():
        try:
            chunk = connection.recv(size - len(chunks))
        except socket.timeout:
            continue
        except OSError:
            # Capture may exit in the middle of a frame. Discard the partial
            # payload so the transport can accept the next camera connection.
            return None
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks) if len(chunks) == size else None
