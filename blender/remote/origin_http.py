"""Origin HTTP policy: active requests != idle TCP connections.

Pinned/tested against Uvicorn 0.34.2 H11Protocol. No site-packages patch.
The guard is installed at the protocol's app boundary BEFORE FastAPI, so even
origin admission rejections carry correlation headers and structured audit logs.
"""
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import queue
import re
import threading
import time
from urllib.parse import parse_qs
import uuid

import h11
from uvicorn.protocols.http.h11_impl import H11Protocol

POLICY_VERSION = "active-request-v1"
ACTIVE_LIMIT = int(os.environ.get("CAST_HTTP_ACTIVE_LIMIT", "64"))
TCP_LIMIT = int(os.environ.get("CAST_HTTP_TCP_LIMIT", "256"))
HEALTH_RESERVE = 4
HEADER_TIMEOUT = 10
STATE = None
AUDIT = None


class AuditSink:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        os.close(fd)
        os.chmod(self.path, 0o600)
        self.handler = RotatingFileHandler(self.path, maxBytes=20*1024**2, backupCount=8, encoding="utf-8")
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.queue = queue.Queue(maxsize=4096)
        self.dropped = self.errors = 0
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="cast-http-audit")
        self.thread.start()

    def write(self, record):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while not self.stopping.is_set() or not self.queue.empty():
            try:
                data = self.queue.get(timeout=.5)
            except queue.Empty:
                continue
            try:
                raw = json.dumps(data, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
                rec = logging.LogRecord("cast.http", logging.INFO, "", 0, raw, (), None)
                if self.handler.shouldRollover(rec):
                    self.handler.doRollover()
                    os.chmod(self.path, 0o600)
                self.handler.stream.write(raw+"\n")
                self.handler.flush()
            except Exception:
                self.errors += 1
            finally:
                self.queue.task_done()
        self.handler.close()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=10)


def audit(event, **fields):
    data = {"ts_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "pid": os.getpid(), "event": event, "policy": POLICY_VERSION, **fields}
    if AUDIT is not None:
        AUDIT.write(data)
    else:
        logging.getLogger("uvicorn.error").info("CAST_HTTP %s", json.dumps(data, ensure_ascii=True))


def configure_audit(path):
    global AUDIT
    AUDIT = AuditSink(path)
    return AUDIT


def safe(value, limit=180):
    if isinstance(value, bytes):
        value = value.decode("latin1", "replace")
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in str(value))[:limit]


class OriginState:
    def __init__(self, server_state, active_limit, tcp_limit):
        if active_limit < 1 or tcp_limit <= active_limit+HEALTH_RESERVE:
            raise ValueError("TCP limit must exceed active limit + health reserve")
        self.server_state = server_state
        self.active_limit, self.tcp_limit = active_limit, tcp_limit
        self.active = self.health_active = 0
        self.rejected = self.connection_rejected = self.idle_evicted = self.header_expired = 0

    def snapshot(self):
        return {"policy": POLICY_VERSION, "active_request_limit": self.active_limit,
                "active_requests": self.active, "health_requests": self.health_active,
                "health_reserved_slots": HEALTH_RESERVE, "tcp_connection_limit": self.tcp_limit,
                "tcp_connections": len(self.server_state.connections),
                "uvicorn_tasks": len(self.server_state.tasks),
                "rejected_requests_total": self.rejected,
                "tcp_rejections_total": self.connection_rejected,
                "idle_evictions_total": self.idle_evicted, "header_timeouts_total": self.header_expired,
                "audit_dropped_total": AUDIT.dropped if AUDIT else 0,
                "audit_errors_total": AUDIT.errors if AUDIT else 0}


def snapshot():
    return STATE.snapshot() if STATE else {
        "policy": POLICY_VERSION, "active_request_limit": ACTIVE_LIMIT,
        "tcp_connection_limit": TCP_LIMIT, "active_requests": 0, "tcp_connections": 0}


class OriginH11Protocol(H11Protocol):
    def __init__(self, config, server_state, app_state, _loop=None):
        global STATE
        if config.limit_concurrency is not None:
            raise ValueError("Do NOT combine origin policy with Uvicorn --limit-concurrency")
        super().__init__(config, server_state, app_state, _loop)
        if not hasattr(server_state, "cast_origin"):
            server_state.cast_origin = OriginState(server_state, ACTIVE_LIMIT, TCP_LIMIT)
        self.origin = server_state.cast_origin
        STATE = self.origin
        self.real_app = self.app
        self.app = self.guarded_app
        self.connection_id = uuid.uuid4().hex[:16]
        self.created = time.monotonic()
        self.idle_since = None
        self.in_app = False
        self.header_timer = None

    def is_idle(self):
        return (not self.in_app and self.cycle is not None and self.cycle.response_complete
                and not self.transport.is_closing()
                and self.conn.our_state is h11.IDLE and self.conn.their_state is h11.IDLE
                and not self.conn.trailing_data[0])

    def connection_made(self, transport):
        # Reclaim ONLY fully completed idle keep-alive connections. Never evict
        # an incomplete header/body or a connection whose application is running.
        conns = self.server_state.connections
        if len(conns) >= self.origin.tcp_limit:
            idle = [p for p in tuple(conns) if isinstance(p, OriginH11Protocol) and p.is_idle()]
            if idle:
                victim = min(idle, key=lambda p:p.idle_since or p.created)
                victim.transport.close()
                conns.discard(victim)
                self.origin.idle_evicted += 1
                audit("idle_connection_evicted", connection_id=victim.connection_id,
                      peer=victim.client, **self.origin.snapshot())
            else:
                self.origin.connection_rejected += 1
                request_id = uuid.uuid4().hex
                audit("tcp_connection_rejected", request_id=request_id,
                      peer=transport.get_extra_info("peername"), reason="tcp-connection-cap",
                      rio_seq=None, seq=None, note="Rejected at accept before HTTP headers; request IDs unavailable",
                      **self.origin.snapshot())
                transport.write(("HTTP/1.1 503 Service Unavailable\r\n"
                    "Content-Type: text/plain; charset=utf-8\r\nContent-Length: 19\r\n"
                    "Connection: close\r\nRetry-After: 1\r\nX-CAST-Origin: cast\r\n"
                    f"X-CAST-Policy: {POLICY_VERSION}\r\n"
                    "X-CAST-Response-Layer: tcp-admission\r\nX-CAST-Reject-Reason: tcp-connection-cap\r\n"
                    f"X-CAST-Request-ID: {request_id}\r\n\r\nService Unavailable").encode("ascii"))
                transport.close()
                # Initialize base fields for the eventual connection_lost callback,
                # but never insert the rejected socket into the live set.
                super().connection_made(transport)
                conns.discard(self)
                return
        super().connection_made(transport)
        self._start_header_timer()

    def _start_header_timer(self):
        if self.header_timer is None:
            self.header_timer = self.loop.call_later(HEADER_TIMEOUT, self._header_timeout)

    def _cancel_header_timer(self):
        if self.header_timer is not None:
            self.header_timer.cancel()
            self.header_timer = None

    def _header_timeout(self):
        self.header_timer = None
        if not self.transport.is_closing():
            self.origin.header_expired += 1
            audit("header_timeout", connection_id=self.connection_id, peer=self.client,
                  reason="incomplete-http-headers", **self.origin.snapshot())
            self.transport.close()

    def data_received(self, data):
        if self.cycle is None or self.cycle.response_complete:
            self.idle_since = None
            self._start_header_timer()
        super().data_received(data)
        if self.cycle is not None and not self.cycle.response_complete:
            self._cancel_header_timer()

    def on_response_complete(self):
        super().on_response_complete()
        if self.transport.is_closing():
            return
        if self.cycle is not None and self.cycle.response_complete:
            if self.conn.trailing_data[0]:
                self._start_header_timer()
            else:
                self.idle_since = time.monotonic()

    def connection_lost(self, exc):
        self._cancel_header_timer()
        super().connection_lost(exc)

    async def guarded_app(self, scope, receive, send):
        request_id = uuid.uuid4().hex
        headers = dict(scope.get("headers", []))
        raw_path = scope.get("path", "")
        path = raw_path if re.fullmatch(r"/(?:healthz|openapi\.json|v1/sessions(?:/[0-9a-f]{32}(?:/(?:frames|status|reset))?)?)", raw_path) else "/unrecognized"
        try:
            query = parse_qs(scope.get("query_string", b"")[:4096].decode("ascii", "replace"), max_num_fields=20)
        except ValueError:
            query = {}
        seq = safe(query.get("seq", [""])[0], 24)
        epoch = safe(query.get("epoch", [""])[0], 40)
        rio = safe(headers.get(b"x-rio-seq", b""))
        is_health = path == "/healthz" and scope.get("method") == "GET"
        fields = {"request_id": request_id, "connection_id": self.connection_id,
                  "peer": self.client, "method": safe(scope.get("method", ""), 12),
                  "path": path, "seq": seq, "epoch": epoch, "rio_seq": rio,
                  "forwarded_for_untrusted": safe(headers.get(b"x-forwarded-for", b""), 250)}
        start = time.perf_counter()
        reason = None
        if is_health:
            if self.origin.health_active >= HEALTH_RESERVE:
                reason = "health-reserve-exhausted"
        elif self.origin.active >= self.origin.active_limit:
            reason = "active-request-limit"
        if reason:
            self.origin.rejected += 1
            audit("request_rejected", **fields, status=503, reason=reason, **self.origin.snapshot())
            await send({"type":"http.response.start", "status":503, "headers":[
                (b"content-type", b"text/plain; charset=utf-8"), (b"content-length", b"19"),
                (b"connection", b"close"), (b"retry-after", b"1"),
                (b"x-cast-origin", b"cast"), (b"x-cast-policy", POLICY_VERSION.encode()),
                (b"x-cast-response-layer", b"http-admission"),
                (b"x-cast-reject-reason", reason.encode()), (b"x-cast-request-id", request_id.encode()),
                *([(b"x-cast-rio-seq",rio.encode())] if rio else []),
            ]})
            await send({"type":"http.response.body", "body":b"Service Unavailable"})
            return
        self.in_app = True
        if is_health:
            self.origin.health_active += 1
        else:
            self.origin.active += 1
        admission = self.origin.snapshot()
        # Never log auth/cookies/body. Successful health probes don't flood logs.
        if not is_health:
            audit("request_start", **fields, **admission)
        status = None
        async def marked_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = dict(message)
                hs = list(message.get("headers", []))
                hs.extend([(b"x-cast-origin",b"cast"),(b"x-cast-policy",POLICY_VERSION.encode()),
                           (b"x-cast-request-id",request_id.encode()),(b"x-cast-response-layer",b"application")])
                if rio:
                    hs.append((b"x-cast-rio-seq",rio.encode()))
                message["headers"] = hs
            await send(message)
        try:
            await self.real_app(scope, receive, marked_send)
        finally:
            if is_health:
                self.origin.health_active -= 1
            else:
                self.origin.active -= 1
            self.in_app = False
            if not is_health or status != 200:
                audit("request_end", **fields, status=status, duration_ms=round((time.perf_counter()-start)*1000, 3),
                      disconnected=self.cycle.disconnected if self.cycle else None, **self.origin.snapshot())
