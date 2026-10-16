"""Bounded out-of-order HTTP ingress, strictly ordered single-stream execution.

No torch imports; admission/ordering live on one asyncio event loop. Completed
HTTP bodies and CPU decoding may overlap the one GPU worker. Frames are NEVER
silently dropped/skipped, and stream controls require a drained pipeline.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import time
import uuid
from fastapi import HTTPException


@dataclass
class Job:
    seq: int
    started: float
    future: asyncio.Future
    state: str = "uploading"
    data: object = None
    ready_at: float = 0
    receive_ms: float = 0
    invalid: object = None


class OrderedPipeline:
    def __init__(self, processor, decoder, validator, *, capacity=4, gap_timeout=5,
                 upload_timeout=10):
        if not 1 <= capacity <= 10:
            raise ValueError("pipeline capacity must be 1..10")
        self.processor, self.decoder, self.validator = processor, decoder, validator
        self.capacity, self.gap_timeout, self.upload_timeout = capacity, gap_timeout, upload_timeout
        self.jobs = {}
        self.sid = None
        self.epoch = None
        self.next_seq = 0
        self.controlling = False
        self.failed = False
        self.stopping = False
        self.wake = asyncio.Event()
        self.accepted_total = self.completed_total = self.rejected_total = 0
        self.last_completed = None
        self.tasks = set()
        self.worker = asyncio.create_task(self._worker(), name="cast-ordered-pipeline")

    def bind(self, sid):
        assert not self.jobs, "cannot bind before drain"
        self.sid = sid
        self.epoch = uuid.uuid4().hex
        self.next_seq = 0
        self.failed = False
        self.last_completed = None
        return self.epoch

    def snapshot(self):
        return {"capacity": self.capacity, "pending": len(self.jobs),
                "next_seq": self.next_seq, "epoch": self.epoch,
                "frames": [{"seq": j.seq, "state": j.state,
                            "age_ms": round((time.perf_counter()-j.started)*1000, 2)}
                           for j in sorted(self.jobs.values(), key=lambda x:x.seq)],
                "failed": self.failed, "control_busy": self.controlling,
                "accepted_total": self.accepted_total, "completed_total": self.completed_total,
                "rejected_total": self.rejected_total, "last_completed_seq": self.last_completed,
                "gap_timeout_seconds": self.gap_timeout,
                "upload_timeout_seconds": self.upload_timeout}

    @asynccontextmanager
    async def control(self):
        if self.controlling or self.jobs:
            raise HTTPException(409, "pipeline not drained; stop sending and await all responses before reset/close")
        self.controlling = True
        try:
            yield
        finally:
            self.controlling = False

    async def handle(self, sid, seq, receive, *, epoch=None, timestamp_ms=None):
        self.validator(sid)
        if self.stopping or self.controlling:
            raise HTTPException(409, "stream control in progress")
        if sid != self.sid:
            raise HTTPException(409, "unknown pipeline session")
        if epoch is not None and epoch != self.epoch:
            raise HTTPException(409, "stale epoch; reset completed, discard old queued requests")
        if self.failed:
            raise HTTPException(409, "inference failed; drain then reset")
        if seq < self.next_seq or seq in self.jobs:
            raise HTTPException(409, f"duplicate/already consumed seq; next_seq={self.next_seq}")
        # Count uploading + decoding + ready + running, not just GPU jobs.
        if len(self.jobs) >= self.capacity or seq >= self.next_seq+self.capacity:
            self.rejected_total += 1
            raise HTTPException(429, f"pipeline window full; next_seq={self.next_seq}, capacity={self.capacity}; frame NOT accepted",
                                headers={"Retry-After":"1"})
        loop = asyncio.get_running_loop()
        job = Job(seq, time.perf_counter(), loop.create_future())
        # Retrieve exceptions even if HTTP client disconnects while a queued job
        # later fails. Admission is not undone by a lost HTTP response.
        job.future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        self.jobs[seq] = job
        self.accepted_total += 1
        task = asyncio.create_task(self._ingest(job, receive, timestamp_ms))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        self.wake.set()
        return await asyncio.shield(job.future)

    async def _ingest(self, job, receive, timestamp_ms):
        try:
            payload = await asyncio.wait_for(receive(), timeout=self.upload_timeout)
            job.receive_ms = (time.perf_counter()-job.started)*1000
            if job.invalid:
                raise job.invalid
            job.state = "decoding"
            decoded = await self.decoder(payload)
            if job.invalid:
                raise job.invalid
            job.data = (decoded, timestamp_ms)
            job.ready_at = time.perf_counter()
            job.state = "ready"
            self.wake.set()
        except asyncio.TimeoutError:
            self._invalidate_from(job.seq, HTTPException(408, "image upload timed out; frame not consumed"))
            self._finish_invalid(job)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                exc = HTTPException(503, "service shutting down")
            if not isinstance(exc, HTTPException):
                exc = HTTPException(400, "image upload/decode interrupted; frame not consumed")
            self._invalidate_from(job.seq, exc)
            self._finish_invalid(job)

    def _finish_invalid(self, job):
        if self.jobs.get(job.seq) is job:
            self.jobs.pop(job.seq)
        job.data = None
        if not job.future.done():
            job.future.set_exception(job.invalid or HTTPException(409, "frame cancelled"))
        self.wake.set()

    def _invalidate_from(self, seq, error):
        for job in list(self.jobs.values()):
            if job.seq >= seq and job.state != "running":
                job.invalid = error if job.seq == seq else HTTPException(
                    409, f"predecessor seq={seq} failed/missing; this frame not consumed, drain and retry from next_seq")
                if job.state == "ready":
                    self._finish_invalid(job)

    async def _worker(self):
        while not self.stopping or self.jobs:
            self.wake.clear()
            job = self.jobs.get(self.next_seq)
            if job and job.state == "ready":
                job.state = "running"
                gpu_start = time.perf_counter()
                decoded, timestamp_ms = job.data
                try:
                    result = await self.processor(self.sid, job.seq, decoded, timestamp_ms)
                    self.next_seq += 1
                    self.completed_total += 1
                    self.last_completed = job.seq
                    result.setdefault("timing_ms", {}).update({
                        "receive_ms": job.receive_ms,
                        "queue_wait_ms": (gpu_start-job.ready_at)*1000,
                        "server_pipeline_ms": (time.perf_counter()-job.started)*1000,
                    })
                    result["epoch"] = self.epoch
                    self.jobs.pop(job.seq)
                    job.data = None
                    job.future.set_result(result)
                except Exception as exc:
                    error = exc if isinstance(exc, HTTPException) else HTTPException(500, "pipeline inference failed; reset required")
                    self.failed = error.status_code >= 500
                    self.jobs.pop(job.seq, None)
                    job.data = None
                    job.future.set_exception(error)
                    self._invalidate_from(job.seq+1, HTTPException(409, "previous inference failed; frame not consumed"))
                continue
            # HTTP connections may arrive out of seq order. Wait a bounded time
            # for the missing predecessor, then reject rather than silently skip.
            if not job and self.jobs:
                first = min(self.jobs)
                if time.perf_counter()-min(j.started for j in self.jobs.values()) > self.gap_timeout:
                    self._invalidate_from(first, HTTPException(409, f"missing seq={self.next_seq}; gap timeout; no later frame consumed"))
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=.05)
            except asyncio.TimeoutError:
                pass

    async def close(self):
        self.stopping = True
        self.wake.set()
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks), return_exceptions=True)
        await self.worker
