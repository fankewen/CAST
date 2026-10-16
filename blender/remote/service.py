"""One GPU / one exclusive causal stream. Images in, rig-local rot6d out."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hmac
import io
import json
import logging
import os
from pathlib import Path
import time
import uuid
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict
from settings import SETTINGS
import torch
from ordered_pipeline import OrderedPipeline
from origin_http import snapshot as http_policy_snapshot

HOME = SETTINGS.state_dir
HOME.mkdir(parents=True, exist_ok=True)
(HOME / "runtime").mkdir(exist_ok=True)
RELEASE = SETTINGS.assets_dir
COORDS = "opencv_right_handed_ydown_zforward_meters"
API_KEY = os.environ.get("CAST_API_KEY", "")
MAX_BODY = 8 * 1024 * 1024
MAX_PIXELS = 1920 * 1080
SESSION_TTL = 120
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
LOG = logging.getLogger("cast")
PIPELINE_CAPACITY = int(os.environ.get("CAST_PIPELINE_CAPACITY", "10"))


def decode_image(payload):
    begin = time.perf_counter()
    try:
        with Image.open(io.BytesIO(payload)) as img:
            if img.format not in ("JPEG", "PNG") or img.width * img.height > MAX_PIXELS:
                raise ValueError("only JPEG/PNG up to 1920*1080 pixels accepted")
            rgb = np.asarray(img.convert("RGB")).copy()
        return rgb, (time.perf_counter()-begin)*1000
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, str(exc)) from exc


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rig: dict
    target_mode: Literal['human','generic'] = 'human'


def validate_rig(data):
    """Only JSON numeric data crosses the trust boundary; no uploaded pickle."""
    if not isinstance(data, dict):
        raise ValueError("rig must be an object")
    j = data.get("joints")
    if type(j) is not int or not 1 <= j <= 150:
        raise ValueError("joints must be an integer in [1,150]")
    names = data.get("joint_names")
    if not isinstance(names, list) or len(names) != j or any(
        not isinstance(n, str) or not 1 <= len(n) <= 256 for n in names
    ) or len(set(names)) != j:
        raise ValueError("joint_names must be unique strings, length J")
    parents = data.get("parents")
    if not isinstance(parents, list) or len(parents) != j or any(
        type(p) is not int or not -1 <= p < i for i, p in enumerate(parents)
    ):
        raise ValueError("parents must be topologically ordered: -1 or index < joint")
    if parents[0] != -1 or any(p < 0 for p in parents[1:]):
        raise ValueError("one root required at index 0; all other parents must be >=0")
    arrays = {}
    for name, shape in [("rest_translations", (j, 3)), ("rest_rotations_quat", (j, 4))]:
        arr = np.asarray(data.get(name), dtype=np.float32)
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite, shape {shape}")
        arrays[name] = arr
    norms = np.linalg.norm(arrays["rest_rotations_quat"], axis=-1)
    if not np.allclose(norms, 1, atol=0.01):
        raise ValueError("rest_rotations_quat must be normalized xyzw quaternions")
    scale = float(data.get("metric_scale", 0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("metric_scale must be positive")
    if data.get("coordinate_system") != COORDS or data.get("units") != "meters":
        raise ValueError(f"coordinate_system must be {COORDS}; units must be meters")
    return dict(
        rig_id=str(data.get("rig_id", "remote"))[:256], joints=j,
        joint_names=np.asarray(names), parents=np.asarray(parents, dtype=np.int32),
        **arrays, metric_scale=np.float32(scale), coordinate_system=COORDS,
        units="meters", skeleton_hash="validated-remote",
    )


class Runtime:
    def __init__(self):
        self.session = None
        self.last_seen = 0
        self.next_seq = 0
        self.failed = False
        self.version = os.environ.get("CAST_BACKBONE", "dinov3")
        if self.version != "dinov3":
            raise ValueError("This distribution provides DINOv3 configs; add and validate another backbone explicitly")
        # No private/demo character is distributed. Client must register its own rig.
        self.rig_key = None
        self.history = None
        self.history_stream = None
        self.target_mode = 'human'
        self.reference_bound = False
        self.last_motion = None
        self.model_info = None
        self.full_graph = None
        self.full_graph_enabled = os.environ.get('CAST_FULL_STEP_GRAPH','0') == '1'
        self.pinned_frame = None

    @torch.inference_mode()
    def load(self):
        from inference.deploy_realtime_video2motion import RealtimeVideo2Motion
        from human_tracking import PromptedBoxSegmenter, HumanTracker
        from person_detector import PersonDetector
        from models.dinov3_infer import build_transform
        from model_selection import resolve_model
        torch.set_num_threads(4)
        selected=resolve_model(SETTINGS.assets_dir,self.version)
        self.model_info=selected['public']
        # Keep compilation artifacts scoped to the selected checkpoint shape.
        os.environ['TORCHINDUCTOR_CACHE_DIR']=str(Path(selected['cache_dir'])/'inductor')
        os.environ['TRITON_CACHE_DIR']=str(Path(selected['cache_dir'])/'triton')
        LOG.warning('MODEL_SELECTED %s',json.dumps(self.model_info,sort_keys=True))
        self.model = RealtimeVideo2Motion(
            config=selected['config'],
            checkpoint=selected['checkpoint'],
            device="cuda:0", amp_dtype="bf16", cache_length=81,
            compile_cache_dir=selected['cache_dir'],
            checkpoint_weights_only=True,
            use_cudagraphs=(not self.full_graph_enabled and os.environ.get("CAST_CUDAGRAPHS", "0") == "1"),
        )
        self.segmenter = PromptedBoxSegmenter(
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            str(SETTINGS.sam_checkpoint),
            image_size=512, device="cuda:0", amp_dtype="bf16",
        )
        self.transform = build_transform(self.model.image_size)
        if self.full_graph_enabled:
            self.pinned_frame=torch.empty((3,self.model.image_size,self.model.image_size),
                                          dtype=torch.float32,device='cpu',pin_memory=True)
        # Skeleton precomputation and graph capture happen at POST /v1/sessions.
        self.segmenter.box(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.segmenter.reset()
        self.person_detector=PersonDetector(weights=SETTINGS.person_checkpoint,device='cuda:0')
        self.human_tracker=HumanTracker(self.person_detector,self.segmenter)
        self.person_detector.detect(np.zeros((720,1280,3),dtype=np.uint8))
        from live_history import LiveHistory
        self.history = LiveHistory(HOME)
        LOG.warning("READY backbone=%s model=%s joints=%d", self.version, self.model_info['id'], self.model.joint_count or 0)

    @torch.inference_mode()
    def prepare_rig(self, rig):
        validated = validate_rig(rig)
        key = json.dumps(rig, sort_keys=True)
        if key != self.rig_key:
            self.full_graph = None
            self.rig_key = None
            path = HOME / "runtime/validated_static.npy"
            np.save(path, validated, allow_pickle=True)
            self.model.prepare(str(path))
            self.model.compile_streaming()
            if self.full_graph_enabled:
                from full_step_graph import FullStepGraph
                self.full_graph=FullStepGraph(self.model,verify=True)
                LOG.warning('FULL_STEP_GRAPH_READY %s',json.dumps(self.full_graph.info(),sort_keys=True))
            self.rig_key = key
        self.names = list(rig["joint_names"])
        self.rig_id = str(rig.get("rig_id", "remote"))[:256]
        from scipy.spatial.transform import Rotation
        rest=Rotation.from_quat(validated['rest_rotations_quat']).as_matrix()
        self.rest_motion=np.concatenate([rest[:,:,0],rest[:,:,1]],axis=-1).astype(np.float32)

    def check_session(self, sid):
        if self.session != sid or time.monotonic() - self.last_seen > SESSION_TTL:
            raise HTTPException(409, "session missing/expired; create a new session")
        if self.failed:
            raise HTTPException(409, "stream failed; reset or recreate session")
        self.last_seen = time.monotonic()

    def acceleration_info(self):
        if self.full_graph is not None:
            return self.full_graph.info()
        return {'engine':'compiled_streaming','enabled':False,
                'full_graph_requested':self.full_graph_enabled}

    @torch.inference_mode()
    def create_session(self, rig, target_mode='human'):
        if self.session and time.monotonic() - self.last_seen <= SESSION_TTL:
            raise HTTPException(409, "GPU stream occupied; close session or wait for idle TTL")
        self.session = None
        self.prepare_rig(rig)
        self.model.reset_streaming()
        self.segmenter.reset()
        self.human_tracker.reset()
        self.target_mode=target_mode
        self.reference_bound=False
        self.last_motion=None
        self.session = uuid.uuid4().hex
        self.last_seen = time.monotonic()
        self.next_seq = 0
        self.failed = False
        self.history_stream = self.history.new_stream()
        return {
            "session_id": self.session, "joint_names": self.names,
            "joints": len(self.names), "next_seq": 0, "idle_ttl_seconds": SESSION_TTL,
            "backbone": self.version, "representation": "rot6d_columns",
            "model": dict(self.model_info),
            "acceleration": self.acceleration_info(),
            "coordinate_system": COORDS, "root_translation_predicted": False,
            "target_mode": self.target_mode,
            "person_detector": 'ssdlite320_mobilenet_v3_large' if target_mode=='human' else None,
            "no_person_behavior": 'motion_valid=false; hold last/rest pose; seq consumed',
        }

    @torch.inference_mode()
    def reset(self, sid):
        # Reset is allowed after a failed frame, but not after expiration.
        if sid != self.session or time.monotonic() - self.last_seen > SESSION_TTL:
            raise HTTPException(409, "session missing/expired")
        self.model.reset_streaming()
        self.segmenter.reset()
        self.human_tracker.reset()
        self.reference_bound=False
        self.last_motion=None
        self.next_seq = 0
        self.failed = False
        self.history_stream = self.history.new_stream()
        self.last_seen = time.monotonic()
        return {"next_seq": 0}

    def close(self, sid):
        if sid != self.session:
            raise HTTPException(409, "unknown session")
        self.session = None
        return {"closed": True}

    @torch.inference_mode()
    def frame(self, sid, seq, decoded_image, timestamp_ms):
        self.check_session(sid)
        if seq != self.next_seq:
            raise HTTPException(409, f"expected seq={self.next_seq}; no frame consumed")
        begin = time.perf_counter()
        rgb, decode_ms = decoded_image
        decoded = begin
        try:
            if self.target_mode=='human':
                box,tracking=self.human_tracker.box(rgb)
            else:
                self.segmenter.prompt_override=None
                box=list(self.segmenter.box(rgb))
                tracking={'mode':'generic','status':'unverified_subject','motion_valid':True,
                          'person_found':None,'reacquired':False,'detector_bbox':None,
                          'detector_score':None,'crop_source':'legacy_sam','detector_ms':0.,
                          'sam_ms':self.segmenter.last_seconds*1000}
            if box is not None:
                cropped = self.transform(Image.fromarray(rgb).crop(box))
                if self.pinned_frame is not None:
                    # One GPU worker owns this buffer; previous output was synced
                    # to CPU before reuse. No numerical preprocessing change.
                    self.pinned_frame.copy_(cropped)
                    cropped=self.pinned_frame
            segmented = time.perf_counter()
            if box is None:
                # Never bind background as reference or advance motion KV caches.
                self.reference_bound=False
                motion=self.last_motion.copy() if self.last_motion is not None else self.rest_motion.copy()
                tracking['motion_source']='held' if self.last_motion is not None else 'rest_pose'
                tracking['reference_rebound']=False
            else:
                rebound=not self.reference_bound
                if rebound:
                    self.model.reset_streaming()
                    self.model.bind_reference(cropped)
                    self.reference_bound=True
                output=(self.full_graph.step(cropped) if self.full_graph is not None
                        else self.model.step(cropped))
                motion = output[0, 0].float().cpu().numpy()
                self.last_motion=motion.copy()
                tracking['motion_source']='predicted'
                tracking['reference_rebound']=rebound
            if not np.isfinite(motion).all():
                raise RuntimeError("non-finite motion")
        except Exception:
            self.failed = True
            LOG.exception("Inference failed")
            raise HTTPException(500, "inference failed; reset session before sending more frames")
        done = time.perf_counter()
        self.next_seq += 1
        self.last_seen = time.monotonic()
        result = {
            "session_id": sid, "seq": seq, "timestamp_ms": timestamp_ms,
            "motion": {"rot6d": motion.tolist()},
            "bbox_xyxy": [int(v) for v in box] if box is not None else None,
            "motion_valid": box is not None,
            "tracking": tracking,
            "timing_ms": {
                "decode": decode_ms,
                "sam_crop": (segmented - decoded) * 1000,
                "model_and_d2h": (done - segmented) * 1000,
                "person_detector": tracking['detector_ms'],
                "sam": tracking['sam_ms'],
                "server_total": decode_ms + (done - begin) * 1000,
            },
        }
        # Image history is best-effort: bounded queue, one sample/second, no disk
        # I/O in the inference thread. Never fail an accepted motion on logging.
        capture_begin = time.perf_counter()
        try:
            if self.history.due(self.history_stream):
                mask = None
                if box is not None and self.segmenter.static_out is not None:
                    mask = self.segmenter.static_out[1].detach().cpu().numpy().copy()
                self.history.submit({
                    "stream": self.history_stream, "session_label": sid[:8],
                    "ts": time.time(), "seq": seq, "timestamp_ms": timestamp_ms,
                    "rgb": rgb, "mask": mask, "bbox": list(box) if box is not None else None,
                    "tracking": dict(tracking), "motion_valid": box is not None,
                    "model": dict(self.model_info),
                    "rig_id": self.rig_id, "joints": len(self.names),
                    "timing_ms": dict(result["timing_ms"]), "motion": result["motion"],
                })
        except Exception:
            LOG.exception("Image history sampling failed; motion still returned")
        result["timing_ms"]["history_capture"] = (time.perf_counter()-capture_begin)*1000
        result["timing_ms"]["server_total"] = decode_ms + (time.perf_counter()-begin)*1000
        return result


pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cast-gpu")
decode_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cast-decode")
busy = asyncio.Lock()
runtime = None
pipeline = None


async def run_gpu(fn, *args):
    if busy.locked():
        raise HTTPException(429, "GPU busy; keep at most one in-flight request")
    async with busy:
        task = asyncio.get_running_loop().run_in_executor(pool, fn, *args)
        # Never release the mutex while a cancelled HTTP request still runs on GPU.
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise


async def decode_payload(payload):
    return await asyncio.get_running_loop().run_in_executor(decode_pool, decode_image, payload)


async def process_frame(sid, seq, decoded, timestamp_ms):
    return await run_gpu(runtime.frame, sid, seq, decoded, timestamp_ms)


def validate_pipeline_session(sid):
    # Admission is read-only with respect to the GPU-owned temporal state.
    if sid != runtime.session or time.monotonic()-runtime.last_seen > SESSION_TTL:
        raise HTTPException(409, "session missing/expired; create a new session")
    if runtime.failed:
        raise HTTPException(409, "stream failed; drain and reset")


@asynccontextmanager
async def lifespan(app):
    global runtime, pipeline
    if len(API_KEY) < 24:
        raise RuntimeError("Set CAST_API_KEY to a random secret of at least 24 characters")
    runtime = Runtime()
    await run_gpu(runtime.load)
    pipeline = OrderedPipeline(process_frame, decode_payload, validate_pipeline_session,
                               capacity=PIPELINE_CAPACITY)
    yield
    await pipeline.close()
    decode_pool.shutdown(wait=True)
    pool.shutdown(wait=True)
    if runtime.history is not None:
        await asyncio.to_thread(runtime.history.close)


app = FastAPI(title="CAST Image-to-Motion", version="1.2.0", lifespan=lifespan)


@app.middleware("http")
async def auth_and_limits(request, call_next):
    from fastapi.responses import JSONResponse
    if request.url.path != "/healthz":
        key = request.headers.get("x-api-key", "")
        if not API_KEY or not hmac.compare_digest(key, API_KEY):
            return JSONResponse({"detail": "invalid X-API-Key"}, status_code=401)
    return await call_next(request)


async def bounded_body(request):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_BODY:
            raise HTTPException(413, "request exceeds 8 MiB")
    return bytes(data)


@app.get("/healthz")
async def health():
    return {"ready": runtime is not None, "backbone": runtime.version if runtime else None,
            "model": runtime.model_info if runtime else None,
            "acceleration": runtime.acceleration_info() if runtime else None,
            "gpu_busy": busy.locked(), "protocol_version": "1.2.0",
            "rig_ready": runtime.rig_key is not None if runtime else False,
            "default_target_mode": "human", "person_localization": "ssdlite320_sam_guard_v1",
            "pipeline_capacity": PIPELINE_CAPACITY,
            "pipeline_pending": len(pipeline.jobs) if pipeline else 0,
            "http_connection_limit": http_policy_snapshot()["tcp_connection_limit"],
            "http_request_limit": http_policy_snapshot()["active_request_limit"],
            "http_policy": http_policy_snapshot()}


@app.post("/v1/sessions")
async def create_session(request: Request):
    try:
        body = SessionRequest.model_validate_json(await bounded_body(request))
        if body.rig is not None:
            validate_rig(body.rig)
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(422, str(exc)) from exc
    async with pipeline.control():
        result = await run_gpu(runtime.create_session, body.rig, body.target_mode)
        result["epoch"] = pipeline.bind(result["session_id"])
        result["pipeline_capacity"] = PIPELINE_CAPACITY
        result["recommended_inflight"] = min(6, PIPELINE_CAPACITY)
        result["protocol_version"] = "1.2.0"
        return result


@app.post("/v1/sessions/{sid}/frames")
async def frame(sid: str, request: Request, seq: int, timestamp_ms: float | None = None,
                epoch: str | None = None):
    if seq < 0 or (timestamp_ms is not None and not np.isfinite(timestamp_ms)):
        raise HTTPException(422, "invalid seq/timestamp_ms")
    if request.headers.get("content-type", "").split(";")[0] not in ("image/jpeg", "image/png"):
        raise HTTPException(415, "send raw image/jpeg or image/png body")
    return await pipeline.handle(sid, seq, lambda: bounded_body(request),
                                 epoch=epoch, timestamp_ms=timestamp_ms)


@app.get("/v1/sessions/{sid}/status")
async def session_status(sid: str):
    if sid != runtime.session or time.monotonic()-runtime.last_seen > SESSION_TTL:
        raise HTTPException(409, "session missing/expired")
    return {"session_id": sid, "pipeline": pipeline.snapshot()}


@app.post("/v1/sessions/{sid}/reset")
async def reset(sid: str):
    async with pipeline.control():
        result = await run_gpu(runtime.reset, sid)
        result["epoch"] = pipeline.bind(sid)
        return result


@app.delete("/v1/sessions/{sid}")
async def close(sid: str):
    async with pipeline.control():
        result = await run_gpu(runtime.close, sid)
        pipeline.bind(None)
        return result
