"""Read-only CAST operations console. No imports of torch or inference code."""
import asyncio
import collections
from contextlib import asynccontextmanager, closing, contextmanager
import hmac
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
import sqlite3
import subprocess
import threading
import time
import urllib.request

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import SETTINGS
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

HOME = Path(__file__).resolve().parent
DEPLOY = SETTINGS.state_dir
DEPLOY.mkdir(parents=True, exist_ok=True)
(DEPLOY / "admin" / "backups").mkdir(parents=True, exist_ok=True)
LOCAL = SETTINGS.admin_local_dir
LOCAL.mkdir(parents=True, exist_ok=True, mode=0o700)
DB = LOCAL / "monitor.sqlite3"
BACKUP = DEPLOY / "admin/backups/monitor.sqlite3"
USER = os.environ.get("CAST_ADMIN_USER", "admin")
PASSWORD = os.environ.get("CAST_ADMIN_PASSWORD", "")
SAMPLE_SECONDS = 5
SESSION_SECONDS = 8 * 3600
SECRETS = [s for s in (PASSWORD, os.environ.get("CAST_API_KEY", "")) if s]
LOGS = {
    "service": DEPLOY / "logs/service.log",
    "supervisor": DEPLOY / "logs/supervisor.log",
    "admin": DEPLOY / "logs/admin.log",
    "http_origin": DEPLOY / "logs/http_origin.jsonl",
}
VISUAL_ROOT = DEPLOY / "diagnostics/offline"
VISUAL_KINDS = ("original", "bbox", "sam_mask", "mask_overlay", "crop", "model_input_256")
VISUAL_SUMMARIES = ("contact_sheet.jpg", "first_frames.jpg", "first_frame_variants.jpg")
LOG = logging.getLogger("cast.admin")
LOG.setLevel(logging.INFO)
latest = {}
sessions = {}
failures = collections.defaultdict(collections.deque)
state_lock = threading.Lock()
database_lock = threading.RLock()
stop = threading.Event()
last_sampler_error = None


def redact(text):
    for value in SECRETS:
        text = text.replace(value, "[REDACTED]")
    return re.sub(r"(?i)((?:api[_-]?key|password|authorization|token)\s*[=:]\s*)\S+",
                  r"\1[REDACTED]", text)


@contextmanager
def connect():
    con = sqlite3.connect(DB, timeout=10)
    try:
        con.row_factory = sqlite3.Row
        with con:
            yield con
    finally:
        con.close()


def initialize():
    if not DB.exists() and BACKUP.exists():
        shutil.copyfile(BACKUP, DB)
    with connect() as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE IF NOT EXISTS samples (ts REAL PRIMARY KEY, data TEXT NOT NULL)")
        con.execute("CREATE TABLE IF NOT EXISTS events "
                    "(id INTEGER PRIMARY KEY, ts REAL, level TEXT, source TEXT, message TEXT)")
        con.execute("CREATE INDEX IF NOT EXISTS events_ts ON events(ts)")
    os.chmod(DB, 0o600)


def event(level, source, message):
    with database_lock, connect() as con:
        con.execute("INSERT INTO events(ts,level,source,message) VALUES (?,?,?,?)",
                    (time.time(), level, source, redact(message)[:3000]))


def tail(path, limit=300, max_bytes=256*1024):
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size-max_bytes))
            raw = f.read(max_bytes).decode("utf-8", "replace")
        return redact("\n".join(raw.splitlines()[-limit:])), size, path.stat().st_mtime
    except FileNotFoundError:
        return "", 0, None


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=3)
        return p.stdout if p.returncode == 0 else ""
    except (subprocess.TimeoutExpired, OSError):
        return ""


def read_process():
    try:
        record = json.loads((DEPLOY / "runtime/api.pid.json").read_text())
        if record.get("hostname") != os.uname().nodename:
            return {"alive": False, "pid": None, "note": "Process in another host/container; use health status"}
        pid = int(record["pid"])
        proc = Path(f"/proc/{pid}")
        cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        if "run_origin.py" not in cmd:
            return {"alive": False, "pid": pid}
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if stat[19] != record.get("start_ticks"):
            return {"alive": False, "pid": None, "note": "Stale PID record"}
        ticks = os.sysconf("SC_CLK_TCK")
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        return {
            "alive": True, "pid": pid, "cpu_seconds": (int(stat[11])+int(stat[12]))/ticks,
            "uptime_seconds": max(0, uptime-int(stat[19])/ticks),
            "rss_mb": int(stat[21])*os.sysconf("SC_PAGE_SIZE")/1024**2,
            "threads": int(stat[17]),
        }
    except (OSError, ValueError, IndexError, KeyError):
        return {"alive": False, "pid": None}


class Sampler:
    def __init__(self):
        self.log_offsets = {}
        self.cpu_prev = None
        self.proc_prev = None
        self.last_backup = 0
        self.old_ready = None
        self.old_pid = None
        self.log_totals = collections.Counter()
        self.newest_log_lines = []
        for name in ("service", "supervisor"):
            path = LOGS[name]
            text, size, _ = tail(path, limit=30000, max_bytes=2*1024*1024)
            counts = self.classify(text)
            self.log_totals.update(counts)
            try:
                self.log_offsets[name] = (path.stat().st_ino, size)
            except OSError:
                self.log_offsets[name] = (None, 0)
            if counts:
                event("info", name, f"启动扫描（最多末尾2MiB，非实时新增告警）：{dict(counts)}")

    @staticmethod
    def classify(text):
        return collections.Counter({
            "concurrency_limit": text.count("Exceeded concurrency limit"),
            "errors": len(re.findall(r"(?m)^.*(?:ERROR:| - ERROR - |Traceback)", text)),
            "warnings": len(re.findall(r"(?m)^.*(?:WARNING:| - WARNING - )", text)),
        })

    def logs(self):
        for name in ("service", "supervisor"):
            path = LOGS[name]
            try:
                st = path.stat()
                inode, offset = self.log_offsets.get(name, (None, 0))
                if st.st_ino != inode or st.st_size < offset:
                    offset = 0
                if st.st_size > offset:
                    with path.open("rb") as f:
                        f.seek(max(offset, st.st_size-512*1024))
                        text = f.read(512*1024).decode("utf-8", "replace")
                    counts = self.classify(text)
                    self.log_totals.update(counts)
                    for kind, count in counts.items():
                        if count:
                            event("error" if kind == "errors" else "warning", name,
                                  f"本轮新增 {kind}: {count}。时间为采集时间，原日志可能不含时间戳。")
                    self.newest_log_lines = redact(text).splitlines()[-8:]
                self.log_offsets[name] = (st.st_ino, st.st_size)
            except OSError:
                pass

    def sample(self):
        start = time.monotonic()
        health = {"ready": False, "error": None, "probe_ms": None}
        try:
            # Never inherit installation proxies; close connection every probe.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(SETTINGS.api_url + "/healthz", timeout=2) as response:
                health.update(json.loads(response.read(4096)))
                health["http_status"] = response.status
            health["probe_ms"] = (time.monotonic()-start)*1000
        except Exception as exc:
            health["error"] = redact(str(exc))[:300]
        proc = read_process()
        now_mono = time.monotonic()
        if proc["alive"] and self.proc_prev and self.proc_prev[0] == proc["pid"]:
            proc["cpu_percent_one_core"] = max(
                0, (proc["cpu_seconds"]-self.proc_prev[1])/(now_mono-self.proc_prev[2])*100)
        else:
            proc["cpu_percent_one_core"] = None
        if proc["alive"]:
            self.proc_prev = (proc["pid"], proc["cpu_seconds"], now_mono)

        cpu = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
        total, idle = sum(cpu), cpu[3]+cpu[4]
        cpu_percent = None
        if self.cpu_prev and total > self.cpu_prev[0]:
            cpu_percent = 100*(1-(idle-self.cpu_prev[1])/(total-self.cpu_prev[0]))
        self.cpu_prev = (total, idle)
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            mem[k] = int(v.strip().split()[0])
        gpu_text = command(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"])
        gpus = []
        for line in gpu_text.splitlines():
            parts = [v.strip() for v in line.split(",")]
            if len(parts) == 7:
                def number(s):
                    try:
                        return float(s)
                    except ValueError:
                        return None
                gpus.append(dict(zip(("index", "name", "util", "memory_used_mb", "memory_total_mb", "temperature", "power_w"),
                                     [parts[0], parts[1]]+[number(v) for v in parts[2:]])))
        ss = command(["ss", "-tnH", "state", "established", f"( sport = :{SETTINGS.api_port} )"])
        connections = len(ss.strip().splitlines()) if ss.strip() else 0
        queues = []
        for line in ss.splitlines():
            p = line.split()
            if len(p) >= 2:
                try:
                    queues.append((int(p[0]), int(p[1])))
                except ValueError:
                    pass
        self.logs()
        if self.old_ready is not None and self.old_ready != health["ready"]:
            event("info" if health["ready"] else "error", "health",
                  "服务恢复就绪" if health["ready"] else f"服务不就绪: {health.get('error')}")
        if self.old_pid is not None and self.old_pid != proc["pid"]:
            event("warning", "process", f"服务 PID 变化: {self.old_pid} → {proc['pid']}")
        self.old_ready, self.old_pid = health["ready"], proc["pid"]
        disk = shutil.disk_usage(LOCAL)
        stamp = time.time()
        data = {
            "ts": stamp, "health": health, "process": proc, "gpus": gpus,
            "host": {"hostname": os.uname().nodename, "cpu_percent": cpu_percent,
                     "cpu_count": os.cpu_count(), "load": list(os.getloadavg()),
                     "memory_total_gb": mem["MemTotal"]/1024**2,
                     "memory_used_gb": (mem["MemTotal"]-mem.get("MemAvailable", mem["MemFree"]))/1024**2,
                     "local_disk_free_gb": disk.free/1024**3},
            "connections": {"established": connections, "configured_limit": health.get("http_connection_limit",16),
                            "active_requests": health.get("http_policy",{}).get("active_requests"),
                            "request_limit": health.get("http_request_limit"),
                            "origin_policy": health.get("http_policy"),
                            "pipeline_pending": health.get("pipeline_pending"),
                            "pipeline_capacity": health.get("pipeline_capacity"),
                            "recv_queue_bytes": sum(q[0] for q in queues), "send_queue_bytes": sum(q[1] for q in queues)},
            "log_observed_counts": dict(self.log_totals),
            "recent_log_lines": self.newest_log_lines,
            "realtime_inference": {"fps": None, "latency_ms": None, "reason": "8080 未持续上报逐帧耗时/完成帧计数；健康探测时间不是推理延迟。"},
        }
        with database_lock, connect() as con:
            con.execute("INSERT OR REPLACE INTO samples VALUES (?,?)", (stamp, json.dumps(data)))
            if stamp-self.last_backup >= 60:
                con.execute("DELETE FROM samples WHERE ts < ?", (stamp-7*86400,))
                con.execute("DELETE FROM events WHERE ts < ?", (stamp-7*86400,))
        with state_lock:
            latest.clear()
            latest.update(data)
        if stamp-self.last_backup >= 60:
            self.backup()
            self.last_backup = stamp

    @staticmethod
    def backup():
        # WAL is local only. Publish a standalone SQLite backup to shared storage.
        local_backup = LOCAL / "backup.sqlite3"
        with database_lock, connect() as src, closing(sqlite3.connect(local_backup)) as dest:
            src.backup(dest)
        temp = BACKUP.with_suffix(".tmp")
        shutil.copyfile(local_backup, temp)
        os.chmod(temp, 0o600)
        temp.replace(BACKUP)


def loop():
    global last_sampler_error
    sampler = Sampler()
    while not stop.is_set():
        start = time.monotonic()
        try:
            sampler.sample()
            last_sampler_error = None
        except Exception as exc:
            last_sampler_error = redact(str(exc))
            LOG.exception("Monitor sampling failed")
        stop.wait(max(.5, SAMPLE_SECONDS-(time.monotonic()-start)))
    try:
        sampler.backup()
    except Exception:
        LOG.exception("Final backup failed")


@asynccontextmanager
async def lifespan(app):
    if len(PASSWORD) < 24:
        raise RuntimeError("Set a random CAST_ADMIN_PASSWORD (>=24 chars) in .env")
    initialize()
    event("info", "admin", "管理员监控启动：只读采集，5秒周期，保留7天；不占用动捕会话。")
    stop.clear()
    thread = threading.Thread(target=loop, daemon=True, name="cast-monitor")
    thread.start()
    try:
        yield
    finally:
        stop.set()
        await asyncio.to_thread(thread.join, 10)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def security(request, call_next):
    path = request.url.path
    if path.startswith("/api/") and path != "/api/login":
        sid = request.cookies.get("cast_admin")
        if not sid or sessions.get(sid, 0) < time.time():
            return JSONResponse({"detail": "请登录管理员控制台"}, status_code=401)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
    return response


@app.get("/")
async def index():
    return FileResponse(HOME / "static/index.html")


@app.get("/assets/{name}")
async def asset(name: str):
    if name not in ("app.js", "style.css"):
        raise HTTPException(404)
    return FileResponse(HOME / "static" / name)


@app.get("/healthz")
async def admin_health():
    return {"status": "ok", "role": "read-only-admin"}


@app.post("/api/login")
async def login(request: Request):
    origin = request.headers.get("origin")
    if origin and origin != str(request.base_url).rstrip("/"):
        raise HTTPException(403, "跨域登录已禁用")
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    for key in list(failures):
        while failures[key] and failures[key][0] < now-300:
            failures[key].popleft()
        if not failures[key]:
            del failures[key]
    if len(failures[ip]) >= 8 or sum(map(len, failures.values())) >= 100:
        raise HTTPException(429, "登录尝试过多，请5分钟后再试")
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 4096:
            raise HTTPException(413)
    try:
        data = json.loads(raw)
        username, password = data["username"], data["password"]
        if not isinstance(username, str) or not isinstance(password, str):
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise HTTPException(400, "登录数据格式错误")
    # Compare bytes to handle non-ASCII input safely.
    good_user = hmac.compare_digest(username.encode(), USER.encode())
    good_password = hmac.compare_digest(password.encode(), PASSWORD.encode())
    if not (good_user and good_password):
        failures[ip].append(now)
        event("warning", "auth", "管理员登录失败（不记录密码）")
        raise HTTPException(401, "用户名或密码错误")
    failures.pop(ip, None)
    for key in list(sessions):
        if sessions[key] < now:
            del sessions[key]
    if len(sessions) >= 64:
        sessions.pop(next(iter(sessions)))
    token = secrets.token_urlsafe(32)
    sessions[token] = now+SESSION_SECONDS
    response = JSONResponse({"ok": True, "username": USER})
    response.set_cookie("cast_admin", token, max_age=SESSION_SECONDS, httponly=True,
                        samesite="strict", secure=os.environ.get("CAST_ADMIN_SECURE_COOKIE") == "1")
    event("info", "auth", "管理员登录成功")
    return response


@app.post("/api/logout")
async def logout(request: Request):
    sessions.pop(request.cookies.get("cast_admin"), None)
    response = JSONResponse({"ok": True})
    response.delete_cookie("cast_admin")
    return response


@app.get("/api/overview")
def overview():
    with state_lock:
        data = dict(latest)
    with database_lock, connect() as con:
        first = con.execute("SELECT MIN(ts),COUNT(*) FROM samples").fetchone()
        alerts = [dict(r) for r in con.execute(
            "SELECT * FROM events WHERE level IN ('warning','error') ORDER BY id DESC LIMIT 8")]
    return {"snapshot": data, "monitor": {"sample_seconds": SAMPLE_SECONDS, "retention_days": 7,
            "first_sample": first[0], "sample_count": first[1], "sampler_error": last_sampler_error,
            "stale": not data or time.time()-data.get("ts", 0) > 20},
            "alerts": alerts, "instance": {"url": SETTINGS.public_api_url,
            "admin_port": SETTINGS.admin_port, "api_port": SETTINGS.api_port, "model": "DINOv3", "mode": "只读监控",
            "deployment": str(DEPLOY), "note": "GPU/主机指标包含同机其他任务，不等于 CAST 独占使用量。"}}


@app.get("/api/history")
def history(minutes: int = 60):
    if not 1 <= minutes <= 7*24*60:
        raise HTTPException(422)
    with database_lock, connect() as con:
        rows = con.execute("SELECT ts,data FROM samples WHERE ts>=? ORDER BY ts",
                           (time.time()-minutes*60,)).fetchall()
    stride = max(1, (len(rows)+479)//480)
    chosen = list(rows[::stride])
    if rows and chosen[-1]["ts"] != rows[-1]["ts"]:
        chosen.append(rows[-1])
    points = []
    for row in chosen:
        d = json.loads(row["data"])
        gpu = d.get("gpus", [{}])[0] if d.get("gpus") else {}
        points.append({"ts": d["ts"], "gpu_util": gpu.get("util"), "gpu_mem": gpu.get("memory_used_mb"),
                       "cpu": d["host"].get("cpu_percent"), "connections": d["connections"]["established"],
                       "probe_ms": d["health"].get("probe_ms"), "ready": d["health"]["ready"]})
    return {"points": points, "raw_count": len(rows), "stride": stride, "minutes": minutes,
            "note": "趋势图为采样点抽取，非峰值聚合；未采集时段无历史。"}


@app.get("/api/events")
def events(level: str = "all", limit: int = 100):
    if level not in ("all", "info", "warning", "error") or not 1 <= limit <= 500:
        raise HTTPException(422)
    with database_lock, connect() as con:
        rows = con.execute("SELECT * FROM events WHERE (?='all' OR level=?) ORDER BY id DESC LIMIT ?",
                           (level, level, limit)).fetchall()
    return {"events": [dict(r) for r in rows]}


@app.get("/api/logs")
def logs(source: str = "service", lines: int = 250, query: str = ""):
    if source not in LOGS or not 1 <= lines <= 1000 or len(query) > 200:
        raise HTTPException(422)
    text, size, modified = tail(LOGS[source], lines)
    if query:
        text = "\n".join(line for line in text.splitlines() if query.lower() in line.lower())
    return {"source": source, "text": text, "bytes": size, "modified": modified,
            "note": "只读末尾最多256KiB；搜索仅限已读取的末尾行。原日志无时间戳的行不能还原事件时间。"}


@app.get("/api/benchmarks")
def benchmarks():
    items = []
    for path in sorted((DEPLOY / "logs").glob("benchmark*.json")):
        try:
            if path.stat().st_size > 1024*1024:
                continue
            d = json.loads(path.read_text())
            if "throughput_fps" in d:
                items.append({"file":path.name,"date":d.get("date"),
                    "shape":d.get("input_shape_hwc"),"frames":d.get("frames"),
                    "warmup":0,"url":SETTINGS.public_api_url,"encoding":True,
                    "roundtrip":{"mean_ms":d.get("mean_roundtrip_ms"),"p95_ms":d.get("p95_ms"),
                                 "fps_from_mean":d.get("throughput_fps")},
                    "window":d.get("window"),"first_frame_ms":None,"joints":d.get("joints"),
                    "model_id":d.get("model",{}).get("id"),"acceleration":d.get("acceleration"),
                    "label":str(d.get("boundary",""))+" Valid motion: "+str(d.get("valid_motion_frames"))})
                continue
            if "results" in d:  # Protocol 1.1 pipelining benchmark, multiple windows.
                for n, r in enumerate(d["results"]):
                    items.append({"file": f"{path.name} / run {n+1}", "date": d.get("date"),
                                  "shape": [d["input"][1],d["input"][0],3], "frames": r["frames"],
                                  "warmup": d.get("warmup"), "url": d.get("url"),
                                  "encoding": d.get("jpeg_encoding_included"),
                                  "roundtrip": {**r["client_ordered_latency"], "fps_from_mean": r["fps"]},
                                  "window": r["window"], "stages": r.get("stages"),
                                  "response_interval": r.get("response_interval"),
                                  "first_frame_ms": None, "joints": d.get("joints"),
                                  "model_id": d.get("model_id"), "acceleration": d.get("acceleration"),
                                  "label": (r.get("label", "") + " 吞吐来自总完成帧/墙钟时间；不是1000/单帧排队延迟。")})
                continue
            if "measured_fps" in d:
                items.append({"file": path.name, "date": d.get("date"),
                              "shape": [d["input"][1],d["input"][0],3],
                              "frames": d["submitted_and_completed"], "warmup": 15,
                              "url": SETTINGS.public_api_url, "encoding": True,
                              "roundtrip": {**d["ordered_client_latency"], "fps_from_mean": d["measured_fps"]},
                              "window": d["window"], "response_interval": d["response_interval"],
                              "first_frame_ms": None, "joints": d.get("joints"),
                              "label": f"节奏发送 {d['target_fps']} FPS；提交前丢弃 {d['dropped_before_submit']} 帧。"})
                continue
            items.append({"file": path.name, "date": d.get("date"),
                          "shape": d.get("input_shape_hwc"), "frames": d.get("frames"),
                          "warmup": d.get("warmup"), "url": d.get("url"),
                          "encoding": d.get("client_encoding_included"),
                          "roundtrip": d.get("roundtrip"), "stages": d.get("stages"),
                          "first_frame_ms": d.get("first_frame_ms"),
                          "joints": d.get("session", {}).get("joints")})
        except (OSError, ValueError):
            continue
    return {"items": items, "note": "历史受控测速，不是当前实时 FPS。测试边界以每条记录的说明为准；不代表当前端侧体验。"}


@app.get("/api/visuals")
def visuals():
    if not VISUAL_ROOT.is_dir():
        return {"frames": [], "summaries": [], "boxes": [], "source": "offline", "note": "暂无已保存的诊断图片。"}
    try:
        boxes = json.loads((VISUAL_ROOT / "boxes.json").read_text())
        provenance = json.loads((VISUAL_ROOT / "provenance.json").read_text())
    except (OSError, ValueError):
        boxes, provenance = [], {}
    by_seq = {x["seq"]: x for x in boxes if isinstance(x, dict) and isinstance(x.get("seq"), int)}
    frames = []
    for p in sorted(VISUAL_ROOT.glob("*_original.png")):
        if not re.fullmatch(r"\d{3}_original\.png", p.name):
            continue
        seq = int(p.name[:3])
        files = {kind: f"/api/visuals/image/{seq:03d}_{kind}.png" for kind in VISUAL_KINDS
                 if (VISUAL_ROOT / f"{seq:03d}_{kind}.png").is_file()}
        frames.append({"seq": seq, "files": files, "bbox": by_seq.get(seq, {}).get("bbox_xyxy")})
    summaries = [{"name": name, "url": f"/api/visuals/image/{name}"} for name in VISUAL_SUMMARIES
                 if (VISUAL_ROOT / name).is_file()]
    return {"frames": frames, "summaries": summaries, "boxes": boxes, "provenance": provenance,
            "source": "offline", "source_size": provenance.get("source_size"), "source_fps": provenance.get("source_fps"),
            "note": "管理员提供的离线诊断样本；不是当前线上会话，不随源码分发。"}


@app.get("/api/visuals/image/{name}")
def visual_image(name: str):
    pattern = r"\d{3}_(?:" + "|".join(VISUAL_KINDS) + r")\.png"
    if name not in VISUAL_SUMMARIES and not re.fullmatch(pattern, name):
        raise HTTPException(404)
    path = VISUAL_ROOT / name
    if not path.is_file() or path.resolve().parent != VISUAL_ROOT.resolve():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png" if name.endswith(".png") else "image/jpeg")


LIVE_ROOT = DEPLOY / "live_history"


def live_index():
    try:
        return json.loads((LIVE_ROOT / "index.json").read_text())
    except (OSError, ValueError):
        return {"records": [], "enabled": False, "updated_at": None}


@app.get("/api/live-history")
def live_history(stream: str = "", before: float | None = None, limit: int = 36):
    if not 1 <= limit <= 100 or (stream and not re.fullmatch(r"[0-9a-f]{32}", stream)):
        raise HTTPException(422)
    data = live_index()
    records = data.pop("records", [])
    groups = {}
    for r in reversed(records):
        g = groups.setdefault(r["stream"], {"stream": r["stream"], "session_label": r["session_label"],
            "rig_id": r["rig_id"], "joints": r["joints"], "first_ts": r["ts"], "last_ts": r["ts"], "samples": 0})
        g["samples"] += 1
        g["first_ts"] = min(g["first_ts"], r["ts"])
    filtered = [r for r in reversed(records) if (not stream or r["stream"] == stream)
                and (before is None or r["ts"] < before)]
    selected = filtered[:limit]
    return {"status": data, "streams": list(groups.values()), "records": selected,
            "total_retained": len(records), "has_more": len(filtered)>limit,
            "next_before": selected[-1]["ts"] if len(filtered)>limit else None,
            "note": "线上成功推理帧的异步抽样历史，不是全帧录像。每会话/重置段首帧及随后最多每秒1帧；队列满会丢弃历史样本，不丢推理帧。"}


def live_folder(record_id):
    if not re.fullmatch(r"[0-9a-f]{32}", record_id):
        raise HTTPException(404)
    folder = LIVE_ROOT / record_id
    if folder.resolve().parent != LIVE_ROOT.resolve() or not (folder / "meta.json").is_file():
        raise HTTPException(404, "记录不存在或已到期清理")
    return folder


@app.get("/api/live-history/{record_id}")
def live_detail(record_id: str):
    folder = live_folder(record_id)
    try:
        data = json.loads((folder / "meta.json").read_text())
    except (OSError, ValueError):
        raise HTTPException(404, "记录已到期清理")
    data["files"] = {kind: f"/api/live-history/{record_id}/image/{kind}"
                     for kind in data["summary"]["available"]}
    return data


@app.get("/api/live-history/{record_id}/image/{kind}")
def live_image(record_id: str, kind: str):
    if kind not in VISUAL_KINDS:
        raise HTTPException(404)
    folder = live_folder(record_id)
    extension = "png" if kind in ("sam_mask", "model_input_256") else "jpg"
    path = folder / f"{kind}.{extension}"
    if not path.is_file() or path.resolve().parent != folder.resolve():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png" if extension == "png" else "image/jpeg")
