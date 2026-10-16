"""Explicit benchmark on your own idle service and authorized local inputs."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from settings import SETTINGS
from client.pipeline_client import PipelinedMotionClient
import cv2
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--url',default=SETTINGS.api_url)
p.add_argument('--video',type=Path,required=True)
p.add_argument('--rig',type=Path,required=True)
p.add_argument('--frames',type=int,default=300)
p.add_argument('--window',type=int,default=2)
p.add_argument('--target-mode',choices=['human','generic'],default='human')
p.add_argument('--output',type=Path,default=SETTINGS.state_dir/'logs/benchmark-local.json')
args=p.parse_args()
if args.frames<1:p.error('frames must be positive')
cap=cv2.VideoCapture(str(args.video));images=[]
for _ in range(min(args.frames,300)):
    ok,image=cap.read()
    if not ok:break
    images.append(image)
cap.release()
if not images:raise ValueError('Video contains no decodable frames')
client=PipelinedMotionClient(args.url,os.environ['CAST_API_KEY'],window=args.window)
meta=client.start(json.loads(args.rig.read_text()),target_mode=args.target_mode)
pending=deque();latency=[];valid=0
def drain_one():
    global valid
    seq,t,f=pending.popleft();r=f.result();assert r['seq']==seq
    motion=np.asarray(r['motion']['rot6d'])
    assert motion.shape==(meta['joints'],6) and np.isfinite(motion).all(), 'Invalid motion output'
    assert r.get('epoch')==meta['epoch'], 'Epoch mismatch'
    latency.append((time.perf_counter()-t)*1000);valid+=int(r.get('motion_valid',True))
try:
    start=time.perf_counter()
    for seq in range(args.frames):
        if len(pending)>=args.window:drain_one()
        t=time.perf_counter();ok,jpg=cv2.imencode('.jpg',images[seq%len(images)],[cv2.IMWRITE_JPEG_QUALITY,90]);assert ok
        pending.append((seq,t,client.submit(jpg.tobytes())))
    while pending:drain_one()
    elapsed=time.perf_counter()-start
finally:client.close()
result={'date':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'frames':args.frames,'valid_motion_frames':valid,
        'window':args.window,'elapsed_seconds':elapsed,'throughput_fps':args.frames/elapsed,
        'mean_roundtrip_ms':float(np.mean(latency)),'p95_ms':float(np.percentile(latency,95)),
        'model':meta.get('model'),'acceleration':meta.get('acceleration'),
        'input_shape_hwc':list(images[0].shape),'joints':meta['joints'],
        'client_encoding_included':True,
        'boundary':'Includes JPEG encoding + HTTP + inference; excludes video decode/Blender UI. First reference bind included; not steady-state-only. Invalid/no-person frames are reported separately.'}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
