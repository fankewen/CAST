"""Local process manager; only manages this checkout's tracked processes.

No auto-restart here: use a host service manager for supervised deployments.
No port eviction, pkill, global environment change, or other-host PID handling.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from settings import ROOT, SETTINGS

def start_ticks(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,IndexError):return None

def matches(data,entry):
    if not data or data.get('hostname')!=os.uname().nodename:return False
    pid=data.get('pid')
    try:
        return (start_ticks(pid)==data['start_ticks'] and
                str(entry).encode() in Path(f'/proc/{pid}/cmdline').read_bytes())
    except (OSError,KeyError):return False

def main():
    p=argparse.ArgumentParser()
    p.add_argument('service',choices=['api','admin'])
    p.add_argument('action',choices=['start','stop','status'])
    args=p.parse_args()
    entry=ROOT/('run_origin.py' if args.service=='api' else 'admin/run_admin.py')
    record=SETTINGS.state_dir/'runtime'/f'{args.service}.pid.json'
    try:data=json.loads(record.read_text())
    except (OSError,ValueError):data=None
    running=matches(data,entry)
    if args.action=='status':
        print(json.dumps({'running':running,'process':data}));return
    if args.action=='stop':
        if not running:print('No matching local process; nothing stopped');return
        os.kill(data['pid'],signal.SIGTERM)
        for _ in range(60):
            if not matches(data,entry):
                record.unlink(missing_ok=True);print('Stopped');return
            time.sleep(1)
        raise RuntimeError('Still shutting down; inspect logs. No forced kill was issued.')
    if running:print('Already running',data['pid']);return
    if data and data.get('hostname')!=os.uname().nodename:
        raise RuntimeError('PID record belongs to another host; use separate CAST_STATE_DIR per instance')
    filename='service.log' if args.service=='api' else 'admin.log'
    with (SETTINGS.state_dir/'logs'/filename).open('ab') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(entry)],cwd=ROOT,stdin=subprocess.DEVNULL,
                              stdout=log,stderr=log,start_new_session=True)
    time.sleep(.2)
    if proc.poll() is not None:raise RuntimeError('Startup failed; inspect '+filename)
    data={'pid':proc.pid,'hostname':os.uname().nodename,'start_ticks':start_ticks(proc.pid),'entry':str(entry)}
    record.write_text(json.dumps(data))
    print('Started process; wait for readiness separately:',proc.pid)

if __name__=='__main__':main()
