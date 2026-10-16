"""Explicitly fetch pinned external code; no weights and no bundled Git history."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--accept-dinov3-license',action='store_true',
               help='Confirm you reviewed and accept licenses/DINOv3-external-LICENSE.md')
args=p.parse_args()
if not args.accept_dinov3_license:
    p.error('Review the separate DINOv3 terms, then explicitly use --accept-dinov3-license')
manifest=json.loads((ROOT/'third_party/dependencies.json').read_text())
for name,entry in manifest.items():
    dest=ROOT/'third_party'/name
    if dest.exists():
        head=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
        if head!=entry['commit']:raise RuntimeError(f'{name}: existing checkout differs; refusing to overwrite')
        dirty=subprocess.check_output(['git','-C',str(dest),'status','--porcelain'],text=True)
        if dirty.strip():raise RuntimeError(f'{name}: existing checkout is modified; inspect before continuing')
        print(name,'already at pinned revision');continue
    subprocess.run(['git','init',str(dest)],check=True)
    subprocess.run(['git','-C',str(dest),'remote','add','origin',entry['url']],check=True)
    subprocess.run(['git','-C',str(dest),'fetch','--depth','1','origin',entry['commit']],check=True)
    subprocess.run(['git','-C',str(dest),'checkout','--detach','FETCH_HEAD'],check=True)
    print(name,entry['commit'])
