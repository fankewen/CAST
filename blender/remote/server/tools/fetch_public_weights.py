"""Explicit downloads for public SAM2-Tiny + torchvision person detector weights.

DINOv3 and CAST weights are NOT downloaded: obtain them from their authorized
providers under the relevant terms. This script does not grant model licenses.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from settings import SETTINGS

p=argparse.ArgumentParser()
p.add_argument('--confirm-model-terms',action='store_true')
args=p.parse_args()
if not args.confirm_model_terms:p.error('Review docs/MODELS.md and explicitly use --confirm-model-terms')
assets=[
    (SETTINGS.sam_checkpoint,'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
     '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'),
    (SETTINGS.person_checkpoint,'https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth',
     'a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2'),
]
for path,url,digest in assets:
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError('Existing file has a different hash; not overwritten')
        print(path.name,'verified');continue
    temp=path.with_suffix('.downloading')
    h=hashlib.sha256()
    with urllib.request.urlopen(url,timeout=60) as response,temp.open('wb') as target:
        while chunk:=response.read(1024**2):target.write(chunk);h.update(chunk)
    if h.hexdigest()!=digest:raise RuntimeError('Download checksum mismatch; temporary file retained for inspection')
    temp.replace(path);print(path.name,'downloaded and SHA256 verified')
