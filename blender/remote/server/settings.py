"""Portable settings. Load before torch so CUDA/cache environment is deterministic.

Paths in .env are relative to blender/remote/server/. No shell evaluation or
interpolation is performed. Process environment overrides .env values.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parent
REPOSITORY_ROOT=ROOT.parents[2]
if not (REPOSITORY_ROOT/'inference/deploy_realtime_video2motion.py').is_file():
    raise RuntimeError('Run the remote service from a complete CAST repository checkout')


def load_env(path):
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):line=line[7:]
        key,sep,value=line.partition('=')
        if not sep or not re.fullmatch(r'[A-Z][A-Z0-9_]*',key):
            raise ValueError('Invalid .env key (values are not evaluated as shell)')
        value=value.strip()
        if len(value)>1 and value[0]==value[-1] and value[0] in "\"'":value=value[1:-1]
        os.environ.setdefault(key,value)


load_env(Path(os.environ.get('CAST_ENV_FILE',ROOT/'.env')))


def path(name,default):
    value=Path(os.environ.get(name,str(default))).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT/value).resolve()


@dataclass(frozen=True)
class Settings:
    state_dir: Path
    assets_dir: Path
    admin_local_dir: Path
    api_port: int
    api_url: str
    public_api_url: str
    admin_port: int
    sam_checkpoint: Path
    person_checkpoint: Path


state=path('CAST_STATE_DIR','var')
assets=path('CAST_ASSETS_DIR','assets')
api_port=int(os.environ.get('CAST_PORT','8080'))
admin_port=int(os.environ.get('CAST_ADMIN_PORT','8081'))
for port in (api_port,admin_port):
    if not 1<=port<=65535:raise ValueError('Invalid port')
SETTINGS=Settings(
    state_dir=state, assets_dir=assets,
    admin_local_dir=path('CAST_ADMIN_STATE_DIR',state/'admin/local'),
    api_port=api_port, api_url=os.environ.get('CAST_INTERNAL_API_URL',f'http://127.0.0.1:{api_port}').rstrip('/'),
    public_api_url=os.environ.get('CAST_PUBLIC_API_URL',f'http://localhost:{api_port}').rstrip('/'),
    admin_port=admin_port,
    sam_checkpoint=path('CAST_SAM_CHECKPOINT',assets/'weights/sam2/sam2.1_hiera_tiny.pt'),
    person_checkpoint=path('CAST_PERSON_CHECKPOINT',assets/'weights/person/ssdlite320_mobilenet_v3_large_coco-a79551df.pth'))

for folder in (state,state/'runtime',state/'logs',state/'tmp',state/'cache'):
    folder.mkdir(parents=True,exist_ok=True,mode=0o700)

# Avoid importing random globally installed 'models', 'data' or SAM packages.
sys.path.insert(0,str(REPOSITORY_ROOT))
sys.path.insert(1,str(ROOT.parent))
sam_repo=path('CAST_SAM2_REPO',REPOSITORY_ROOT/'sam2')
sys.path.insert(2,str(sam_repo))
os.environ.setdefault('CAST_DINOV3_REPO',str(path('CAST_DINOV3_REPO',REPOSITORY_ROOT/'third_party/dinov3')))
os.environ.setdefault('CAST_DINOV3_WEIGHTS',str(path('CAST_DINOV3_WEIGHTS',REPOSITORY_ROOT/'third_party/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')))
# Expand explicitly set relative model paths, too.
for name in ('CAST_DINOV3_REPO','CAST_DINOV3_WEIGHTS'):
    os.environ[name]=str(path(name,os.environ[name]))
os.environ.setdefault('CUDA_VISIBLE_DEVICES',os.environ.get('CAST_GPU','0'))
os.environ.setdefault('TORCH_HOME',str(state/'cache/torch'))
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(state/'cache/inductor'))
os.environ.setdefault('TRITON_CACHE_DIR',str(state/'cache/triton'))
os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(state/'cache/extensions'))
os.environ.setdefault('TMPDIR',str(state/'tmp'))
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','4')
os.environ.setdefault('XFORMERS_IGNORE_FLASH_VERSION_CHECK','1')
os.environ.setdefault('PYTHONUNBUFFERED','1')
os.environ.setdefault('CAST_PORT',str(api_port))
os.environ.setdefault('CAST_PIPELINE_CAPACITY','10')
os.environ.setdefault('CAST_HTTP_ACTIVE_LIMIT','64')
os.environ.setdefault('CAST_HTTP_TCP_LIMIT','256')
os.environ.setdefault('CAST_FULL_STEP_GRAPH','1')
os.environ.setdefault('CAST_HISTORY_ENABLED','0')
os.environ.setdefault('CAST_HOST','127.0.0.1')
