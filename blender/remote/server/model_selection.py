"""Resolve relative model manifests, require hashes, no silent legacy fallback."""
import hashlib
import json
import os
import re
from pathlib import Path
from settings import REPOSITORY_ROOT, SETTINGS


MODEL_CONFIGS = {
    'CAST-B': REPOSITORY_ROOT/'configs/experiment/CAST_B_dinov3_causal.yaml',
    'CAST-L': REPOSITORY_ROOT/'configs/experiment/CAST_L_dinov3_causal.yaml',
}


def load_model_config(family, *, portable=False):
    """Read the repository model definition and apply only streaming/path overrides."""
    from utils.config_utils import load_yaml_config
    if family not in MODEL_CONFIGS:
        raise ValueError('Model family must be CAST-B or CAST-L')
    source = load_yaml_config(str(MODEL_CONFIGS[family]))
    config = {
        'model_family': family,
        'data': {key: source['data'][key] for key in (
            'max_joints', 'image_size', 'dynamic_batch_max_joints',
            'graph_attention_auto_group_max_pair_ratio')},
        'model': source['model'],
    }
    params = config['model']['params']
    for stage in ('encoder_cfg', 'decoder_cfg'):
        stage_params = params[stage]['params']
        if stage_params.get('causal_chunk_size', 1) != 1:
            raise ValueError('Remote inference requires frame-wise causal attention')
        stage_params.update(causal=True, causal_chunk_size=1)
    backbone = params['image_backbone_cfg']['params']
    for key, variable in (
        ('repo', 'CAST_DINOV3_REPO'), ('weights', 'CAST_DINOV3_WEIGHTS'),
        ('compile_cache_dir', 'TORCHINDUCTOR_CACHE_DIR'),
    ):
        backbone[key] = '${' + variable + '}' if portable else os.environ[variable]
    return config


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while data:=f.read(8*1024**2):h.update(data)
    return h.hexdigest()


def contained(root,name):
    p=(root/name).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('Manifest paths must remain under the assets directory')
    return p


def resolve_model(home,version):
    root=Path(home).resolve()
    pointer=root/'active_model.json'
    if not pointer.is_file():
        raise RuntimeError('No active model. Run tools/prepare_checkpoint.py with your authorized checkpoint and config.')
    spec=json.loads(pointer.read_text())
    if not spec.get('verified') or spec['backbone']!=version:
        raise RuntimeError('Unverified or incompatible model manifest')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}',str(spec.get('id',''))):
        raise ValueError('Invalid model id')
    if spec.get('model_family') not in MODEL_CONFIGS:
        raise ValueError('Model family must be CAST-B or CAST-L; prepare the checkpoint again')
    paths={key:contained(root,spec[key]) for key in ('config','checkpoint')}
    for key,p in paths.items():
        if not p.is_file() or sha256(p)!=spec[key+'_sha256']:
            raise RuntimeError(f'Model {key} missing or SHA256 mismatch')
    keys=('id','model_family','backbone','global_step','encoder_layers','decoder_layers','q_dim','num_heads','image_size','cache_length','amp_dtype','source_sha256','checkpoint_name')
    public={key:spec[key] for key in keys}
    return {**{key:str(value) for key,value in paths.items()},
            'cache_dir':str(SETTINGS.state_dir/'cache'/spec['id']),'public':public}
