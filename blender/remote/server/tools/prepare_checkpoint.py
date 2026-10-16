"""Validate a CAST-B or CAST-L checkpoint and write a relative runtime manifest.

Use only checkpoints you are authorized to obtain/use. weights_only=True never
falls back to arbitrary pickle execution. A mismatch is an error, not partial load.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from settings import SETTINGS
import torch
import yaml
from utils.config_utils import instantiate_from_config
from model_selection import MODEL_CONFIGS, load_model_config, sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--model', choices=tuple(MODEL_CONFIGS), required=True,
                   help='CAST model family; selects the matching main-repository configuration')
    p.add_argument('--model-id',required=True)
    p.add_argument('--activate',action='store_true',help='Select for NEXT service start; does not restart a running service')
    args=p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}',args.model_id):p.error('Invalid model-id')
    if not args.checkpoint.is_file():p.error('Checkpoint does not exist')
    dest=SETTINGS.assets_dir/'models'/args.model_id
    if dest.exists():p.error('Model directory exists; use a new id (never overwrite active weights)')
    # Do not silently infer attention heads or temporal mode from tensor shapes.
    cfg=load_model_config(args.model)
    params=cfg['model']['params']
    data=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    source=data.get('model_state',data)
    if not isinstance(source,dict) or not source:p.error('Missing model_state tensors')
    state={}
    for key,value in source.items():
        if not isinstance(key,str) or not isinstance(value,torch.Tensor):p.error('Invalid model state')
        state[key]=value
    model=instantiate_from_config(cfg['model'])
    expected=model.trainable_state_dict()
    missing=sorted(set(expected)-set(state));extra=sorted(set(state)-set(expected))
    mismatches=[k for k in expected.keys()&state.keys() if expected[k].shape!=state[k].shape]
    if missing or extra or mismatches:
        raise RuntimeError(json.dumps({'missing':missing,'unexpected':extra,'shape_mismatch':mismatches}))
    model.load_trainable_state_dict(state)
    for key,value in model.trainable_state_dict().items():
        if not torch.equal(value,state[key]):raise RuntimeError('Tensor changed during load: '+key)
    step=int(data.get('global_step',0))
    dest.mkdir(parents=True,mode=0o700)
    config_path=dest/'config.yaml'
    # Preserve environment references so the configuration remains portable.
    config_path.write_text(yaml.safe_dump(load_model_config(args.model, portable=True),sort_keys=False))
    runtime=dest/'runtime.pt'
    torch.save({'model_state':state,'global_step':step},runtime)
    check=torch.load(runtime,map_location='cpu',weights_only=True,mmap=True)['model_state']
    for key in state:
        if not torch.equal(check[key],state[key]):raise RuntimeError('Serialization mismatch')
    spec={'id':args.model_id,'model_family':args.model,'backbone':'dinov3','global_step':step,
          'checkpoint_name':args.checkpoint.name,'source_sha256':sha256(args.checkpoint),
          'checkpoint':str(runtime.relative_to(SETTINGS.assets_dir)),
          'checkpoint_sha256':sha256(runtime),'config':str(config_path.relative_to(SETTINGS.assets_dir)),
          'config_sha256':sha256(config_path),'encoder_layers':params['encoder_cfg']['params']['num_layers'],
          'decoder_layers':params['decoder_cfg']['params']['decoder_layers'],
          'q_dim':params['encoder_cfg']['params']['q_dim'],'num_heads':params['encoder_cfg']['params']['num_heads'],
          'image_size':cfg['data']['image_size'],'cache_length':81,'amp_dtype':'bf16',
          'verified':True,'tensor_count':len(state),
          'license_note':'Weights are external; this manifest grants no model license.',
          'prepared_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (dest/'manifest.json').write_text(json.dumps(spec,indent=2))
    if args.activate:
        pointer=SETTINGS.assets_dir/'active_model.json'
        if pointer.exists():shutil.copyfile(pointer,dest/'previous_active_model.json')
        temp=pointer.with_suffix('.next');temp.write_text(json.dumps(spec,indent=2));temp.replace(pointer)
    print(json.dumps({'model_id':args.model_id,'step':step,'tensors':len(state),'activated_for_next_start':args.activate},indent=2))


if __name__=='__main__':main()
