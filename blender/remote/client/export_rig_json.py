"""Blender: select an armature, set scene.camera; run with -- --output rig.json."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import bpy
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cast_remote_live import build_rig_asset

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
args=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
rig=bpy.context.object
if rig is None or rig.type!='ARMATURE':raise RuntimeError('Select the target armature')
camera=bpy.context.scene.camera
if camera is None:raise RuntimeError('Set an active scene camera before exporting')
args.output.parent.mkdir(parents=True,exist_ok=True)
asset=build_rig_asset(rig,camera,str(args.output.parent))
data=np.load(asset['static'],allow_pickle=True).item() # locally created trusted file
serial={k:v.tolist() if isinstance(v,(np.ndarray,np.generic)) else v for k,v in data.items()}
args.output.write_text(json.dumps(serial,indent=2))
print('Exported JSON rig; keep bind_transforms.npz locally for RigDriver:',args.output)
