import argparse, hashlib, json, os, sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector, Matrix

COORD='opencv_right_handed_ydown_zforward_meters'

def clean_matrix(m):
 m=np.array(m,dtype=np.float64); R=m[:3,:3]; u,s,v=np.linalg.svd(R); out=np.eye(4); out[:3,:3]=u@v; out[:3,3]=m[:3,3]
 if np.linalg.det(out[:3,:3])<0: raise ValueError('reflected bone transform')
 if s.max()/s.min()>1.001: raise ValueError('nonuniform bone scaling unsupported')
 return out

def import_clip(p):
 bpy.ops.wm.read_factory_settings(use_empty=True)
 bpy.ops.import_scene.fbx(filepath=str(p),automatic_bone_orientation=False,ignore_leaf_bones=False,use_anim=True)
 rigs=[o for o in bpy.context.scene.objects if o.type=='ARMATURE']; meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
 if len(rigs)!=1 or not meshes: raise ValueError(f'{p}: expected one rig and meshes')
 rig=rigs[0]; roots=[b for b in rig.data.bones if b.parent is None]
 if len(roots)!=1: raise ValueError('multiple skeleton roots')
 bones=[]
 def visit(b):
  bones.append(b)
  for c in b.children: visit(c)
 visit(roots[0]); names=[b.name for b in bones]; parents=np.array([names.index(b.parent.name) if b.parent else -1 for b in bones])
 if not rig.animation_data or not rig.animation_data.action: raise ValueError('missing rig animation')
 action=rig.animation_data.action; start,end=map(float,action.frame_range)
 if abs(start-round(start))+abs(end-round(end))>1e-3: raise ValueError('fractional endpoints require explicit time policy')
 scene=bpy.context.scene; fps=scene.render.fps/scene.render.fps_base
 for im in bpy.data.images:
  if im.size[0]==0: raise ValueError('missing texture '+im.name)
 rest=np.array([clean_matrix(rig.matrix_world@b.matrix_local) for b in bones])
 return rig,meshes,names,parents,rest,int(round(start)),int(round(end)),fps

def bounds(meshes):
 dg=bpy.context.evaluated_depsgraph_get()
 return np.array([tuple(e.matrix_world@Vector(v)) for o in meshes for e in [o.evaluated_get(dg)] for v in e.bound_box])

def local(G,parents):
 return np.stack([G[j] if p<0 else np.linalg.inv(G[p])@G[j] for j,p in enumerate(parents)])

def r6(R):return np.concatenate([R[..., :,0],R[..., :,1]],axis=-1).astype('float32')

def build_character(root,out,name):
 files=sorted((root/name).glob('*.fbx')); target=out/name; target.mkdir(parents=True,exist_ok=True)
 baseline=None; clips=[]; all_bounds=[]
 for p in files:
  rig,meshes,names,parents,rest,start,end,fps=import_clip(p)
  if baseline is None: baseline=(names,parents,rest)
  elif names!=baseline[0] or not np.array_equal(parents,baseline[1]) or not np.allclose(rest,baseline[2],atol=2e-5):raise ValueError('different rest rig across animations '+str(p))
  G=[]; frame_bounds=[]
  for f in range(start,end+1):
   bpy.context.scene.frame_set(f); bpy.context.view_layer.update()
   G.append([clean_matrix(rig.matrix_world@rig.pose.bones[n].matrix) for n in names]); frame_bounds.append(bounds(meshes))
  rig.data.pose_position='REST'; bpy.context.view_layer.update(); all_bounds.append(bounds(meshes)); rig.data.pose_position='POSE'
  all_bounds.extend(frame_bounds)
  clips.append(dict(path=p,clip_name=p.stem.replace(' ','_'),start=start,end=end,fps=fps,G=np.array(G),bounds=np.array(frame_bounds)))
 names,parents,rest=baseline; points=np.concatenate(all_bounds); lo=points.min(0); hi=points.max(0); center=(lo+hi)/2; extent=float(max(hi-lo))
 az,el=np.radians([15.,18.]); back=np.array([np.sin(az)*np.cos(el),-np.cos(az)*np.cos(el),np.sin(el)])
 forward=-back; right=np.cross(forward,[0.,0.,1.]); right/=np.linalg.norm(right); down=np.cross(forward,right)
 R=np.stack([right,down,forward]); assert np.allclose(R@R.T,np.eye(3),atol=1e-7) and np.linalg.det(R)>.999
 projected=(points-center)@R.T; tan=36/(2*50); distance=float(np.max(np.max(np.abs(projected[:,:2]),axis=1)/(.82*tan)-projected[:,2]))
 cam_pos=center+back*distance; C=np.eye(4); C[:3,:3]=R; C[:3,3]=-R@cam_pos
 rest_cv=C@rest; Lrest=local(rest_cv,parents); diagonal=float(np.linalg.norm(np.ptp(rest_cv[:,:3,3],axis=0))); metric_scale=2/diagonal
 skeleton_hash=hashlib.sha256(json.dumps(names).encode()+parents.tobytes()+rest.astype('float32').tobytes()).hexdigest()
 quats=[]
 for m in Lrest:
  q=Matrix(m[:3,:3].tolist()).to_quaternion(); quats.append([q.x,q.y,q.z,q.w])
 static=dict(skeleton_id=name,joints=len(names),joint_names=names,parents=parents,rest_translations=Lrest[:,:3,3].astype('float32'),rest_rotations_quat=np.array(quats,dtype='float32'),metric_scale=np.float32(metric_scale),coordinate_system=COORD,units='meters',skeleton_hash=skeleton_hash,valid_joint_mask=np.ones(len(names),bool))
 np.save(target/'static.npy',static); np.savez(target/'bind_transforms.npz',world_rest=rest,camera_from_world=C,parents=parents,joint_names=np.array(names),rest_local=Lrest)
 arrays={k:[] for k in ['rot6d','world_pos','canonical_root_pos']}; motions=[]; offset=0; residuals=[]
 for c in clips:
  centers=(c['bounds'].min(1)+c['bounds'].max(1))/2
  projected=np.einsum('ij,tkj->tki',C[:3,:3],c['bounds']-centers[:,None])
  clip_distance=float(np.max(np.max(np.abs(projected[:,:,:2]),axis=2)/(.82*tan)-projected[:,:,2]))
  locations=centers+back*clip_distance; frame_C=np.repeat(C[None],len(centers),axis=0);frame_C[:,:3,3]=-locations@C[:3,:3].T
  G=frame_C[:,None]@c['G']; locals_=np.stack([local(g,parents) for g in G]); R=locals_[...,:3,:3]; P=G[...,:3,3]
  fk=np.zeros_like(P); gr=np.zeros_like(R)
  for j,p in enumerate(parents):
   if p<0: fk[:,j]=P[:,j]; gr[:,j]=R[:,j]
   else: fk[:,j]=fk[:,p]+np.einsum('tij,j->ti',gr[:,p],Lrest[j,:3,3]); gr[:,j]=gr[:,p]@R[:,j]
  fk_error=float(np.max(np.linalg.norm(fk-P,axis=-1))); residuals.append(fk_error)
  if fk_error>0.001: raise ValueError(f'fixed-offset FK error exceeds 1mm {name}/{c["clip_name"]}: {fk_error}')
  arrays['rot6d'].append(r6(R)); arrays['world_pos'].append(P.astype('float32')); arrays['canonical_root_pos'].append(P[:,0].astype('float32'))
  motion=dict(clip_name=c['clip_name'],offset=offset,frames=len(G),fps=c['fps'],camera='front',camera_folder='front',motion_index=len(motions),image=dict(tar='images/front.tar',rgb_ext='.jpg'),original_frame_start=c['start'],original_frame_end=c['end'],source_fbx=str(c['path']),fixed_offset_fk_max_error_m=fk_error)
  motion.update(camera_centers=centers.tolist(),camera_locations=locations.tolist(),camera_distance=clip_distance)
  motions.append(motion); offset+=len(G)
  rawgt=target/'reference'; rawgt.mkdir(exist_ok=True)
  np.savez_compressed(rawgt/(c['clip_name']+'.npz'),global_transforms=G,local_transforms=locals_,camera_from_world=frame_C,fps=c['fps'],frame_indices=np.arange(c['start'],c['end']+1),joint_names=np.array(names),parents=parents)
 for k,v in arrays.items():np.save(target/(k+'.npy'),np.concatenate(v))
 camera=dict(resolution=512,lens_mm=50,sensor_width_mm=36,fx=512*50/36,fy=512*50/36,cx=256,cy=256,world_to_camera=C.tolist(),location=cam_pos.tolist(),center=center.tolist(),extent=extent,azimuth_deg=15,elevation_deg=18,distance=distance,framing='full-animation projected bounds, 9 percent margin each side')
 meta=dict(dataset='Mixamo',format='merged_per_rig',skeleton_id=name,joints=len(names),joint_names=names,total_frames=offset,coordinate_system=COORD,units='meters',array_files={k:k+'.npy' for k in arrays},motions=motions,camera=camera,metric_scale_definition='2 / rest skeleton AABB diagonal in meters',reference_type='Mixamo auto-retargeted reference, not captured ground truth',split='evaluation_only')
 meta['lighting']=dict(light_power_multiplier=0.55,world_strength=0.385)
 (target/'meta.json').write_text(json.dumps(meta,indent=2))
 # Save an animation-free target mesh with original bind and textures.
 rig,meshes,*_=import_clip(files[0]); rig.animation_data_clear()
 for b in rig.pose.bones:b.matrix_basis=Matrix.Identity(4)
 for act in list(bpy.data.actions):bpy.data.actions.remove(act)
 bpy.context.view_layer.update(); bpy.ops.file.pack_all(); bpy.ops.wm.save_as_mainfile(filepath=str(target/'target_rest.blend'))
 print('EXPORTED',name,offset,len(names),'FK_MAX',max(residuals),flush=True)
 return meta

def render_clip(root,out,name,clip,preview=False):
 meta=json.loads((out/name/'meta.json').read_text()); m=next(x for x in meta['motions'] if x['clip_name']==clip); cam=meta['camera']
 rig,meshes,names,parents,rest,start,end,fps=import_clip(Path(m['source_fbx']))
 scene=bpy.context.scene; scene.render.engine='BLENDER_EEVEE'; scene.eevee.taa_render_samples=16
 scene.render.resolution_x=scene.render.resolution_y=512; scene.render.resolution_percentage=100
 scene.render.film_transparent=True; scene.render.image_settings.file_format='PNG'; scene.render.image_settings.color_mode='RGBA'
 scene.world=bpy.data.worlds.new('World'); scene.world.use_nodes=True; scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.17,.17,.17,1); scene.world.node_tree.nodes['Background'].inputs[1].default_value=.385
 center=Vector(cam['center']); extent=cam['extent']; bpy.ops.object.camera_add(location=cam['location']); camera=bpy.context.object; camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler(); camera.data.lens=50; camera.data.sensor_width=36; camera.data.sensor_fit='HORIZONTAL'; camera.data.clip_start=.01; camera.data.clip_end=1000; scene.camera=camera
 for delta,power,size in [((-.8,-1.2,1.5),80,1),((1,-.7,.6),52,1),((0,1,1.4),80,.8)]:
  bpy.ops.object.light_add(type='AREA',location=center+Vector(delta)*extent); light=bpy.context.object; light.data.energy=0.55*power*extent**2; light.data.shape='DISK'; light.data.size=size*extent; light.rotation_euler=(center-light.location).to_track_quat('-Z','Y').to_euler()
 scene.view_settings.view_transform='Standard'; scene.view_settings.look='Medium High Contrast'
 dest=out/name/('preview' if preview else 'rgba')/clip; dest.mkdir(parents=True,exist_ok=True)
 bpy.context.view_layer.update()
 assert np.max(np.abs(np.diag([1.,-1.,-1.,1.])@np.array(camera.matrix_world.inverted())-np.array(cam['world_to_camera'])))<1e-4
 for idx,f in enumerate(range(start,end+1)):
  path=dest/(f'{idx:06d}.png')
  if preview and idx not in {0,(end-start+1)//2,end-start}:continue
  if path.exists():continue
  camera.location=m['camera_locations'][idx]
  scene.frame_set(f); scene.render.filepath=str(path); bpy.ops.render.render(write_still=True)
 (dest/'DONE').write_text(str(end-start+1)); print('RENDERED',name,clip,end-start+1,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--character',required=True);ap.add_argument('--clip');ap.add_argument('--preview',action='store_true');ap.add_argument('--all',action='store_true');args=ap.parse_args(sys.argv[sys.argv.index('--')+1:])
 if args.clip:render_clip(args.root,args.out,args.character,args.clip,args.preview)
 else:
  meta=build_character(args.root,args.out,args.character)
  if args.all:
   for m in meta['motions']:render_clip(args.root,args.out,args.character,m['clip_name'],args.preview)
