from __future__ import annotations
import argparse, concurrent.futures, hashlib, io, json, math, os, shutil, sys, tarfile, time, traceback
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation
from metric_scale import metric_scale_from_static
ROOT = MASK_ROOT = VERT_MANIFEST = OFFICIAL = None
VIEWS = [0, 15, 30, 45, 60, 75, 90, 135, 180, 225, 270, 315]
TARGET_SHARD_BYTES = 256 * 1024 * 1024
CAMERA_ROT_X = 1.4783514738082886
CAMERA_R_WORLD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0923132374882698, -0.9957300424575806], [0.0, 0.9957300424575806, 0.0923132374882698]], dtype=np.float32)
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
RX90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--mask-root', type=Path, required=True)
    p.add_argument('--vertices-root', type=Path, required=True)
    p.add_argument('--third-party-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--jpeg-quality', type=int, default=95)
    p.add_argument('--pilot-motion')
    p.add_argument('--pilot-view', type=int, default=0)
    return p.parse_args()

def atomic_json(path: Path, obj: Any):
    tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(tmp, path)

def add_bytes(tf: tarfile.TarFile, name: str, data: bytes):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 420
    tf.addfile(info, io.BytesIO(data))

def ry(deg: int):
    r = math.radians(deg)
    c, s = (math.cos(r), math.sin(r))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

def matrix_to_rot6d(m):
    m = np.asarray(m, dtype=np.float32)
    return np.concatenate((m[..., :, 0], m[..., :, 1]), axis=-1).astype(np.float32)

def safe_id(text: str):
    out = ''.join((c if c.isalnum() or c in '-_' else '_' for c in text))
    return out[:120]

def camera_params(species: str):
    if species == 'Crow':
        view_scale, object_position = (0.25, 1.5)
    elif species == 'Jaws':
        view_scale, object_position = (0.5, 1.0)
    else:
        view_scale, object_position = (1.3, 1.0)
    loc = np.array([0.0, -3.8 / view_scale, object_position], dtype=np.float32)
    fx = fy = 1000.0
    cx = cy = 360.0
    return (view_scale, object_position, loc, fx, fy, cx, cy)

def load_bvh(path: Path):
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    from utils import bvh as BVH
    from utils import animation
    anim, names, frametime = BVH.load(str(path))
    lengths = np.linalg.norm(anim.offsets, axis=1)
    adjacency = [[] for _ in anim.parents]
    for joint, parent in enumerate(anim.parents):
        if int(parent) >= 0:
            adjacency[joint].append((int(parent), float(lengths[joint])))
            adjacency[int(parent)].append((joint, float(lengths[joint])))
    diameter = 0.0
    for source in range(len(adjacency)):
        stack = [(source, -1, 0.0)]
        while stack:
            node, previous, distance = stack.pop()
            diameter = max(diameter, distance)
            for next_node, length in adjacency[node]:
                if next_node != previous:
                    stack.append((next_node, node, distance + length))
    local = animation.transforms_local(anim).astype(np.float32)
    global_ = animation.transforms_global(anim).astype(np.float32)
    return (anim, names, float(frametime), float(diameter), local, global_)

def skeleton_signature(names, parents, offsets):
    h = hashlib.sha256()
    h.update('\n'.join(map(str, names)).encode())
    h.update(np.asarray(parents, dtype=np.int32).tobytes())
    h.update(np.round(np.asarray(offsets, dtype=np.float32), 6).tobytes())
    return h.hexdigest()[:12]

def task_cache_id(species: str, char_folder: str, view: int):
    p = Path(char_folder)
    if p.parent.name == 'characters_fix_facezplus':
        base = species
    else:
        base = f'{species}__motion_{hashlib.sha1(p.name.encode()).hexdigest()[:10]}'
    return f'{safe_id(base)}__y{view:03d}'

def build_one(task: dict):
    import cv2
    global ROOT, MASK_ROOT, OFFICIAL
    ROOT, MASK_ROOT, OFFICIAL = (Path(task[k]) for k in ('source_root', 'mask_root', 'third_party_root'))
    t0 = time.time()
    out_root = Path(task['output_root'])
    cache_id = task['cache_id']
    final = out_root / cache_id
    try:
        if final.exists():
            raise FileExistsError(final)
        tmp = out_root / '.tmp' / f'{cache_id}.{os.getpid()}'
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        (tmp / 'images').mkdir()
        species = task['species']
        view = int(task['view'])
        view_scale, obj_pos, cam_loc, fx, fy, cx, cy = camera_params(species)
        Rsrc = BLENDER_TO_OPENCV @ CAMERA_R_WORLD.T @ RX90 @ ry(view)
        tcam = BLENDER_TO_OPENCV @ CAMERA_R_WORLD.T @ -cam_loc
        arrays_rot = []
        arrays_pos = []
        arrays_root = []
        motions_meta = []
        offset = 0
        static_data = None
        skel_hash = None
        joint_names = None
        parents = None
        scale0 = None
        shard_idx = 0
        shard_frames = 0
        shard_paths = []
        tf = None
        tf_path = None

        def open_shard():
            nonlocal tf, tf_path, shard_idx, shard_frames
            if tf is not None:
                tf.close()
            tf_path = tmp / 'images' / f'shard-{shard_idx:06d}.tar'
            tf = tarfile.open(tf_path, 'w', format=tarfile.PAX_FORMAT)
            shard_paths.append(tf_path)
            shard_frames = 0
            shard_idx += 1
        open_shard()
        for mi, motion in enumerate(task['motions']):
            if mi and tf.fileobj.tell() >= int(task['target_shard_bytes']):
                open_shard()
            bvh = ROOT / 'bvh' / motion / 'y0.bvh'
            anim, names, frametime, diameter, local, global_ = load_bvh(bvh)
            frames = int(global_.shape[0])
            if frames <= 0 or not np.isfinite(diameter) or diameter <= 0:
                raise ValueError(f'Invalid motion {motion}')
            scale = 1.0 / diameter
            if static_data is None:
                joint_names = [str(x) for x in names]
                parents = np.asarray(anim.parents, dtype=np.int32)
                skel_hash = skeleton_signature(joint_names, parents, np.asarray(anim.offsets) * scale)
                scale0 = scale
                rest_t = np.asarray(anim.offsets, dtype=np.float32) * scale
                rest_t[0] = 0
                rest_R = np.broadcast_to(np.eye(3, dtype=np.float32), (len(names), 3, 3)).copy()
                rest_R[0] = Rsrc
                rest_q = Rotation.from_matrix(rest_R).as_quat().astype(np.float32)
                static_data = {'coordinate_system': 'opencv_right_handed_ydown_zforward_meters', 'units': 'meters', 'skeleton_id': cache_id, 'species': species, 'view': view, 'joints': np.asarray(len(names)), 'skeleton_hash': skel_hash, 'joint_names': np.asarray(joint_names, dtype=object), 'source_joint_names': np.asarray(joint_names, dtype=object), 'parents': parents, 'rest_translations': rest_t, 'rest_rotations_quat': rest_q, 'metric_scale': np.asarray(scale, dtype=np.float32), 'valid_joint_mask': np.ones(len(names), dtype=np.bool_), 'image_height': np.asarray(720), 'image_width': np.asarray(720), 'source_character_folder': task['character_folder']}
            else:
                if list(map(str, names)) != joint_names or not np.array_equal(np.asarray(anim.parents, dtype=np.int32), parents):
                    raise ValueError(f'skeleton mismatch in {motion}')
                current_rest = np.asarray(anim.offsets, dtype=np.float32) * scale
                current_rest[0] = 0
                if not np.allclose(current_rest, static_data['rest_translations'], rtol=0, atol=1e-07):
                    raise ValueError(f'rest offsets mismatch in {motion}')
                if abs(scale - scale0) > 1e-05:
                    raise ValueError(f'scale mismatch in {motion}: {scale} vs {scale0}')
            src_pos = global_[:, :, :3, 3] * scale
            cam_pos = np.einsum('ij,tkj->tki', Rsrc, src_pos) + tcam[None, None, :]
            local_R = local[:, :, :3, :3].copy()
            root_src_R = global_[:, 0, :3, :3]
            local_R[:, 0] = np.einsum('ij,tjk->tik', Rsrc, root_src_R)
            rot6d = matrix_to_rot6d(local_R)
            arrays_rot.append(rot6d)
            arrays_pos.append(cam_pos.astype(np.float32))
            arrays_root.append(cam_pos[:, 0].astype(np.float32))
            clip = safe_id(motion) + '_' + hashlib.sha1(motion.encode()).hexdigest()[:10]
            video = ROOT / 'video' / motion / f'y{view}.mp4'
            mask_dir = MASK_ROOT / motion / f'y{view}'
            cap = cv2.VideoCapture(str(video))
            current_rel = f'images/{tf_path.name}'
            frame_offset = shard_frames
            for frame_index in range(frames):
                _, frame = cap.read()
                _, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(task['jpeg_quality'])])
                mask_bytes = (mask_dir / f'{frame_index:05d}.png').read_bytes()
                stem = f'{clip}.{frame_index:06d}'
                add_bytes(tf, stem + '.jpg', jpg.tobytes())
                add_bytes(tf, stem + '.png', mask_bytes)
                shard_frames += 1
            cap.release()
            motions_meta.append({'motion_index': mi, 'clip_name': clip, 'source_motion': motion, 'offset': offset, 'frames': frames, 'fps': 1.0 / frametime, 'camera_folder': f'{motion}\\y{view}', 'camera': {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'width': 720, 'height': 720, 'view_degrees': view, 'view_scale': view_scale, 'object_position': obj_pos, 'location': cam_loc.tolist(), 'right': [1.0, 0.0, 0.0], 'up': [0.0, 0.0923132374882698, 0.9957300424575806], 'forward': [0.0, 0.9957300424575806, -0.0923132374882698], 'coordinate_system': 'opencv_right_handed_ydown_zforward_meters'}, 'images': {'images_dir': 'images', 'masks_dir': 'images', 'image_glob': f'{clip}.*.jpg', 'mask_glob': f'{clip}.*.png', 'image_count': frames, 'mask_count': frames, 'resolution': [720, 720]}, 'image': {'tar': current_rel, 'frame_offset_in_tar': frame_offset}})
            offset += frames
        if tf is not None:
            tf.close()
        rot = np.concatenate(arrays_rot, axis=0).astype(np.float32)
        pos = np.concatenate(arrays_pos, axis=0).astype(np.float32)
        rootp = np.concatenate(arrays_root, axis=0).astype(np.float32)
        np.save(tmp / 'rot6d.npy', rot)
        np.save(tmp / 'world_pos.npy', pos)
        np.save(tmp / 'canonical_root_pos.npy', rootp)
        static_data['metric_scale'] = metric_scale_from_static(static_data)
        np.save(tmp / 'static.npy', static_data, allow_pickle=True)
        meta = {'coordinate_system': 'opencv_right_handed_ydown_zforward_meters', 'units': 'meters', 'skeleton_id': cache_id, 'species': species, 'view': view, 'source_character_folder': task['character_folder'], 'static_file': 'static.npy', 'skeleton_hash': skel_hash, 'joints': len(joint_names), 'joint_names': joint_names, 'source_joint_names': joint_names, 'total_frames': int(offset), 'array_shapes': {'rot6d': list(rot.shape), 'world_pos': list(pos.shape), 'canonical_root_pos': list(rootp.shape)}, 'array_dtypes': {'rot6d': 'float32', 'world_pos': 'float32', 'canonical_root_pos': 'float32'}, 'array_files': {'rot6d': 'rot6d.npy', 'world_pos': 'world_pos.npy', 'canonical_root_pos': 'canonical_root_pos.npy'}, 'image_shard_count': len(shard_paths), 'motions': motions_meta}
        atomic_json(tmp / 'meta.json', meta)
        os.replace(tmp, final)
        return {'ok': True, 'cache_id': cache_id, 'species': species, 'view': view, 'motions': len(motions_meta), 'frames': int(offset), 'joints': len(joint_names), 'image_shards': len(shard_paths), 'seconds': time.time() - t0, 'path': str(final)}
    except Exception as e:
        return {'ok': False, 'cache_id': cache_id, 'error': repr(e), 'traceback': traceback.format_exc(), 'seconds': time.time() - t0}

def main():
    a = parse_args()
    if a.workers < 1 or not 1 <= a.jpeg_quality <= 100:
        raise ValueError('Invalid workers or JPEG quality')
    global ROOT, MASK_ROOT, OFFICIAL, VERT_MANIFEST
    ROOT, MASK_ROOT, OFFICIAL = (p.resolve() for p in (a.source_root, a.mask_root, a.third_party_root))
    VERT_MANIFEST = a.vertices_root.resolve() / 'manifest.json'
    out = a.output_root.resolve()
    if out.exists():
        raise FileExistsError(out)
    man = json.loads(VERT_MANIFEST.read_text())
    jobs = man['jobs']
    if man.get('status') != 'complete':
        raise ValueError('Vertex extraction is incomplete')
    if a.pilot_motion:
        jobs = [j for j in jobs if j['motion'] == a.pilot_motion]
        views = [a.pilot_view]
    else:
        views = VIEWS
    groups = defaultdict(list)
    char_for = {}
    species_for = {}
    for j in jobs:
        motion = j['motion']
        species = motion.split('#', 1)[0]
        char = j['character_folder']
        groups[species, char].append(motion)
        char_for[species, char] = char
        species_for[species, char] = species
    tasks = []
    for (species, char), motions in sorted(groups.items()):
        for view in views:
            tasks.append({'output_root': str(out), 'cache_id': task_cache_id(species, char, view), 'species': species, 'character_folder': char, 'view': view, 'motions': sorted(motions), 'jpeg_quality': a.jpeg_quality, 'target_shard_bytes': TARGET_SHARD_BYTES, 'source_root': str(ROOT), 'mask_root': str(MASK_ROOT), 'third_party_root': str(OFFICIAL)})
    if not tasks:
        raise ValueError('No matching motions')
    out.mkdir(parents=True)
    status_path = out / 'build_status.json'
    atomic_json(status_path, {'status': 'running', 'tasks': len(tasks), 'completed': 0, 'failed': 0, 'started_at': time.time()})
    results = []
    t0 = time.time()
    workers = min(a.workers, len(tasks))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(build_one, tasks, chunksize=1), 1):
            results.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != 'traceback'}), flush=True)
            if i % 5 == 0 or not r['ok']:
                atomic_json(status_path, {'status': 'running', 'tasks': len(tasks), 'completed': i, 'passed': sum((x['ok'] for x in results)), 'failed': sum((not x['ok'] for x in results)), 'frames': sum((x.get('frames', 0) for x in results)), 'seconds': time.time() - t0, 'last': {k: v for k, v in r.items() if k != 'traceback'}})
    fail = [x for x in results if not x['ok']]
    summary = {'status': 'complete' if not fail else 'failed', 'tasks': len(tasks), 'passed': len(tasks) - len(fail), 'failed': len(fail), 'cache_dirs': len(results), 'motions': sum((x.get('motions', 0) for x in results if x['ok'])), 'frames': sum((x.get('frames', 0) for x in results if x['ok'])), 'seconds': time.time() - t0, 'source_mask_root': str(MASK_ROOT), 'source_video_root': str(ROOT / 'video'), 'failures': fail}
    atomic_json(out / 'manifest.json', summary)
    atomic_json(status_path, {k: v for k, v in summary.items() if k != 'failures'})
    if fail:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
