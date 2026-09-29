from __future__ import annotations
import argparse, shutil, json, math, os, re, time, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from metric_scale import metric_scale_from_static
SRC = DST = STATUS = None
RX90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
PAT = re.compile('^(.*)__y(\\d{3})$')

def ry(deg):
    r = math.radians(deg)
    c, s = (math.cos(r), math.sin(r))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

def parse_paths():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    source, output = (args.source_root.resolve(), args.output_root.resolve())
    if source == output or source in output.parents or output in source.parents:
        parser.error('Source and output must be separate directory trees')
    output.parent.mkdir(parents=True, exist_ok=True)
    return (source, output, output.with_name(output.name + '.status.json'))

def link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)

def write_status(d):
    t = STATUS.with_suffix('.tmp')
    t.write_text(json.dumps(d, indent=2) + '\n')
    os.replace(t, STATUS)

def effective_camera(cam, view):
    A = RX90 @ ry(view)
    At = A.T
    loc = At @ np.asarray(cam['location'], dtype=np.float64)
    right = At @ np.asarray(cam['right'], dtype=np.float64)
    up = At @ np.asarray(cam['up'], dtype=np.float64)
    forward = At @ np.asarray(cam['forward'], dtype=np.float64)
    R = np.stack([right, -up, forward], axis=1)
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = loc
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ loc
    out = {k: v for k, v in cam.items() if k not in ('blender_camera_location', 'location', 'right', 'up', 'forward', 'coordinate_system')}
    out.update({'location': loc.tolist(), 'right': right.tolist(), 'up': up.tolist(), 'forward': forward.tolist(), 'location_units': 'meters', 'camera_coordinate_system': 'opencv_right_handed_ydown_zforward_meters', 'world_coordinate_system': 'normalized_source_bvh_world_meters', 'camera_to_world': c2w.tolist(), 'world_to_camera': w2c.tolist(), 'source_view_degrees': int(view), 'recovery_formula': 'p_world = location + right*x_camera - up*y_camera + forward*z_camera'})
    return out

def build():
    global SRC, DST, STATUS
    SRC, DST, STATUS = parse_paths()
    if DST.exists():
        raise SystemExit(f'refuse existing target: {DST}')
    groups = defaultdict(list)
    for d in SRC.iterdir():
        if not d.is_dir():
            continue
        m = PAT.match(d.name)
        if m and (d / 'meta.json').is_file():
            groups[m.group(1)].append((int(m.group(2)), d))
    if not groups:
        raise RuntimeError('No view caches found')
    bad = {k: sorted((v for v, _ in xs)) for k, xs in groups.items() if sorted((v for v, _ in xs)) != [0, 15, 30, 45, 60, 75, 90, 135, 180, 225, 270, 315]}
    if bad:
        raise RuntimeError(f'non-12-view groups: {list(bad.items())[:10]}')
    DST.mkdir()
    t0 = time.time()
    results = []
    write_status({'status': 'building', 'source': str(SRC), 'destination': str(DST), 'groups': len(groups), 'completed': 0, 'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    for gi, (base, items) in enumerate(sorted(groups.items()), 1):
        items = sorted(items)
        out = DST / base
        (out / 'images').mkdir(parents=True)
        metas = [json.loads((d / 'meta.json').read_text()) for _, d in items]
        statics = [np.load(d / 'static.npy', allow_pickle=True).item() for _, d in items]
        s0 = statics[0]
        names = [str(x) for x in s0['joint_names']]
        parents = np.asarray(s0['parents'])
        J = len(names)
        for (view, d), m, s in zip(items, metas, statics):
            if [str(x) for x in s['joint_names']] != names or not np.array_equal(np.asarray(s['parents']), parents):
                raise RuntimeError(f'{base} skeleton mismatch y{view}')
            if not np.allclose(np.asarray(s['rest_translations'], float), np.asarray(s0['rest_translations'], float), atol=1e-07):
                raise RuntimeError(f'{base} rest translations mismatch y{view}')
        total = sum((int(m['total_frames']) for m in metas))
        rot_out = np.lib.format.open_memmap(out / 'rot6d.npy', mode='w+', dtype=np.float32, shape=(total, J, 6))
        pos_out = np.lib.format.open_memmap(out / 'world_pos.npy', mode='w+', dtype=np.float32, shape=(total, J, 3))
        root_out = np.lib.format.open_memmap(out / 'canonical_root_pos.npy', mode='w+', dtype=np.float32, shape=(total, 3))
        offset = 0
        motions = []
        shard_index = 0
        for (view, d), m in zip(items, metas):
            n = int(m['total_frames'])
            rot = np.load(d / 'rot6d.npy', mmap_mode='r')
            pos = np.load(d / 'world_pos.npy', mmap_mode='r')
            rp = np.load(d / 'canonical_root_pos.npy', mmap_mode='r')
            rot_out[offset:offset + n] = rot
            pos_out[offset:offset + n] = pos
            root_out[offset:offset + n] = rp
            old_to_new = {}
            for old_tar in sorted((d / 'images').glob('*.tar')):
                new_rel = f'images/shard-{shard_index:06d}.tar'
                link_or_copy(old_tar, out / new_rel)
                old_to_new[old_tar.relative_to(d).as_posix()] = new_rel
                shard_index += 1
            for x in m['motions']:
                y = json.loads(json.dumps(x))
                y['motion_index'] = len(motions)
                y['offset'] = offset + int(x['offset'])
                y['camera_folder'] = f"{x.get('source_motion', x['clip_name'])}\\y{view:03d}"
                y['camera'] = effective_camera(x['camera'], view)
                y['image']['tar'] = old_to_new[x['image']['tar']]
                for key in ('bvh', 'source_bvh', 'source_video'):
                    y.pop(key, None)
                y['source_view_degrees'] = view
                motions.append(y)
            offset += n
        del rot_out, pos_out, root_out
        static = dict(s0)
        static['skeleton_id'] = base
        static.pop('view', None)
        static['views'] = [v for v, _ in items]
        rest_q = np.asarray(static['rest_rotations_quat'], dtype=np.float32).copy()
        rest_q[0] = Rotation.from_matrix(RX90).as_quat().astype(np.float32)
        static['rest_rotations_quat'] = rest_q
        static['metric_scale'] = metric_scale_from_static(static)
        np.save(out / 'static.npy', static, allow_pickle=True)
        meta = {'coordinate_system': 'opencv_right_handed_ydown_zforward_meters', 'units': 'meters', 'skeleton_id': base, 'species': metas[0].get('species'), 'source_character_folder': metas[0].get('source_character_folder'), 'static_file': 'static.npy', 'skeleton_hash': s0['skeleton_hash'], 'joints': J, 'joint_names': names, 'source_joint_names': [str(x) for x in s0.get('source_joint_names', names)], 'views': [v for v, _ in items], 'total_frames': total, 'array_shapes': {'rot6d': [total, J, 6], 'world_pos': [total, J, 3], 'canonical_root_pos': [total, 3]}, 'array_dtypes': {'rot6d': 'float32', 'world_pos': 'float32', 'canonical_root_pos': 'float32'}, 'array_files': {'rot6d': 'rot6d.npy', 'world_pos': 'world_pos.npy', 'canonical_root_pos': 'canonical_root_pos.npy'}, 'image_shard_count': shard_index, 'motions': motions}
        (out / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
        results.append({'cache_id': base, 'views': len(items), 'motions': len(motions), 'frames': total, 'joints': J, 'image_tars': shard_index})
        if gi % 10 == 0 or gi == len(groups):
            write_status({'status': 'building', 'source': str(SRC), 'destination': str(DST), 'groups': len(groups), 'completed': gi, 'last_cache': base, 'elapsed_seconds': time.time() - t0})
    manifest = {'status': 'complete', 'cache_root': str(DST), 'source_cache': str(SRC), 'coordinate_contract': {'motion_arrays': 'per-motion OpenCV camera coordinates in meters', 'camera_extrinsics': 'each motion stores location/right/up/forward and camera_to_world/world_to_camera for exact recovery to normalized source BVH world meters', 'static': 'one canonical reset skeleton per skeleton instance; independent of camera view'}, 'counts': {'cache_dirs': len(results), 'skeleton_instances': len(results), 'views_per_skeleton': 12, 'motion_views': sum((x['motions'] for x in results)), 'frames': sum((x['frames'] for x in results)), 'image_tars': sum((x['image_tars'] for x in results))}, 'per_cache_files': ['meta.json', 'static.npy', 'rot6d.npy', 'world_pos.npy', 'canonical_root_pos.npy', 'images/shard-*.tar'], 'built_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'seconds': time.time() - t0}
    (DST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    write_status({'status': 'complete', **manifest})
    print(json.dumps(manifest, indent=2))
if __name__ == '__main__':
    try:
        build()
    except Exception as e:
        if STATUS is not None:
            write_status({'status': 'failed', 'error': repr(e), 'traceback': traceback.format_exc()})
        raise
