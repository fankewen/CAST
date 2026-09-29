from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, sys, tarfile, threading, time, traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from metric_scale import metric_scale_from_static
from render_official_full_frames import canonical_asset_id
CATS = ('aquatic', 'avian', 'biped', 'hexapod', 'octopod', 'others', 'quadruped', 'serpentine')
SCALE = 0.5
COORD = 'opencv_right_handed_ydown_zforward_meters'

def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--render-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--only-asset', action='append', default=[])
    p.add_argument('--reject-assets', type=Path, default=Path(__file__).with_name('quality_rejections.json'))
    return p.parse_args()

def atomic(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name('.' + p.name + '.tmp')
    q.write_text(json.dumps(x, indent=2, ensure_ascii=False) + '\n')
    os.replace(q, p)

def unwrap(x):
    return x.item() if isinstance(x, np.ndarray) and x.shape == () and (x.dtype == object) else x

def entries(src, rejection_path=None):
    rejection_path = rejection_path or Path(__file__).with_name('quality_rejections.json')
    q = json.loads(rejection_path.read_text())
    reject = {str(x['asset_id']) for x in q['rejections']}
    out = []
    seen = set()
    for split in ('train', 'seen', 'unseen'):
        for cat in CATS:
            p = src / 'datalist' / ('train' if split == 'train' else 'validate') / (f'{cat}_128.txt' if split == 'train' else f'{cat}_{split}_128.txt')
            if not p.is_file():
                continue
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                a = canonical_asset_id(line)
                if a in seen:
                    raise RuntimeError(f'duplicate {a}')
                seen.add(a)
                if a not in reject:
                    out.append({'asset_id': a, 'source_split': split, 'category': cat, 'topology_regime': 'train' if split == 'train' else split})
    if not seen:
        raise RuntimeError('Empty source inventory')
    q = {**q, 'rejected_asset_count': len(reject & seen)}
    return (out, q)

def r6mat(x):
    x = np.asarray(x, np.float64)
    a = x[..., :3]
    b = x[..., 3:]
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    return np.stack((a, b, np.cross(a, b)), axis=-1)

def mat6(r):
    return np.concatenate((r[..., :, 0], r[..., :, 1]), axis=-1).astype(np.float32)

def matquat(r):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(np.asarray(r).reshape(-1, 3, 3)).as_quat()
    return q.reshape(np.asarray(r).shape[:-2] + (4,)).astype(np.float32)

def gl2cv(c):
    x = np.asarray(c, np.float64).copy()
    x[:3, 1] *= -1
    x[:3, 2] *= -1
    return x

def scalet(m):
    x = np.asarray(m, np.float64).copy()
    x[..., :3, 3] *= SCALE
    return x.astype(np.float32)

def load_asset(src, a, Asset):
    with np.load(src / 'mobjaverse' / a / 'raw_data.npz', allow_pickle=True) as z:
        v = np.asarray(z['vertices'])
        p = np.asarray(z['parents'], np.int32)
        ml = np.asarray(z['matrix_local']).copy()
        mb = np.asarray(z['matrix_basis']).copy()
        raw_names = unwrap(z['joint_names']) if 'joint_names' in z.files else None
    names = [str(n) for n in list(raw_names)] if raw_names is not None else [f'joint_{j:03d}' for j in range(len(p))]
    x = Asset(vertices=v, parents=p, matrix_local=ml, matrix_basis=mb, joint_names=names)
    if not len(p) or p[0] != -1 or any(not 0 <= int(parent) < j for j, parent in enumerate(p[1:], 1)):
        raise ValueError(f'{a} non-topological skeleton')
    x.matrix_basis[:, 0, :3, 3] = 0
    x.normalize_vertices((-1.0, 1.0))
    return x

def rest(asset):
    J = len(asset.parents)
    local = np.empty((J, 4, 4), np.float64)
    for j, p in enumerate(asset.parents):
        local[j] = asset.matrix_local[j] if p < 0 else np.linalg.inv(asset.matrix_local[p]) @ asset.matrix_local[j]
    off = local[:, :3, 3].astype(np.float32) * SCALE
    rot = local[:, :3, :3]
    off[0] = 0
    rot[0] = np.eye(3)
    return (off, mat6(rot))

def static(entry, src, rend, Asset):
    a = entry['asset_id']
    m = json.loads((rend / 'assets' / a[:2] / a / 'complete.json').read_text())
    if m.get('status') != 'complete' or m.get('source_quality', {}).get('zero_skin'):
        raise ValueError(f'{a} is not a valid completed animated render')
    x = load_asset(src, a, Asset)
    F = len(x.matrix_basis)
    if F != m['source_frames'] or F != m['rendered_frames']:
        raise ValueError(f'{a} frame mismatch')
    off, rr = rest(x)
    p = np.asarray(x.parents, np.int32)
    names = [str(n) for n in x.joint_names]
    h = hashlib.sha256()
    h.update(p.tobytes())
    h.update(off.tobytes())
    h.update(rr[1:].tobytes())
    h.update(json.dumps(names, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    return {**entry, 'parents': p, 'offsets': off, 'rest_rot6d': rr, 'joint_names': names, 'key': h.hexdigest(), 'joints': len(p), 'frames': F, 'resolution': int(m['render']['resolution'])}

def camera(marker):
    c2s = gl2cv(np.asarray(marker['camera']['camera_to_world'], np.float64))
    s2c = np.linalg.inv(c2s)
    n = int(marker['render']['resolution'])
    f = n / (2 * math.tan(float(marker['camera']['yfov_radians']) / 2))
    return (c2s, s2c, np.array([f, f, n / 2, n / 2], np.float32))

def motion(asset, s2c):
    g = asset.get_matrix(asset.matrix_basis)
    R = g[:, :, :3, :3]
    L = np.empty_like(R)
    L[:, 0] = s2c[:3, :3] @ R[:, 0]
    for j in range(1, len(asset.parents)):
        L[:, j] = np.swapaxes(R[:, int(asset.parents[j])], -1, -2) @ R[:, j]
    p = g[:, :, :3, 3] @ s2c[:3, :3].T + s2c[:3, 3]
    return (mat6(L), (p * SCALE).astype(np.float32))

def copytar(path, dst, a, F):
    exp = {f'{a}.view00.{i:06d}{e}' for i in range(F) for e in ('.jpg', '.png')}
    seen = set()
    with tarfile.open(path, 'r') as t:
        for m in t:
            if m.name not in exp or m.name in seen:
                raise ValueError(f'{a} bad tar member {m.name}')
            seen.add(m.name)
            fp = t.extractfile(m)
            if fp is None:
                raise ValueError(f'{a} unreadable {m.name}')
            dst.addfile(m, fp)
    if seen != exp:
        raise ValueError(f'{a} tar count {len(seen)} != {len(exp)}')

def group(gid, items, src, rend, out, Asset):
    rep = items[0]
    J = rep['joints']
    T = sum((x['frames'] for x in items))
    tmp = out / '.tmp' / f'{gid}.{os.getpid()}.{threading.get_ident()}'
    final = out / gid
    if tmp.exists():
        shutil.rmtree(tmp)
    if final.exists():
        raise RuntimeError(f'existing {final}')
    (tmp / 'images').mkdir(parents=True)
    arrays = {'rot6d': np.lib.format.open_memmap(tmp / 'rot6d.npy', mode='w+', dtype=np.float32, shape=(T, J, 6)), 'world_pos': np.lib.format.open_memmap(tmp / 'world_pos.npy', mode='w+', dtype=np.float32, shape=(T, J, 3)), 'canonical_root_pos': np.lib.format.open_memmap(tmp / 'canonical_root_pos.npy', mode='w+', dtype=np.float32, shape=(T, 3))}
    motions = []
    offset = 0
    with tarfile.open(tmp / 'images' / 'shard-000000.tar', 'w', format=tarfile.GNU_FORMAT) as dst:
        for mi, item in enumerate(items):
            a = item['asset_id']
            d = rend / 'assets' / a[:2] / a
            marker = json.loads((d / 'complete.json').read_text())
            asset = load_asset(src, a, Asset)
            F = item['frames']
            c2s, s2c, K = camera(marker)
            rr, pp = motion(asset, s2c)
            end = offset + F
            if rr.shape != (F, J, 6) or pp.shape != (F, J, 3):
                raise ValueError(f'{a} motion shape')
            if not np.isfinite(rr).all() or not np.isfinite(pp).all():
                raise ValueError(f'{a} non-finite motion')
            if int(marker['render']['resolution']) != rep['resolution']:
                raise ValueError('All assets in a skeleton cache must use the same resolution')
            arrays['rot6d'][offset:end] = rr
            arrays['world_pos'][offset:end] = pp
            arrays['canonical_root_pos'][offset:end] = pp[:, 0]
            copytar(d / 'view_00.tar', dst, a, F)
            C = scalet(c2s)
            n = int(marker['render']['resolution'])
            cam = {'fx': float(K[0]), 'fy': float(K[1]), 'cx': float(K[2]), 'cy': float(K[3]), 'width': n, 'height': n, 'location': C[:3, 3].tolist(), 'right': C[:3, 0].tolist(), 'up': (-C[:3, 1]).tolist(), 'forward': C[:3, 2].tolist(), 'location_units': 'meters', 'camera_coordinate_system': COORD, 'camera_to_world': C.tolist(), 'world_to_camera': scalet(s2c).tolist(), 'camera_to_source': C.tolist(), 'source_to_camera': scalet(s2c).tolist(), 'camera_is_static_for_motion': True, 'recovery_formula': 'p_source = camera_to_source @ p_camera'}
            motions.append({'camera': cam, 'camera_folder': f'{a}/view00', 'clip_name': f'{a}.view00', 'fps': 30.0, 'frames': F, 'image': {'frame_offset_in_tar': 0, 'tar': 'images/shard-000000.tar'}, 'images': {'image_count': F, 'image_glob': f'{a}.view00.*.jpg', 'images_dir': 'images', 'mask_count': F, 'mask_glob': f'{a}.view00.*.png', 'masks_dir': 'images', 'resolution': [n, n]}, 'motion_index': mi, 'offset': offset, 'source_motion': a, 'source_split': item['source_split'], 'category': item['category'], 'topology_regime': item['topology_regime']})
            offset = end
    for x in arrays.values():
        x.flush()
    del x, arrays
    names = list(rep['joint_names'])
    static_obj = {'coordinate_system': COORD, 'units': 'meters', 'skeleton_id': gid, 'species': 'Mobjaverse', 'joints': np.asarray(J), 'skeleton_hash': rep['key'][:12], 'joint_names': np.asarray(names, dtype=object), 'source_joint_names': np.asarray(names, dtype=object), 'parents': rep['parents'], 'rest_translations': rep['offsets'], 'rest_rotations_quat': matquat(r6mat(rep['rest_rot6d'])), 'valid_joint_mask': np.ones(J, bool), 'image_height': np.asarray(rep['resolution']), 'image_width': np.asarray(rep['resolution']), 'source_character_folder': np.asarray('Mobjaverse'), 'views': np.asarray([0]), 'skeleton_dedup_tolerance_m': np.asarray(0.0), 'skeleton_dedup_representative_cache': np.asarray(rep['asset_id']), 'skeleton_dedup_source_caches': np.asarray([x['asset_id'] for x in items])}
    static_obj['metric_scale'] = metric_scale_from_static(static_obj)
    np.save(tmp / 'static.npy', static_obj, allow_pickle=True)
    meta = {'coordinate_system': COORD, 'units': 'meters', 'skeleton_id': gid, 'joints': J, 'joint_names': names, 'source_joint_names': names, 'skeleton_hash': rep['key'][:12], 'total_frames': T, 'static_file': 'static.npy', 'array_files': {'rot6d': 'rot6d.npy', 'world_pos': 'world_pos.npy', 'canonical_root_pos': 'canonical_root_pos.npy'}, 'array_shapes': {'rot6d': [T, J, 6], 'world_pos': [T, J, 3], 'canonical_root_pos': [T, 3]}, 'array_dtypes': {'rot6d': 'float32', 'world_pos': 'float32', 'canonical_root_pos': 'float32'}, 'image_shard_count': 1, 'skeleton_dedup': {'criterion': 'bit_exact_joint_names_parents_offsets_nonroot_rest_rotations', 'tolerance_m': 0.0, 'representative_cache': rep['asset_id'], 'source_caches': [x['asset_id'] for x in items]}, 'motions': motions}
    atomic(tmp / 'meta.json', meta)
    os.replace(tmp, final)
    return {'assets': len(items), 'frames': T}

def main():
    a = cli()
    a.source_root = a.source_root.resolve()
    a.render_root = a.render_root.resolve()
    a.output_root = a.output_root.resolve()
    if a.output_root.exists():
        raise RuntimeError(f'refuse existing {a.output_root}')
    sys.path.insert(0, str(a.source_root))
    from src.rig_package.info.asset import Asset
    a.output_root.mkdir(parents=True)
    (a.output_root / '.tmp').mkdir()
    start = time.time()
    es, quality = entries(a.source_root, a.reject_assets)
    if a.only_asset:
        want = {canonical_asset_id(x) for x in a.only_asset}
        es = [x for x in es if x['asset_id'] in want]
        if want != {x['asset_id'] for x in es}:
            raise RuntimeError('requested rejected/missing asset')
    if not es:
        raise ValueError('No usable assets after applying the manual rejection list')
    atomic(a.output_root / 'BUILD_STATUS.json', {'status': 'inventory', 'assets': len(es)})
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        rows = list(ex.map(lambda x: static(x, a.source_root, a.render_root, Asset), es, chunksize=8))
    groups = defaultdict(list)
    for x in rows:
        groups[x['key']].append(x)
    ordered = sorted((sorted(v, key=lambda x: x['asset_id']) for v in groups.values()), key=lambda v: v[0]['asset_id'])
    inv = {'usable_assets': len(rows), 'cache_dirs': len(ordered), 'merged_assets': len(rows) - len(ordered), 'frames': sum((x['frames'] for x in rows)), 'quality_rejections': quality['rejected_asset_count'], 'criterion': 'bit-exact joint names, parents, normalized metric local offsets, and non-root rest-local rotations; root translation/rotation canonicalized'}
    print(json.dumps(inv, indent=2), flush=True)
    completed = []
    fail = []
    atomic(a.output_root / 'BUILD_STATUS.json', {**inv, 'status': 'building', 'completed': 0})
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        fut = {ex.submit(group, f'mobj_skeleton_{i:06d}', g, a.source_root, a.render_root, a.output_root, Asset): (i, g) for i, g in enumerate(ordered)}
        for done, f in enumerate(as_completed(fut), 1):
            try:
                completed.append(f.result())
            except Exception as e:
                i, g = fut[f]
                fail.append({'index': i, 'assets': [x['asset_id'] for x in g], 'error': repr(e), 'traceback': traceback.format_exc()})
            if done % 100 == 0 or done == len(ordered):
                print(f'progress {done}/{len(ordered)} failed={len(fail)}', flush=True)
                atomic(a.output_root / 'BUILD_STATUS.json', {**inv, 'status': 'building', 'completed': done, 'failed': len(fail), 'seconds': time.time() - start})
    A = sum((x['assets'] for x in completed))
    F = sum((x['frames'] for x in completed))
    status = 'complete' if not fail and A == len(rows) and (F == inv['frames']) else 'failed'
    manifest = {'status': status, 'cache_root': str(a.output_root), 'source_root': str(a.source_root), 'source_render_cache': str(a.render_root), 'coordinate_system': COORD, 'units': 'meters', 'temporal_policy': {'truncate': False, 'pad_on_disk': False, 'repeat_tail': False, 'resample': False}, 'counts': {'source_assets': A, 'rejected_source_assets': quality['rejected_asset_count'], 'cache_dirs': len(completed), 'merged_assets': A - len(completed), 'motions': A, 'frames': F, 'image_tars': len(completed)}, 'skeleton_criterion': inv['criterion'], 'camera_recovery': 'one static camera extrinsic/intrinsic record per motion in meta.json', 'quality_rejections_source': str(a.reject_assets.resolve()), 'failures': fail, 'seconds': time.time() - start, 'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    atomic(a.output_root / 'BUILD_STATUS.json', manifest)
    if status == 'complete':
        (a.output_root / 'CACHE_READY').write_text('complete\n')
        shutil.rmtree(a.output_root / '.tmp', ignore_errors=True)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'failures'}, indent=2), flush=True)
    return 0 if status == 'complete' else 2
if __name__ == '__main__':
    raise SystemExit(main())

