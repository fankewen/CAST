from __future__ import annotations
import argparse, json, os, shutil, time, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from metric_scale import metric_scale_from_static
SRC = DST = STATUS = None
TOL = 1e-05
TOPOLOGY_ONLY_MOTIONS = frozenset({'Skunk#walk'})

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

def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)

def identity_fk(offsets, parents):
    p = np.zeros_like(offsets, dtype=np.float64)
    for j, parent in enumerate(parents):
        p[j] = offsets[j] if int(parent) < 0 else p[int(parent)] + offsets[j]
    return p

def load_item(d):
    s = np.load(d / 'static.npy', allow_pickle=True).item()
    m = json.loads((d / 'meta.json').read_text())
    names = tuple(map(str, s['joint_names']))
    parents = np.asarray(s['parents'], np.int32)
    offsets = np.asarray(s['rest_translations'], np.float64)
    lengths = np.linalg.norm(offsets, axis=1)
    shape = identity_fk(offsets, parents)
    rest_q = np.asarray(s['rest_rotations_quat'], np.float64)
    valid = np.asarray(s.get('valid_joint_mask', np.ones(len(names), bool)), bool)
    motions = frozenset(str(x['source_motion']) for x in m['motions'])
    return {'path': d, 'cache': d.name, 'species': str(m.get('species', d.name.split('__', 1)[0])), 'static': s, 'meta': m, 'names': names, 'parents': parents, 'offsets': offsets, 'lengths': lengths, 'shape': shape, 'rest_q': rest_q, 'valid': valid, 'motions': motions}

def errors(a, b):
    return {'offset': float(np.max(np.abs(a['offsets'] - b['offsets']))), 'length': float(np.max(np.abs(a['lengths'] - b['lengths']))), 'shape': float(np.max(np.abs(a['shape'] - b['shape']))), 'rest_q': float(np.max(np.abs(a['rest_q'] - b['rest_q'])))}

def same(a, b):
    if a['species'] != b['species'] or a['names'] != b['names']:
        return False
    if not np.array_equal(a['parents'], b['parents']) or not np.array_equal(a['valid'], b['valid']):
        return False
    if a['species'] == 'Skunk' and (a['motions'] == TOPOLOGY_ONLY_MOTIONS or b['motions'] == TOPOLOGY_ONLY_MOTIONS):
        return True
    e = errors(a, b)
    return e['offset'] <= TOL and e['length'] <= TOL and (e['shape'] <= TOL * len(a['parents'])) and (e['rest_q'] <= TOL)

def cluster_complete(items):
    clusters = []
    for x in sorted(items, key=lambda z: z['cache']):
        for c in clusters:
            if all((same(x, y) for y in c)):
                c.append(x)
                break
        else:
            clusters.append([x])
    return clusters

def representative(cluster):
    best = None
    for x in cluster:
        score = max((max(errors(x, y)['offset'], errors(x, y)['shape']) for y in cluster))
        key = (score, x['cache'])
        if best is None or key < best[0]:
            best = (key, x)
    return best[1]

def main():
    global SRC, DST, STATUS
    SRC, DST, STATUS = parse_paths()
    if DST.exists():
        raise RuntimeError(f'refuse existing destination: {DST}')
    items = [load_item(d) for d in sorted(SRC.iterdir()) if d.is_dir() and (d / 'static.npy').is_file() and (d / 'meta.json').is_file()]
    if not items:
        raise RuntimeError('No source caches found')
    clusters = cluster_complete(items)
    by_species = defaultdict(list)
    for c in clusters:
        by_species[c[0]['species']].append(c)
    for s in by_species:
        by_species[s].sort(key=lambda c: representative(c)['cache'])
    DST.mkdir()
    t0 = time.time()
    results = []
    done = 0
    write_json(STATUS, {'status': 'building', 'source': str(SRC), 'destination': str(DST), 'input_caches': len(items), 'target_caches': len(clusters), 'completed': 0, 'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    for species in sorted(by_species):
        species_clusters = by_species[species]
        for si, c in enumerate(species_clusters, 1):
            out_id = species if len(species_clusters) == 1 else f'{species}__skeleton_{si:02d}'
            out = DST / out_id
            (out / 'images').mkdir(parents=True)
            rep = representative(c)
            J = len(rep['names'])
            total = sum((int(x['meta']['total_frames']) for x in c))
            rot_out = np.lib.format.open_memmap(out / 'rot6d.npy', mode='w+', dtype=np.float32, shape=(total, J, 6))
            pos_out = np.lib.format.open_memmap(out / 'world_pos.npy', mode='w+', dtype=np.float32, shape=(total, J, 3))
            root_out = np.lib.format.open_memmap(out / 'canonical_root_pos.npy', mode='w+', dtype=np.float32, shape=(total, 3))
            offset = 0
            motions = []
            shard_index = 0
            source_folders = []
            for x in sorted(c, key=lambda z: z['cache']):
                d = x['path']
                m = x['meta']
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
                for mm in m['motions']:
                    y = json.loads(json.dumps(mm))
                    y['motion_index'] = len(motions)
                    y['offset'] = offset + int(mm['offset'])
                    y['image']['tar'] = old_to_new[mm['image']['tar']]
                    y['source_cache_before_skeleton_dedup'] = x['cache']
                    motions.append(y)
                source_folders.append(str(m.get('source_character_folder', '')))
                offset += n
            del rot_out, pos_out, root_out
            static = dict(rep['static'])
            static['skeleton_id'] = out_id
            static['species'] = species
            static['skeleton_dedup_tolerance_m'] = TOL
            static['skeleton_dedup_representative_cache'] = rep['cache']
            static['skeleton_dedup_source_caches'] = [x['cache'] for x in sorted(c, key=lambda z: z['cache'])]
            topology_merge = len(c) > 1 and any(x['motions'] == TOPOLOGY_ONLY_MOTIONS for x in c)
            static['metric_scale'] = metric_scale_from_static(static)
            np.save(out / 'static.npy', static, allow_pickle=True)
            meta = {'coordinate_system': 'opencv_right_handed_ydown_zforward_meters', 'units': 'meters', 'skeleton_id': out_id, 'species': species, 'source_character_folders': sorted(set(source_folders)), 'static_file': 'static.npy', 'skeleton_hash': str(static.get('skeleton_hash', '')), 'joints': J, 'joint_names': list(rep['names']), 'source_joint_names': [str(v) for v in static.get('source_joint_names', rep['names'])], 'views': sorted(set((int(v) for mm in motions for v in [mm.get('source_view_degrees', mm.get('camera', {}).get('source_view_degrees', -1))] if int(v) >= 0))), 'total_frames': total, 'array_shapes': {'rot6d': [total, J, 6], 'world_pos': [total, J, 3], 'canonical_root_pos': [total, 3]}, 'array_dtypes': {'rot6d': 'float32', 'world_pos': 'float32', 'canonical_root_pos': 'float32'}, 'array_files': {'rot6d': 'rot6d.npy', 'world_pos': 'world_pos.npy', 'canonical_root_pos': 'canonical_root_pos.npy'}, 'image_shard_count': shard_index, 'skeleton_dedup': {'criterion': 'exact joint definitions and parents; max offset/bone-length <=1e-5 m; identity-FK shape <=1e-5 m * joint_count; rest quaternion <=1e-5', 'tolerance_m': TOL, 'representative_cache': rep['cache'], 'source_caches': [x['cache'] for x in sorted(c, key=lambda z: z['cache'])]}, 'motions': motions}
            if topology_merge:
                meta['skeleton_dedup']['criterion'] = 'same species, ordered joint names, parents, and valid mask'
            write_json(out / 'meta.json', meta)
            results.append({'motions': len(motions), 'frames': total, 'image_tars': shard_index})
            done += 1
            if done % 10 == 0 or done == len(clusters):
                write_json(STATUS, {'status': 'building', 'source': str(SRC), 'destination': str(DST), 'input_caches': len(items), 'target_caches': len(clusters), 'completed': done, 'last_cache': out_id, 'elapsed_seconds': time.time() - t0})
    manifest = {'status': 'complete', 'cache_root': str(DST), 'source_cache': str(SRC), 'dedup_tolerance_m': TOL, 'criterion': 'same species, exact ordered joint definitions and parents, identical valid mask, offset and bone length <=1e-5 m, identity-FK shape <= J*1e-5 m, rest quaternion <=1e-5', 'counts': {'source_cache_dirs': len(items), 'cache_dirs': len(results), 'directory_reduction': len(items) - len(results), 'motion_views': sum((x['motions'] for x in results)), 'frames': sum((x['frames'] for x in results)), 'image_tars': sum((x['image_tars'] for x in results))}, 'built_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'seconds': time.time() - t0}
    manifest['criterion'] = 'same species and joint tree; geometry tolerance or topology-only motion rule'
    write_json(STATUS, {'status': 'complete', **manifest})
    print(json.dumps(manifest, indent=2))
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        if STATUS is not None:
            write_json(STATUS, {'status': 'failed', 'error': repr(e), 'traceback': traceback.format_exc()})
        raise
