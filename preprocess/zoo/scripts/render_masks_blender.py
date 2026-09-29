import argparse, json, math, os, shutil, sys, time
from pathlib import Path
import bpy, numpy as np
from mathutils import Vector
VIEWS = [0, 15, 30, 45, 60, 75, 90, 135, 180, 225, 270, 315]

def argv():
    return sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []

def import_obj(path):
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=path)
    new = [o for o in bpy.data.objects if o.name not in before]
    if not new:
        raise RuntimeError(path)
    return new[0]

def setup(scene):
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    try:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except:
        scene.render.engine = 'BLENDER_EEVEE'
    scene.render.use_persistent_data = True
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'BW'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.compression = 100
    if hasattr(scene, 'eevee'):
        scene.eevee.taa_render_samples = 1
    if hasattr(scene, 'eevee_next'):
        scene.eevee_next.taa_render_samples = 1
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.render.dither_intensity = 0
    w = scene.world
    w.use_nodes = True
    bg = next((n for n in w.node_tree.nodes if n.type == 'BACKGROUND'), None)
    if bg:
        bg.inputs['Color'].default_value = (0, 0, 0, 1)
        bg.inputs['Strength'].default_value = 0
    mat = bpy.data.materials.new('MASK_WHITE')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    em = nodes.new('ShaderNodeEmission')
    em.inputs['Color'].default_value = (1, 1, 1, 1)
    em.inputs['Strength'].default_value = 10
    links.new(em.outputs['Emission'], out.inputs['Surface'])
    return mat

def load_character(folder, old, mat):
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
    obj = import_obj(os.path.join(folder, 'base_mesh.obj'))
    obj.rotation_euler = (math.radians(90), 0, 0)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj

def rot(deg):
    r = math.radians(deg)
    c, s = (math.cos(r), math.sin(r))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--jobs-json', required=True)
    p.add_argument('--scene', required=True)
    p.add_argument('--mask-root', required=True)
    p.add_argument('--worker-id', required=True)
    a = p.parse_args(argv())
    jobs = json.loads(Path(a.jobs_json).read_text())
    bpy.ops.wm.open_mainfile(filepath=a.scene)
    scene = bpy.context.scene
    mat = setup(scene)
    cam = bpy.data.objects.get('Camera')
    if cam is None:
        raise ValueError('The bundled blank.blend must contain Camera')
    scene.camera = cam
    obj = None
    char_now = None
    total = done = 0
    t0 = time.time()
    for ji, j in enumerate(jobs, 1):
        if j['character_folder'] != char_now:
            obj = load_character(j['character_folder'], obj, mat)
            char_now = j['character_folder']
        verts = np.load(j['path'], mmap_mode='r')
        frames = int(verts.shape[0])
        cam.location = Vector((0.0, -3.8 / float(j['view_scale']), float(j['object_position'])))
        for deg in VIEWS:
            total += 1
            out = Path(a.mask_root) / j['motion'] / f'y{deg}'
            tmp = out.with_name(out.name + '.tmp_mask')
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True, exist_ok=True)
            R = rot(deg)
            for fi in range(frames):
                vv = np.asarray(verts[fi]) if deg == 0 else np.asarray(verts[fi]) @ R.T
                obj.data.vertices.foreach_set('co', vv.ravel())
                obj.data.update()
                scene.render.filepath = str(tmp / f'{fi:05d}.png')
                bpy.ops.render.render(write_still=True)
            os.replace(tmp, out)
            done += 1
        print(f"[MASK_WORKER {a.worker_id}] {ji}/{len(jobs)} {j['motion']} frames={frames} views_done={done}/{total}", flush=True)
    print('MASK_WORKER_DONE', a.worker_id, len(jobs), done, time.time() - t0, flush=True)
if __name__ == '__main__':
    main()
