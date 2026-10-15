"""Read FBX axis/unit declarations and bind matrices using Blender's FBX parser.

FBX FrontAxis is the opposite of the forward direction used by Blender's
RIGHT_HAND_AXES table. Bone-local bases must be converted along with world
bases; changing only the world frame leaves parent-local rotations unchanged.
"""
from __future__ import annotations

import hashlib
import numpy as np


def fbx_source_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _child(element, name):
    return next((e for e in element.elems if e.id == name), None)


def _name(value):
    return value.decode('utf-8', errors='strict').split('\x00\x01')[0]


def read_fbx_coordinates(path):
    # The worker runs in Blender, which ships the binary FBX parser.
    from io_scene_fbx import parse_fbx
    from bpy_extras.io_utils import axis_conversion

    tree, version = parse_fbx.parse(str(path))
    settings = _child(tree, b'GlobalSettings')
    properties = _child(settings, b'Properties70') if settings else None
    if properties is None:
        raise ValueError('FBX has no GlobalSettings/Properties70 axis declaration')
    values = {_name(p.props[0]): p.props[4] for p in properties.elems
              if p.id == b'P' and len(p.props) > 4}
    keys = ('UpAxis', 'UpAxisSign', 'FrontAxis', 'FrontAxisSign', 'CoordAxis', 'CoordAxisSign')
    missing = [k for k in (*keys, 'UnitScaleFactor') if k not in values]
    if missing:
        raise ValueError(f'FBX axis/unit declaration is incomplete: {missing}')
    axes = {k: int(values[k]) for k in keys}
    for k in ('UpAxis', 'FrontAxis', 'CoordAxis'):
        if axes[k] not in (0, 1, 2) or axes[k+'Sign'] not in (-1, 1):
            raise ValueError(f'Invalid FBX axis declaration: {axes}')
    if len({axes[k] for k in ('UpAxis', 'FrontAxis', 'CoordAxis')}) != 3:
        raise ValueError(f'FBX axes are not distinct: {axes}')
    meters_per_unit = float(values['UnitScaleFactor']) * .01
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError('Invalid FBX UnitScaleFactor')
    unit = np.eye(3)
    up = unit[axes['UpAxis']] * axes['UpAxisSign']
    forward = unit[axes['FrontAxis']] * -axes['FrontAxisSign']
    right = unit[axes['CoordAxis']] * axes['CoordAxisSign']
    source_to_cv = np.stack([right, -up, forward])
    assert np.array_equal(source_to_cv @ source_to_cv.T, np.eye(3))
    def axis_string(vector):
        i = int(np.argmax(np.abs(vector)))
        return ('-' if vector[i] < 0 else '') + 'XYZ'[i]
    up_string, forward_string = axis_string(up), axis_string(forward)
    # Explicit import settings avoid Blender's fallback for left-handed files.
    source_to_blender = np.asarray(axis_conversion(from_up=up_string,
                                                   from_forward=forward_string).to_4x4(), dtype=np.float64)
    source_to_blender[:3, :3] *= meters_per_unit
    objects = _child(tree, b'Objects')
    if objects is None:
        raise ValueError('FBX has no Objects section')
    models = {int(e.props[0]): _name(e.props[1]) for e in objects.elems if e.id == b'Model'}
    bone_models = {int(e.props[0]) for e in objects.elems
                   if e.id == b'Model' and e.props[2] in (b'LimbNode', b'Root', b'Null')}
    matrices = {}
    sources = {}
    def add(uid, array, source):
        uid = int(uid)
        matrix = np.asarray(array, dtype=np.float64).reshape(4, 4).T
        if uid in matrices and not np.allclose(matrices[uid], matrix, rtol=1e-5, atol=1e-5):
            raise ValueError(f'Conflicting FBX bind matrices for {models.get(uid, uid)}')
        matrices[uid] = matrix
        sources.setdefault(uid, []).append(source)
    for e in objects.elems:
        if e.id == b'Pose' and len(e.props) >= 3 and e.props[2] == b'BindPose':
            for node in e.elems:
                if node.id != b'PoseNode':
                    continue
                uid, matrix = _child(node, b'Node'), _child(node, b'Matrix')
                if uid and matrix:
                    add(uid.props[0], matrix.props[0], 'BindPose')
    clusters = {}
    for e in objects.elems:
        if e.id == b'Deformer' and len(e.props) >= 3 and e.props[2] == b'Cluster':
            matrix = _child(e, b'TransformLink')
            if matrix:
                clusters[int(e.props[0])] = matrix.props[0]
    connections = _child(tree, b'Connections')
    if connections:
        for c in connections.elems:
            if c.id == b'C' and len(c.props) >= 3 and c.props[0] == b'OO':
                child, parent = map(int, c.props[1:3])
                if child in models and parent in clusters:
                    add(child, clusters[parent], 'SkinCluster.TransformLink')
    by_name = {}
    source_by_name = {}
    for uid, matrix in matrices.items():
        if uid not in bone_models:
            continue
        name = models[uid]
        if name in by_name:
            raise ValueError(f'Ambiguous FBX bind bone name: {name}')
        by_name[name] = matrix
        source_by_name[name] = sorted(set(sources[uid]))
    metadata = dict(fbx_version=version, **axes, UnitScaleFactor=float(values['UnitScaleFactor']),
                    meters_per_source_unit=meters_per_unit,
                    source_handedness='right' if np.linalg.det(source_to_cv) > 0 else 'left',
                    blender_import_up=up_string, blender_import_forward=forward_string,
                    source_to_opencv_basis=source_to_cv.tolist(),
                    source_to_blender=source_to_blender.tolist(),
                    bone_local_basis_conversion='R_target = A @ R_source @ A.T; t_target = A @ t_source')
    return metadata, by_name, source_by_name
