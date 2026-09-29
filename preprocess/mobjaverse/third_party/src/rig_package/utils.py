from numpy import ndarray
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy

def assert_ndarray(arr, name: str="arr", shape: Optional[Tuple[int, ...]]=None, dtype=None):
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{name} must be a numpy.ndarray or None, got {type(arr)}")
    if shape is not None:
        # shape may contain None as wildcard
        if len(shape) != arr.ndim:
            raise ValueError(f"{name}: expected shape length {len(shape)} but array ndim is {arr.ndim}")
        for i, (exp, actual) in enumerate(zip(shape, arr.shape)):
            if exp > 0 and exp != actual:
                raise ValueError(f"{name} shape mismatch at axis {i}: expected {exp}, got {actual}")
    if dtype is not None:
        if not np.issubdtype(arr.dtype, dtype):
            raise ValueError(f"{name} dtype must be {dtype}, got {arr.dtype}")

def assert_list(arr, name: str="arr", dtype=None):
    if not isinstance(arr, list):
        raise ValueError(f"found type {type(arr)}, expect a list")
    if dtype is not None:
        for x in arr:
            if not isinstance(x, dtype):
                raise ValueError(f"found type {type(x)} in {name}, expect all to be {dtype}")

def normalize_rot(x: ndarray) -> ndarray:
    """normalize rotation in matrix"""
    try:
        x = np.asarray(x)
        assert x.shape[-2:] in [(3, 3), (4, 4)]
        is_homo = (x.shape[-2:] == (4, 4))
        y = x.copy()
        R = y[..., :3, :3]
        orig_shape = R.shape
        Rf = R.reshape(-1, 3, 3)
        U, S, Vt = np.linalg.svd(Rf)
        Rn = U @ Vt
        det = np.linalg.det(Rn)
        mask = det < 0
        if np.any(mask):
            Vt[mask, -1, :] *= -1
            Rn[mask] = U[mask] @ Vt[mask]
        y[..., :3, :3] = Rn.reshape(orig_shape)
        return y
    except Exception as e:
        print("error in normalize_rot:", str(e))
        return x

#######################################################
# WARNING: AI GENERATED CODE
def normalize(v, eps: float=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

def skew(v):
    """
    v: (..., 3)
    return: (..., 3, 3)
    """
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    O = np.zeros_like(vx)
    return np.stack([
        np.stack([ O, -vz,  vy], axis=-1),
        np.stack([ vz,  O, -vx], axis=-1),
        np.stack([-vy,  vx,  O], axis=-1),
    ], axis=-2)

def rotation_between_vectors(
    a: np.ndarray,
    b: np.ndarray,
    eps: float=1e-6,
    reference: ndarray=np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    a = normalize(a)
    b = normalize(b)
    c = np.sum(a * b, axis=-1, keepdims=True)
    v = np.cross(a, b) 
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    I = np.eye(3)
    mask_same = c > (1.0 - eps)
    mask_oppo = c < (-1.0 + eps)
    mask_general = ~(mask_same | mask_oppo)

    R = np.zeros(a.shape[:-1] + (3, 3))

    # --- same direction ---
    if np.any(mask_same):
        R[mask_same[..., 0]] = I

    # --- opposite direction ---
    if np.any(mask_oppo):
        a_op = a[mask_oppo[..., 0]]
        ref = np.broadcast_to(reference, a_op.shape)
        axis = np.cross(a_op, ref)
        bad = np.linalg.norm(axis, axis=-1) < eps
        if np.any(bad):
            alt = np.array([1.0, 0.0, 0.0])
            ref2 = np.broadcast_to(alt, a_op.shape)
            axis[bad] = np.cross(a_op[bad], ref2[bad])
        axis = normalize(axis)
        K = skew(axis)
        R_op = I + 2.0 * np.matmul(K, K)
        R[mask_oppo[..., 0]] = R_op

    # --- general case ---
    if np.any(mask_general):
        v_g = v[mask_general[..., 0]]
        c_g = c[mask_general[..., 0]]
        K = skew(v_g)
        R_g = I + K+ np.matmul(K, K) / (1.0 + c_g)[..., None]
        R[mask_general[..., 0]] = R_g
    return R

def mat4_to_dual_quaternion(M):
    R = M[:3, :3]
    t = M[:3, 3]
    qw = np.sqrt(max(1.0 + np.trace(R), 1e-8)) / 2
    qx = (R[2,1] - R[1,2]) / (4*qw+1e-8)
    qy = (R[0,2] - R[2,0]) / (4*qw+1e-8)
    qz = (R[1,0] - R[0,1]) / (4*qw+1e-8)
    q_real = np.array([qw, qx, qy, qz], dtype=np.float32)
    t_quat = np.array([0, t[0], t[1], t[2]], dtype=np.float32)
    w1, x1, y1, z1 = t_quat
    w2, x2, y2, z2 = q_real
    qd = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32) * 0.5
    return q_real, qd

def dq_apply(qr, qd, point):
    p = np.array([0, point[0], point[1], point[2]], dtype=np.float32)
    w,x,y,z = qr
    qr_conj = np.array([w, -x, -y, -z], dtype=np.float32)
    def qmul(a, b):
        aw,ax,ay,az = a
        bw,bx,by,bz = b
        return np.array([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ], dtype=np.float32)
    r = qmul(qmul(qr, p), qr_conj)
    t = qmul(qd*2.0, qr_conj)[1:]
    return r[1:] + t

def quat_mul(a, b):
    """
    a, b: (..., 4)  [w, x, y, z]
    """
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=1)

def dq_apply_batch(qr, qd, v):
    """
    qr: (N, 4) real quaternion
    qd: (N, 4) dual quaternion
    v : (N, 3)
    """
    # v as pure quaternion
    zeros = np.zeros((v.shape[0], 1), dtype=v.dtype)
    vq = np.concatenate([zeros, v], axis=1)   # (N, 4)

    # q * v * q_conj
    qr_conj = qr.copy()
    qr_conj[:, 1:] *= -1

    t = quat_mul(qd, qr_conj)
    t[:, 1:] *= 2

    v_rot = quat_mul(quat_mul(qr, vq), qr_conj)[:, 1:]
    return v_rot + t[:, 1:]

def linear_blend_skinning_dqs(
    vertices: np.ndarray,
    matrix_local: np.ndarray,
    matrix: np.ndarray,
    skin: np.ndarray,
    pad: int=1,
    value: float=1.0,
) -> ndarray:
    J = matrix_local.shape[0]
    N = vertices.shape[0]
    trans = matrix @ np.linalg.inv(matrix_local)
    dq_real = np.zeros((J,4), dtype=np.float32)
    dq_dual = np.zeros((J,4), dtype=np.float32)
    for j in range(J):
        qr, qd = mat4_to_dual_quaternion(trans[j])
        dq_real[j] = qr
        dq_dual[j] = qd
    qr = skin @ dq_real
    qd = skin @ dq_dual
    wsum = skin.sum(axis=1)
    valid = wsum > 1e-12
    norm = np.linalg.norm(qr[valid], axis=1, keepdims=True)
    qr[valid] /= norm
    qd[valid] /= norm
    out = vertices.copy()
    out[valid] = dq_apply_batch(qr[valid], qd[valid], vertices[valid])
    return out

#  AI GENERATED CODE END
#######################################################

def linear_blend_skinning(
    vertices: ndarray,
    matrix_local: ndarray,
    matrix: ndarray,
    skin: ndarray,
    pad: int=1,
    value: float=1.0,
) -> ndarray:
    """
    Args:
        vertices: (N, 4-pad)
        matrix_local: (J, 4, 4)
        matrix: (J, 4, 4)
        skin: (N, J)
        pad: 0 or 1
        value: value to pad
    Returns:
        (N, 3) vertices using LBS algorithm: Skinning with dual quaternions, Kavan, 2007
    """
    J = matrix_local.shape[0]
    N = vertices.shape[0]
    assert_ndarray(vertices, name='vertices', shape=(N, 3))
    assert_ndarray(matrix_local, name="matrix_local", shape=(J, 4, 4))
    assert_ndarray(matrix, name="matrix", shape=(J, 4, 4))
    assert_ndarray(skin, name="skin", shape=(N, J))
    assert vertices.shape[-1] + pad == 4
    
    # (4, N)
    padded = np.pad(vertices, ((0, 0), (0, pad)), 'constant', constant_values=(0, value)).T
    # (J, 4, 4)
    trans = matrix @ np.linalg.inv(matrix_local)

    # --- 核心优化部分：一行搞定 ---
    # j: Joint(骨骼数), c: row(4), k: col(4), n: Vertex(顶点数)
    # trans(J, 4, 4) -> jck
    # padded(4, N)   -> kn
    # skin(N, J)     -> nj
    # 结果输出 g(4, N) -> cn
    g = np.einsum('jck, kn, nj -> cn', trans, padded, skin, optimize=True)

    # 最终除法计算
    final = g[:3, :] / (np.sum(skin, axis=1) + 1e-8)
    return final.T
    
    # # (4, N)
    # padded = np.pad(vertices, ((0, 0), (0, pad)), 'constant', constant_values=(0, value)).T
    # # (J, 4, 4)
    # trans = matrix @ np.linalg.inv(matrix_local)
    # weighted_per_bone_matrix = []
    # # (J, N)
    # mask = (skin > 0).T
    # for i in range(J):
    #     offset = np.zeros((4, N), dtype=np.float32)
    #     offset[:, mask[i]] = (trans[i] @ padded[:, mask[i]]) * skin.T[i, mask[i]]
    #     weighted_per_bone_matrix.append(offset)
    # weighted_per_bone_matrix = np.stack(weighted_per_bone_matrix)
    # g = np.sum(weighted_per_bone_matrix, axis=0)
    # final = g[:3, :] / (np.sum(skin, axis=1) + 1e-8)
    # return final.T

def axis_angle_to_matrix(axis_angle: ndarray) -> ndarray:
    """
    Turn axis angle representation to matrix representation.
    """
    res = np.pad(scipy.spatial.transform.Rotation.from_rotvec(axis_angle).as_matrix(), ((0, 0), (0, 1), (0, 1)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
    assert res.ndim == 3
    res[:, -1, -1] = 1
    return res

def sample_surface(
    num_samples: int,
    vertices: ndarray,
    faces: ndarray,
    mask: Optional[ndarray]=None,
    face_index: Optional[ndarray]=None,
    random_lengths: Optional[ndarray]=None,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    '''
    Randomly pick samples proportional to face area.
    
    See sample_surface: https://github.com/mikedh/trimesh/blob/main/trimesh/sample.py
    
    Args:
        mask: (num_faces,), only sample points on the faces where value is True.
    Return:
        vertex_samples: sampled vertices
        
        original_face_index: on which face is sampled
        
        face_index: sampled faces
        
        random_lengths: sampled vectors on face
    '''
    original_face_indices = np.arange(len(faces))
    # sample according to mask
    if mask is not None:
        assert_ndarray(arr=mask, name="mask", shape=(faces.shape[0],))
        original_face_indices = original_face_indices[mask]
        faces = faces[mask]
    if face_index is None:
        # get face area
        offset_0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        offset_1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        face_weight = np.linalg.norm(np.cross(offset_0, offset_1, axis=-1), axis=-1)
        
        weight_cum = np.cumsum(face_weight, axis=0)
        face_pick = np.random.rand(num_samples) * weight_cum[-1]
        _face_index = np.searchsorted(weight_cum, face_pick)
    else:
        _face_index = face_index
    # map face_index back to original indices
    original_face_index = original_face_indices[_face_index]
    
    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vertices[faces[:, 0]]
    tri_vectors = vertices[faces[:, 1:]]
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[_face_index]
    tri_vectors = tri_vectors[_face_index]
    
    if random_lengths is None:
        # randomly generate two 0-1 scalar components to multiply edge vectors b
        random_lengths = np.random.rand(len(tri_vectors), 2, 1)
    
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)
    
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    vertex_samples = sample_vector + tri_origins
    return vertex_samples, original_face_index, _face_index, random_lengths

def sample_barycentric(
    vertex_group: ndarray,
    faces: ndarray,
    face_index: ndarray,
    random_lengths: ndarray,
) -> ndarray:
    v_origins = vertex_group[faces[face_index, 0]]
    v_vectors = vertex_group[faces[face_index, 1:]]
    v_vectors -= v_origins[:, np.newaxis, :]
    
    sample_vector = (v_vectors * random_lengths).sum(axis=1)
    v_samples = sample_vector + v_origins
    return v_samples

def sample_vertex_groups(
    vertices: ndarray,
    faces: ndarray,
    num_samples: int,
    num_vertex_samples: Optional[int]=None,
    vertex_normals: Optional[ndarray]=None,
    face_normals: Optional[ndarray]=None,
    vertex_groups: Optional[ndarray]=None,
    face_mask: Optional[ndarray]=None,
    deterministic_params: Optional[Dict[str, ndarray]]=None,
) -> Tuple[ndarray, ndarray|None, ndarray|None, Dict[str, ndarray]]:
    """
    Choose num_samples samples on the mesh and get their positions and normals.
    If vertex_group is provided, get its weights using barycentric sampling.
    
    Return:
        sampled_vertices, sampled_normals, sampled_vertex_groups, deterministic_params
    
    Args:
        vertices: (N, 3)
        
        faces: (F, 3)
        
        num_samples: how many samples
        
        num_vertex_samples:
            At most num_vertex_samples unique vertices to be included,
            these points will be concatenated in the last (if shuffle is False).
        
        vertex_normals: (N, 3), sampled_normals will be None if not provided
        
        face_normals: (N, 3), sampled_normals will be None if not provided
        
        vertex_groups: (N, m), sampled_vertex_groups will be None if not provided
        
        face_mask:
            (F,) or (F, m), if shape is (F,), use the same mask across all
            vertex groups. Only sample on faces where value is True.
        
        deterministic_params:
            A dict of parameters to be used directly instead of random sampling.
    """
    
    if num_vertex_samples is None:
        num_vertex_samples = 0
    if num_vertex_samples > num_samples:
        raise ValueError(f"num_vertex_samples cannot be larger than num_samples, found: {num_vertex_samples} > {num_samples}")
    
    def get_mask_perm(mask: Optional[ndarray]):
        if mask is None:
            vertex_mask = np.arange(vertices.shape[0])
        else:
            vertex_mask = np.unique(mask)
        perm = np.random.permutation(vertex_mask.shape[0])
        return vertex_mask[perm[:num_vertex_samples]]
    
    if vertex_groups is not None:
        if vertex_groups.ndim == 1:
            assert_ndarray(arr=vertex_groups, name="vertex_groups", shape=(vertices.shape[0],))
            vertex_groups = vertex_groups[:, None]
        else:
            assert_ndarray(arr=vertex_groups, name="vertex_groups", shape=(vertices.shape[0], -1))
            vertex_groups = vertex_groups
    
    if vertex_groups is not None:
        if face_mask is not None:
            assert_ndarray(arr=face_mask, name="mask", shape=(faces.shape[0],))
        perm = None
        _mask = None
        if deterministic_params is not None:
            perm = deterministic_params['perm']
            origin_face_index = deterministic_params['original_face_index']
            face_index = deterministic_params['face_index']
            random_lengths = deterministic_params['random_lengths']
            _num_samples = num_samples - len(perm)
            face_vertices, origin_face_index, face_index, random_lengths = sample_surface(
                num_samples=_num_samples,
                vertices=vertices,
                faces=faces,
                mask=_mask,
                face_index=face_index,
                random_lengths=random_lengths,
            )
        else:
            if face_mask is not None:
                assert face_mask.ndim == 1
                perm = get_mask_perm(faces[face_mask])
                _mask = face_mask
            else:
                perm = get_mask_perm(None)
                _mask = None
            _num_samples = num_samples - len(perm)
            
            face_vertices, origin_face_index, face_index, random_lengths = sample_surface(
                num_samples=_num_samples,
                vertices=vertices,
                faces=faces,
                mask=_mask,
            )
        
        sampled_vertices = np.concatenate([vertices[perm], face_vertices], axis=0)
        if vertex_normals is not None and face_normals is not None:
            sampled_normals = np.concatenate([vertex_normals[perm], face_normals[origin_face_index]], axis=0)
        else:
            sampled_normals = None
        
        g = sample_barycentric(
            vertex_group=vertex_groups,
            faces=faces,
            face_index=face_index,
            random_lengths=random_lengths,
        )
        sampled_vertex_groups = np.concatenate([vertex_groups[perm], g], axis=0)
        
    else: # otherwise only sample vertices and normals
        if deterministic_params is not None:
            perm = deterministic_params['perm']
            face_index = deterministic_params['face_index']
            origin_face_index = deterministic_params['original_face_index']
            random_lengths = deterministic_params['random_lengths']
            num_samples -= len(perm)
            face_vertices, origin_face_index, face_index, random_lengths = sample_surface(
                num_samples=num_samples,
                vertices=vertices,
                faces=faces,
                mask=face_mask,
                face_index=face_index,
                random_lengths=random_lengths,
            )
        else:
            if face_mask is not None:
                assert_ndarray(arr=face_mask, name="mask", shape=(faces.shape[0],))
                perm = get_mask_perm(faces[face_mask])
            else:
                perm = get_mask_perm(None)
            num_samples -= len(perm)
            face_vertices, origin_face_index, face_index, random_lengths = sample_surface(
                num_samples=num_samples,
                vertices=vertices,
                faces=faces,
                mask=face_mask,
            )
        n_vertex = vertices[perm]
        sampled_vertices = np.concatenate([n_vertex, face_vertices], axis=0)
        if vertex_normals is not None and face_normals is not None:
            sampled_normals = np.concatenate([vertex_normals[perm], face_normals[origin_face_index]], axis=0)
        else:
            sampled_normals = None
        sampled_vertex_groups = None
    d = {
        "perm": perm,
        "original_face_index": origin_face_index,
        "face_index": face_index,
        "random_lengths": random_lengths,
    }
    return sampled_vertices, sampled_normals, sampled_vertex_groups, d

def get_matrix_basis(
    matrix: ndarray,
    matrix_local: ndarray,
    parents: List[int]|ndarray,
    dfs_order: Optional[List[int]]=None,
) -> ndarray:
    """
    Solve matrix_basis given matrix, matrix_world and matrix_local.
    """
    J = matrix_local.shape[0]
    assert matrix_local.shape == matrix.shape or matrix.ndim == 4, f"matrix_local: {matrix_local.shape}, matrix: {matrix.shape}"
    assert matrix_local.shape == (J, 4, 4)
    assert len(parents) == J
    if dfs_order is None:
        _dfs_order = [i for i in range(J)]
    else:
        _dfs_order = dfs_order
    matrix_basis = np.zeros(matrix.shape)
    for i in _dfs_order:
        pid = parents[i]
        if pid == -1:
            matrix_basis[..., i, :, :] = np.linalg.inv(matrix_local[i]) @ matrix[..., i, :, :]
        else:
            pid = parents[i]
            matrix_parent = matrix[..., pid, :, :]
            matrix_local_parent = matrix_local[pid]
            
            matrix_basis[..., i, :, :] = np.linalg.inv(
                matrix_parent @
                (np.linalg.inv(matrix_local_parent) @ matrix_local[i])
            ) @ matrix[..., i, :, :]
    return matrix_basis