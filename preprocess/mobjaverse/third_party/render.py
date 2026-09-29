import platform
import os
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
RENDER_ENV = "egl"

from dataclasses import dataclass
from numpy import ndarray
from PIL import Image
from typing import List, Tuple, Dict

from src.rig_package.info.asset import Asset

import imageio
import math
import numpy as np
import pyrender
import random
import trimesh

@dataclass(frozen=False)
class Scene():
    lens: float # camera's lens
    
    dis: float # distance from camera to origin
    
    theta: float # polar angle, start from -y(0) and go to +z(pi/2) in blender
    
    phi: float # azimuth angle, start from -y(0) and go to +x(pi/2) in blender
    
    sensor_size: float=36.0
    
    frame: int=0
    
    image: ndarray|None=None
    
    depth: ndarray|None=None
    
    @property
    def fov(self) -> float:
        return 2 * math.atan(self.sensor_size/(2*self.lens))
    
    @property
    def camera(self) -> ndarray:
        dis = self.dis
        theta = self.theta
        phi = self.phi
        cth = math.cos(theta)
        sth = math.sin(theta)
        cph = math.cos(phi)
        sph = math.sin(phi)
        return np.array([
            [cph, -sth*sph, sph*cth, cth*sph*dis],
            [sph, sth*cph, -cph*cth, -cth*cph*dis],
            [0, cth, sth, sth*dis],
            [0, 0, 0, 1],
        ], dtype=np.float32)
    
    @property
    def yfov(self) -> float:
        return math.atan(self.sensor_size/(self.lens*2))*2
    
    def get_visibility(
        self,
        vertices: ndarray,
    ) -> ndarray:
        assert self.depth is not None
        camera_pose = self.camera
        yfov = self.yfov
        N = vertices.shape[0]
        # world → cam
        T_cw = np.linalg.inv(camera_pose)
        V_h = np.concatenate([vertices, np.ones((N,1))], axis=1)
        V_cam = (T_cw @ V_h.T).T[:, :3]
        
        z_cam = -V_cam[:,2]
        valid = z_cam > 0
        
        # intrinsics
        H = self.depth.shape[0]
        W = self.depth.shape[1]
        fy = 0.5 * H / np.tan(yfov / 2)
        fx = fy * (W / H)
        cx, cy = W / 2, H / 2
        
        u = fx * (V_cam[:,0] / -V_cam[:,2]) + cx
        v = fy * (V_cam[:,1] / -V_cam[:,2]) + cy
        v = H - v
        
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui = np.round(np.clip(u, 0, W-1)).astype(int)
        vi = np.round(np.clip(v, 0, H-1)).astype(int)
        
        offsets = np.array([
            [ 0,  0],
            [-1,  0],
            [ 1,  0],
            [ 0, -1],
            [ 0,  1],
            [-1, -1],
            [ 1, -1],
            [-1,  1],
            [ 1,  1],
        ])
        eps = 0.02
        visible = np.zeros(N, dtype=bool)
        for du, dv in offsets:
            u_n = np.clip(ui+du, 0, W-1)
            v_n = np.clip(vi+dv, 0, H-1)
            depth_n = self.depth[v_n, u_n]
            visible |= (depth_n > 0) & (z_cam <= depth_n + eps)
        visible &= valid & inside
        return visible

@dataclass(frozen=True)
class Renderer():
    
    resolution: int=512
    
    views: int=1
    
    fixed_views: List[Tuple[float, float]]|None=None
    
    dis: float=3.0
    
    lens: float=32.0
    
    sensor_size: float=36.0
    
    def get_vertex_group(self, asset: Asset) -> Dict[str, ndarray]:
        dis = self.dis
        lens = self.lens
        sensor_size = self.sensor_size
        
        array_scenes = []
        matrix_basis = asset.matrix_basis
        faces = asset.faces
        assert faces is not None
        vertex_colors = asset.vertex_colors
        texture = asset.texture # make sure vertex_colors is merged into texture in the dataset
        uvs = asset.uvs.copy() if asset.uvs is not None else None
        if uvs is not None:
            uvs[:, 1] = 1 - uvs[:, 1]
        resolution = self.resolution
        r = pyrender.OffscreenRenderer(viewport_width=resolution, viewport_height=resolution)
        if texture is not None and (texture.shape[0]==0 or texture.shape[1]==0):
            texture = None
        material = None
        new_texture = None
        new_uvs = None
        inverse_indices = None
        unique_indices = None
        
        # different intensity
        intensity = 1.5 if texture is None else 7.5
        
        for frame, pose in enumerate(matrix_basis):
            vertices = asset.vertices_with_pose(matrix_basis=pose, inplace=False, dqs=False)
            if texture is not None and uvs is not None:
                mesh, material, new_texture, new_uvs, inverse_indices, unique_indices = make_textured_mesh(
                    vertices=vertices,
                    faces=faces,
                    texture=texture,
                    uvs=uvs,
                    new_uvs=new_uvs,
                    new_texture=new_texture,
                    inverse_indices=inverse_indices,
                    unique_indices=unique_indices,
                    cached_material=material,
                )
            else:
                mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    vertex_colors=vertex_colors,
                    process=False,
                )
                material = None
            
            mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, material=material)
            scene = pyrender.Scene()
            scene.add(mesh)
            
            scenes = {}
            if self.fixed_views is not None:
                for i, (theta, phi) in enumerate(self.fixed_views):
                    theta = theta / 180.0 * math.pi
                    phi = phi / 180.0 * math.pi
                    scenes[i] = Scene(
                        lens=lens,
                        dis=dis,
                        theta=theta,
                        phi=phi,
                        sensor_size=sensor_size,
                    )
            else:
                views = np.random.randint(1, self.views+1)
                for i in range(views):
                    if i == 0: # main view
                        theta = 0.0
                        phi = 0.0
                    else:
                        theta = random.uniform(-math.pi/2, math.pi/2)
                        phi = random.uniform(0, 2*math.pi)
                    scenes[i] = Scene(
                        lens=lens,
                        dis=dis,
                        theta=theta,
                        phi=phi,
                        sensor_size=sensor_size,
                    )
            yfov = math.atan(sensor_size/(lens*2))*2
            color = np.ones(3)
            
            # render !!!
            light = pyrender.DirectionalLight(color=color, intensity=intensity)
            light_node = scene.add(light)
            for id, s in scenes.items():
                camera = pyrender.PerspectiveCamera(yfov=yfov)
                scene.add(camera, name=f"{id}", pose=s.camera)
            camera_nodes = [node for node in scene.get_nodes() if isinstance(node, pyrender.Node) and node.camera is not None]
            for cam_node in camera_nodes:
                scene.main_camera_node = cam_node
                name = cam_node.name
                light_node.matrix = cam_node.matrix
                if RENDER_ENV == "osmesa":
                    color, depth = r.render(scene)
                else:
                    color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.OFFSCREEN)
                scenes[int(name)].image = color
                scenes[int(name)].depth = depth
                scenes[int(name)].frame = frame
            for id in range(len(scenes)):
                array_scenes.append(scenes[id])
        
        r.delete()
        if asset.meta is None:
            asset.meta = {}
        asset.meta['render'] = array_scenes
        return {}

def make_textured_mesh(
    vertices: ndarray,
    faces: ndarray,
    texture: ndarray,
    uvs: ndarray,
    new_texture=None,
    new_uvs=None,
    inverse_indices=None,
    unique_indices=None,
    cached_material=None,
):
    unrolled_vertices = vertices[faces.flatten()] # (F*3, 3)
    combined = np.hstack((unrolled_vertices, uvs))
    
    if inverse_indices is None or unique_indices is None:
        unique_combined, unique_indices, inverse_indices = np.unique(combined, axis=0, return_index=True, return_inverse=True)
    else:
        unique_combined = combined[unique_indices]
        
    new_vertices = unique_combined[:, :3]
    new_faces = inverse_indices.reshape(-1, 3) # (F, 3)
    
    if new_texture is None:
        new_uvs = unique_combined[:, 3:]
        if texture.dtype != np.uint8:
            texture_uint8 = (np.clip(texture, 0, 1) * 255).astype(np.uint8)
        else:
            texture_uint8 = texture
        h, w = texture_uint8.shape[0], texture_uint8.shape[1]
        num_images = w // h
        cols = int(math.ceil(math.sqrt(num_images)))
        
        if num_images == cols:
            rows = (num_images + cols - 1) // cols
            atlas = np.zeros((rows*h, cols*h, 3), dtype=np.uint8)
            for i in range(num_images):
                row = num_images//cols - i//cols - 1
                col = i % cols
                src_x0 = i * h
                src_x1 = (i + 1) * h
                dst_y0 = row * h
                dst_y1 = (row + 1) * h
                dst_x0 = col * h
                dst_x1 = (col + 1) * h
                atlas[dst_y0:dst_y1, dst_x0:dst_x1] = texture_uint8[:, src_x0:src_x1]
            texture_uint8 = atlas
            uvs_new = new_uvs.copy()
            v = uvs_new[:, 1]
            u = uvs_new[:, 0]
            tile_id = np.floor(u * num_images).astype(int)
            tile_id = np.clip(tile_id, 0, num_images - 1)
            local_u = u * num_images - tile_id
            row = tile_id // cols
            col = tile_id % cols
            uvs_new[:, 1] = (row + v) / rows
            uvs_new[:, 0] = (col + local_u) / cols
            new_uvs = uvs_new
        else:
            pass
            
        new_texture = Image.fromarray(texture_uint8).resize((512, 512)) # downsample to speed up
    
    if cached_material is None:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorTexture=pyrender.Texture(source=new_texture, source_channels='RGB'),
            doubleSided=True,
            alphaMode='OPAQUE',
        )
    else:
        material = cached_material

    visuals = trimesh.visual.TextureVisuals(uv=new_uvs, image=new_texture)
    
    mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        visual=visuals,
        process=False,
    )
    
    return mesh, material, new_texture, new_uvs, inverse_indices, unique_indices

def save_color_video(
    colors,
    path: str,
    fps: int=30,
    s: float=1.0,
    normalize: bool=False,
):
    assert len(colors) > 0, "empty list"
    
    frames = []
    for color in colors:
        if normalize:
            color = (color-color.min()) / (color.max()-color.min()+1e-8)
        color = np.clip(color*s, 0, 255).astype(np.uint8)
        frames.append(color)
    
    if os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(
        path,
        frames,
        fps=fps,
        codec="libx264",
        quality=5,
        macro_block_size=None,
    )

def main():
    renderer = Renderer()
    data = np.load("mobjaverse/004444/raw_data.npz", allow_pickle=True)
    
    # unwrap None object to None
    def unwrap(x):
        if isinstance(x, ndarray) and x.shape==() and x.dtype==object:
            return x.item()
        return x
    data = {k: unwrap(v) for k, v in data.items()}
    asset = Asset(**data)
    assert asset.matrix_basis is not None
    
    asset.matrix_basis[:, 0, :3, 3] = 0 # remove root translation
    asset.normalize_vertices((-1.0, 1.0))
    _ = renderer.get_vertex_group(asset)
    images = []
    assert asset.meta is not None
    for scene in asset.meta['render']:
        images.append(scene.image)
    save_color_video(images, "res.mp4", fps=30)
    

if __name__ == "__main__":
    main()
