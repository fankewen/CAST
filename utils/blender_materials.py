"""Portable, declarative serialization of the shader nodes imported from FBX.

Texture files retain their original encoded bytes. Never use save_render here:
that applies the scene's display transform to an asset, a second time.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil


# FBX's Blender importer builds these ordinary shader nodes. Unsupported graphs
# fail explicitly instead of silently substituting a different material.
NODE_PROPERTIES = {
    "ShaderNodeBsdfPrincipled": ("distribution", "subsurface_method"),
    "ShaderNodeOutputMaterial": ("target", "is_active_output"),
    "ShaderNodeTexImage": ("interpolation", "projection", "projection_blend", "extension"),
    "ShaderNodeNormalMap": ("space", "uv_map"),
    "ShaderNodeBump": ("invert",),
    "ShaderNodeUVMap": ("uv_map", "from_instancer"),
    "ShaderNodeTexCoord": ("from_instancer",),
    "ShaderNodeMapping": ("vector_type",),
    "ShaderNodeRGB": (),
    "ShaderNodeValue": (),
    "ShaderNodeMath": ("operation", "use_clamp"),
    "ShaderNodeVectorMath": ("operation",),
    "ShaderNodeMixRGB": ("blend_type", "use_alpha", "use_clamp"),
    "ShaderNodeInvert": (),
    "ShaderNodeSeparateColor": ("mode",),
    "ShaderNodeCombineColor": ("mode",),
    "ShaderNodeEmission": (),
    "ShaderNodeBsdfTransparent": (),
    "ShaderNodeMixShader": (),
    "ShaderNodeAddShader": (),
    "NodeReroute": (),
}
MATERIAL_PROPERTIES = (
    "diffuse_color", "metallic", "roughness", "use_backface_culling",
    "surface_render_method", "blend_method", "alpha_threshold",
    "use_transparent_shadow", "use_screen_refraction",
)


def _value(value):
    return list(value) if hasattr(value, "__len__") and not isinstance(value, str) else value


class MaterialExporter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.images = {}

    def image(self, image):
        import bpy

        if image is None:
            return None
        if image.source not in {"FILE", "GENERATED"}:
            raise ValueError(f"Unsupported texture source {image.source}: {image.name}")
        if image in self.images:
            return self.images[image]
        directory = self.output_dir / "textures"
        directory.mkdir(parents=True, exist_ok=True)
        source = Path(bpy.path.abspath(image.filepath)) if image.filepath else None
        suffix = (source.suffix if source else Path(image.name).suffix).lower()
        if not suffix:
            suffix = {"JPEG": ".jpg", "PNG": ".png", "OPEN_EXR": ".exr"}.get(image.file_format, ".png")
        target = directory / f"image_{len(self.images):03d}{suffix}"
        if image.packed_file is not None and not image.is_dirty:
            target.write_bytes(bytes(image.packed_file.data))
        elif source is not None and source.is_file() and not image.is_dirty:
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
        elif image.has_data:
            # Saving an image datablock preserves texture color space; rendering
            # an image would bake AgX/exposure into the file.
            target = target.with_suffix(".exr" if image.is_float else ".png")
            duplicate = image.copy()
            try:
                # Image.copy() copies settings, but may allocate a fresh black
                # buffer for generated/edited images. Copy the pixels explicitly.
                import numpy as np
                pixels = np.empty(len(image.pixels), dtype=np.float32)
                image.pixels.foreach_get(pixels)
                duplicate.pixels.foreach_set(pixels)
                duplicate.file_format = "OPEN_EXR" if image.is_float else "PNG"
                duplicate.filepath_raw = str(target.resolve())
                duplicate.save()
            finally:
                bpy.data.images.remove(duplicate)
        else:
            raise ValueError(f"Texture has no readable file or pixels: {image.name}")
        result = {
            "path": target.relative_to(self.output_dir).as_posix(),
            "colorspace": image.colorspace_settings.name,
            "alpha_mode": image.alpha_mode,
        }
        self.images[image] = result
        return result

    def material(self, material):
        if material is None:
            return {"name": "Unassigned", "use_nodes": False, "properties": {"diffuse_color": [0.8, 0.8, 0.8, 1.0]}}
        result = {
            "name": material.name,
            "use_nodes": bool(material.use_nodes),
            "properties": {key: _value(getattr(material, key)) for key in MATERIAL_PROPERTIES if hasattr(material, key)},
            "nodes": [], "links": [],
        }
        if not material.use_nodes:
            return result
        for node in material.node_tree.nodes:
            if node.bl_idname not in NODE_PROPERTIES:
                raise ValueError(f"Unsupported shader node {node.bl_idname} in {material.name}; refusing to simplify it")
            if node.bl_idname == "ShaderNodeTexCoord" and node.object is not None:
                raise ValueError("Object-referenced texture coordinates require the original object")
            record = {
                "name": node.name, "type": node.bl_idname,
                "properties": {key: _value(getattr(node, key)) for key in NODE_PROPERTIES[node.bl_idname] if hasattr(node, key)},
                "inputs": self.sockets(node.inputs), "outputs": self.sockets(node.outputs),
            }
            if node.bl_idname == "ShaderNodeTexImage":
                record["image"] = self.image(node.image)
            result["nodes"].append(record)
        for link in material.node_tree.links:
            result["links"].append({
                "from_node": link.from_node.name, "from_socket": link.from_socket.identifier,
                "to_node": link.to_node.name, "to_socket": link.to_socket.identifier,
            })
        return result

    @staticmethod
    def sockets(sockets):
        return {socket.identifier: _value(socket.default_value) for socket in sockets if hasattr(socket, "default_value")}


def _socket(sockets, identifier):
    return next((socket for socket in sockets if socket.identifier == identifier), None)


def restore_materials(serialized, asset_dir):
    """Restore shader graphs without eval, pickles, or version-specific .blend files."""
    import bpy

    payload = json.loads(str(serialized))
    if payload.get("version") != 1:
        raise ValueError("Unsupported mesh material schema version")
    asset_dir = Path(asset_dir).resolve()
    materials, images = [], {}
    for record in payload["materials"]:
        material = bpy.data.materials.new(record["name"])
        for key, value in record["properties"].items():
            if key not in MATERIAL_PROPERTIES:
                raise ValueError(f"Unsupported material property: {key}")
            if hasattr(material, key):
                setattr(material, key, value)
        material.use_nodes = True
        tree = material.node_tree
        if not record["use_nodes"]:
            principled = tree.nodes.get("Principled BSDF")
            principled.inputs["Base Color"].default_value = material.diffuse_color
            principled.inputs["Metallic"].default_value = material.metallic
            principled.inputs["Roughness"].default_value = material.roughness
            materials.append(material)
            continue
        tree.nodes.clear()
        nodes = {}
        for item in record["nodes"]:
            node_type = item["type"]
            if node_type not in NODE_PROPERTIES:
                raise ValueError(f"Unsupported shader node: {node_type}")
            node = tree.nodes.new(node_type)
            node.name = item["name"]
            for key, value in item["properties"].items():
                if key not in NODE_PROPERTIES[node_type] or not hasattr(node, key):
                    raise ValueError(f"Cannot restore {node_type}.{key} in this Blender version")
                setattr(node, key, value)
            for side in ("inputs", "outputs"):
                for identifier, value in item[side].items():
                    socket = _socket(getattr(node, side), identifier)
                    if socket is None:
                        raise ValueError(f"Cannot restore {node_type}.{identifier} in this Blender version")
                    socket.default_value = value
            image_record = item.get("image")
            if image_record is not None:
                path = (asset_dir / image_record["path"]).resolve()
                if not path.is_relative_to(asset_dir) or not path.is_file():
                    raise FileNotFoundError(f"Missing or invalid asset texture: {path}")
                image_key = (str(path), image_record["colorspace"], image_record["alpha_mode"])
                if image_key not in images:
                    image = bpy.data.images.load(str(path), check_existing=False)
                    image.colorspace_settings.name = image_record["colorspace"]
                    image.alpha_mode = image_record["alpha_mode"]
                    images[image_key] = image
                node.image = images[image_key]
            nodes[item["name"]] = node
        for link in record["links"]:
            output = _socket(nodes[link["from_node"]].outputs, link["from_socket"])
            input_socket = _socket(nodes[link["to_node"]].inputs, link["to_socket"])
            if output is None or input_socket is None:
                raise ValueError(f"Cannot restore shader link: {link}")
            tree.links.new(output, input_socket)
        materials.append(material)
    return materials
