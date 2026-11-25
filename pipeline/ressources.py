"""Load reusable assets and normalize paths in Blender packages."""

from pathlib import Path

import bpy


STONE_MATERIAL_NAME = "Open-Video Stone"


def load_stone(library):
    with bpy.data.libraries.load(str(library)) as (source, destination):
        if STONE_MATERIAL_NAME not in source.materials:
            raise RuntimeError(f"Missing material {STONE_MATERIAL_NAME}")
        destination.materials = [STONE_MATERIAL_NAME]
    material = destination.materials[0]
    textures = [node.image for node in material.node_tree.nodes if node.type == "TEX_IMAGE"]
    if len(textures) != 3 or any(image is None or image.packed_file is None for image in textures):
        raise RuntimeError("Expected all three texture maps in the material library")
    return material


def relative_packed_paths(images):
    for image in images:
        if not image.packed_files:
            continue
        filename = Path(image.filepath).name
        image.filepath = f"//{filename}"
        for packed in image.packed_files:
            packed.filepath = f"//{Path(packed.filepath).name or filename}"
