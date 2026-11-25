"""Small layers for the Principled materials used in Blender scenes."""

from math import isfinite


def add_triplanar_bump(material, image, *, tile_size, distance, strength=0.3):
    """Layer a repeating height map in world units over existing surface normals."""
    if not all(isfinite(value) for value in (tile_size, distance, strength)) or (
        tile_size <= 0 or distance < 0 or not 0 <= strength <= 1
    ):
        raise ValueError("Expected a positive tile size, nonnegative distance and strength in [0, 1]")
    tree = material.node_tree
    shader = tree.nodes.get("Principled BSDF") if tree else None
    if shader is None:
        raise ValueError(f"Missing Principled BSDF on {material.name}")
    normal = shader.inputs["Normal"]
    previous = normal.links[0].from_socket if normal.is_linked else None
    nodes, links = tree.nodes, tree.links
    geometry = nodes.new("ShaderNodeNewGeometry")
    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs[3].default_value = 1 / tile_size
    links.new(geometry.outputs["Position"], scale.inputs[0])
    tile = nodes.new("ShaderNodeTexImage")
    tile.image = image
    tile.label = image.name
    tile.projection = "BOX"
    tile.projection_blend = 0.15
    links.new(scale.outputs["Vector"], tile.inputs["Vector"])
    bump = nodes.new("ShaderNodeBump")
    bump.label = "Relief texturé Open-Video"
    bump.inputs["Distance"].default_value = distance
    bump.inputs["Strength"].default_value = strength
    links.new(tile.outputs["Color"], bump.inputs["Height"])
    if previous is not None:
        links.new(previous, bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], normal)


def fade_bump_by_distance(bump, near, far):
    """Fade fine relief between two camera distances, preserving its incoming strength."""
    if (bump.bl_idname != "ShaderNodeBump" or not all(isfinite(value) for value in (near, far))
            or not 0 <= near < far):
        raise ValueError("Expected a bump node and increasing nonnegative camera distances")

    tree = bump.id_data
    strength = bump.inputs["Strength"]
    previous = strength.links[0].from_socket if strength.is_linked else None
    camera = tree.nodes.new("ShaderNodeCameraData")
    fade = tree.nodes.new("ShaderNodeMapRange")
    fade.label = "Atténuation du microrelief avec la distance"
    fade.interpolation_type = "SMOOTHSTEP"
    fade.clamp = True
    for name, value in (("From Min", near), ("From Max", far), ("To Min", 1), ("To Max", 0)):
        fade.inputs[name].default_value = value
    tree.links.new(camera.outputs["View Distance"], fade.inputs["Value"])

    weight = tree.nodes.new("ShaderNodeMath")
    weight.operation = "MULTIPLY"
    weight.inputs[1].default_value = strength.default_value
    if previous is not None:
        tree.links.new(previous, weight.inputs[1])
    tree.links.new(fade.outputs["Result"], weight.inputs[0])
    tree.links.new(weight.outputs[0], strength)
