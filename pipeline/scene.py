"""Shared Blender scene, material and lighting helpers."""

import bpy
from mathutils import Vector


def blank_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def color_material(name, color, roughness):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    return material


def rounded_box(name, position, dimensions, material, width):
    bpy.ops.mesh.primitive_cube_add(size=1, location=position)
    obj = bpy.context.object
    obj.name = name
    obj.scale = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    for face in obj.data.polygons:
        face.use_smooth = True
    bevel = obj.modifiers.new("Rounded edges", "BEVEL")
    bevel.width, bevel.segments = width, 5
    bevel.harden_normals = True
    obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
    return obj


def stone_slab(position, dimensions, material, width):
    obj = rounded_box("Pierre de bordure Open-Video", position, dimensions, material, width)
    uv_layer = obj.data.uv_layers.active.data
    for face in obj.data.polygons:
        axes = [axis for axis in range(3) if abs(face.normal[axis]) < 0.5]
        for loop_index in face.loop_indices:
            point = obj.data.vertices[obj.data.loops[loop_index].vertex_index].co
            uv_layer[loop_index].uv = (point[axes[0]] / 0.46, point[axes[1]] / 0.46)
    return obj


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def light(scene, name, kind, position, target, power, color, size):
    data = bpy.data.lights.new(name, kind)
    data.energy = power
    data.color = color
    if kind == "AREA":
        data.size = size
    else:
        data.angle = size
    lamp = bpy.data.objects.new(name, data)
    scene.collection.objects.link(lamp)
    lamp.location = position
    aim(lamp, target)


def world(scene, color, strength):
    scene.world = bpy.data.worlds.new("Environment")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (*color, 1)
    background.inputs["Strength"].default_value = strength
