"""Add a bounded, softly fading ground mist to a Blender scene."""

from math import isfinite

import bmesh
import bpy
from mathutils import Vector


def add_ground_mist(scene, bounds, density, color=(1, 1, 1), anisotropy=0.0, edge_fade=0.2, step_rate=1.0):
    """Scatter light in a box defined by its lower and upper world coordinates.

    Density is greatest at ground level and fades to zero at the top and on
    the four vertical sides. Edge fade is a fraction of each horizontal span.
    Step rate sets the Cycles sampling interval for this material.
    """
    lower, upper = map(Vector, bounds)
    if len(lower) != 3 or len(upper) != 3 or any(not isfinite(value) for value in (*lower, *upper)):
        raise ValueError('Mist bounds need finite XYZ coordinates')
    dimensions = upper - lower
    if min(dimensions) <= 0 or not isfinite(density) or density <= 0:
        raise ValueError('Mist dimensions and density must be positive')
    if (len(color) != 3 or not all(0 <= channel <= 1 for channel in color)
            or not -1 < anisotropy < 1 or not 0 < edge_fade <= 0.5):
        raise ValueError('Invalid scattering color, anisotropy or edge fade')
    cycles_properties = bpy.types.Material.bl_rna.properties['cycles'].fixed_type.properties
    rate_property = cycles_properties['volume_step_rate']
    if not rate_property.hard_min <= step_rate <= rate_property.hard_max:
        raise ValueError('Unsupported volume step rate')

    name = 'Open-Video · Brume basse'
    mesh = bpy.data.meshes.new(name)
    geometry = bmesh.new()
    try:
        bmesh.ops.create_cube(geometry, size=1)
        for vertex in geometry.verts:
            for axis, span in enumerate(dimensions):
                vertex.co[axis] *= span
        geometry.to_mesh(mesh)
    finally:
        geometry.free()
    mist = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(mist)
    mist.location = (lower + upper) / 2
    mist.display_type = 'WIRE'

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.cycles.volume_step_rate = step_rate
    mesh.materials.append(material)
    tree = material.node_tree
    tree.nodes.clear()
    coordinates = tree.nodes.new('ShaderNodeTexCoord')
    separate = tree.nodes.new('ShaderNodeSeparateXYZ')
    tree.links.new(coordinates.outputs['Generated'], separate.inputs['Vector'])

    def math_node(operation, left, right):
        node = tree.nodes.new('ShaderNodeMath')
        node.operation = operation
        for socket, value in zip(node.inputs, (left, right)):
            if isinstance(value, bpy.types.NodeSocket):
                tree.links.new(value, socket)
            else:
                socket.default_value = value
        return node.outputs[0]

    def fade(value, threshold, start, end):
        node = tree.nodes.new('ShaderNodeMapRange')
        node.interpolation_type = 'SMOOTHSTEP'
        node.clamp = True
        tree.links.new(value, node.inputs['Value'])
        node.inputs['From Max'].default_value = threshold
        node.inputs['To Min'].default_value = start
        node.inputs['To Max'].default_value = end
        return node.outputs['Result']

    x, y, z = separate.outputs
    edge_x = math_node('MINIMUM', x, math_node('SUBTRACT', 1, x))
    edge_y = math_node('MINIMUM', y, math_node('SUBTRACT', 1, y))
    edges = fade(math_node('MINIMUM', edge_x, edge_y), edge_fade, 0, 1)
    height = fade(z, 1, density, 0)

    scatter = tree.nodes.new('ShaderNodeVolumeScatter')
    scatter.inputs['Color'].default_value = (*color, 1)
    scatter.inputs['Anisotropy'].default_value = anisotropy
    tree.links.new(math_node('MULTIPLY', edges, height), scatter.inputs['Density'])
    output = tree.nodes.new('ShaderNodeOutputMaterial')
    tree.links.new(scatter.outputs['Volume'], output.inputs['Volume'])
    return mist
