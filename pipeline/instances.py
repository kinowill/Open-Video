"""Instance a collection on points while sharing its original meshes."""

import bpy


def scatter(scene, name, prototypes, points, variants, rotations, scales):
    mesh = bpy.data.meshes.new(f"{name} placement points")
    mesh.from_pydata(points, [], [])
    mesh.attributes.new("variant", "INT", "POINT").data.foreach_set("value", variants)
    mesh.attributes.new("rotation", "FLOAT_VECTOR", "POINT").data.foreach_set("vector", rotations)
    mesh.attributes.new("size", "FLOAT_VECTOR", "POINT").data.foreach_set("vector", scales)
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)

    graph = bpy.data.node_groups.new(f"{name} instances", "GeometryNodeTree")
    graph.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    graph.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = graph.nodes, graph.links
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    source = nodes.new("GeometryNodeCollectionInfo")
    source.inputs["Collection"].default_value = prototypes
    source.inputs["Separate Children"].default_value = True
    source.inputs["Reset Children"].default_value = True
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    instance.inputs["Pick Instance"].default_value = True
    links.new(group_input.outputs["Geometry"], instance.inputs["Points"])
    links.new(source.outputs["Instances"], instance.inputs["Instance"])
    for socket, attribute_name, kind in (
        ("Instance Index", "variant", "INT"), ("Rotation", "rotation", "FLOAT_VECTOR"), ("Scale", "size", "FLOAT_VECTOR"),
    ):
        attribute = nodes.new("GeometryNodeInputNamedAttribute")
        attribute.data_type = kind
        attribute.inputs["Name"].default_value = attribute_name
        links.new(attribute.outputs["Attribute"], instance.inputs[socket])
    links.new(instance.outputs["Instances"], group_output.inputs["Geometry"])
    obj.modifiers.new("Collection instances", "NODES").node_group = graph
    return obj
