"""Blend calibrated perspective photos over the existing Cycles surface."""

import json
import math
import struct
from bisect import bisect_right
from pathlib import Path
from time import perf_counter

import bpy
from mathutils import Vector

from pipeline.profondeur_projection import bake_depth_images


def _numbers(values, length):
    return (isinstance(values, (tuple, list)) and len(values) == length
            and all(type(value) in (int, float) and math.isfinite(value) for value in values))


def _read_manifest(path, center):
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a capture manifest object: {path}")
    size = (manifest.get("width"), manifest.get("height"))
    if any(type(side) is not int or side <= 0 for side in size):
        raise ValueError(f"Expected positive capture width and height: {path}")
    entries = manifest.get("views")
    if not isinstance(entries, list) or not all(isinstance(view, dict) for view in entries):
        raise ValueError(f"Expected a list of capture views: {path}")
    views = [entry for entry in entries if entry.get("split") == "texture"]
    if not views:
        views = [entry for entry in entries if entry.get("split", "train") == "train"]
    if not views:
        raise ValueError(f"No texture or training views in capture manifest: {path}")
    filenames = set()
    directions = []
    for view in views:
        name = view.get("file")
        if not isinstance(name, str) or not name.strip() or name in filenames:
            raise ValueError(f"Expected a distinct image file for every capture view: {path}")
        filenames.add(name)
        camera = view.get("camera")
        matrix = view.get("projection")
        if (not _numbers(camera, 3) or not isinstance(matrix, list) or len(matrix) != 4
                or not all(_numbers(row, 4) for row in matrix)):
            raise ValueError(f"Expected a finite camera position and 4×4 projection matrix: {name}")
        offset = Vector(camera) - center
        if not math.isfinite(offset.length) or offset.length == 0:
            raise ValueError(f"Capture camera must be distinct from the subject center: {name}")
        # A perspective camera is the common zero of the X, Y and W clip rows.
        # This catches pose convention mismatches before building the shader.
        for index in (0, 1, 3):
            row = matrix[index]
            length = math.hypot(*row[:3])
            at_camera = math.fsum(coefficient * coordinate for coefficient, coordinate in zip(row, [*camera, 1]))
            if not length or abs(at_camera) > 0.001 * length * offset.length:
                raise ValueError(f"Capture position and perspective matrix disagree: {name}")
        at_center = math.fsum(coefficient * coordinate for coefficient, coordinate in zip(matrix[3], [*center, 1]))
        if at_center <= 0:
            raise ValueError(f"Capture camera points away from the subject center: {name}")
        directions.append(offset.normalized())
    return size, views, directions


def _scene_directions(scene, center):
    directions = []
    current_frame, subframe = scene.frame_current, scene.frame_subframe
    try:
        for frame in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(frame)
            camera = scene.camera
            if camera is None or camera.type != "CAMERA" or camera.data.type != "PERSP":
                raise ValueError(f"Expected a perspective scene camera at frame {frame}")
            offset = camera.matrix_world.translation - center
            if not math.isfinite(offset.length) or offset.length == 0:
                raise ValueError(f"Invalid camera position at frame {frame}")
            directions.append(offset.normalized())
    finally:
        scene.frame_set(current_frame, subframe=subframe)
    return directions


def _elevation(direction):
    return math.atan2(direction.z, math.hypot(direction.x, direction.y))


def _azimuth(direction):
    return math.atan2(direction.y, direction.x) % math.tau


def _capture_ring(directions, trajectory):
    if len(directions) < 3:
        return None
    elevations = [_elevation(direction) for direction in directions]
    mean = sum(elevations) / len(elevations)
    if max(elevations) - min(elevations) > math.radians(12):
        return None
    if abs(mean) > math.radians(60) or any(abs(_elevation(direction) - mean) > math.radians(30)
                                           for direction in trajectory):
        return None
    order = sorted(range(len(directions)), key=lambda index: _azimuth(directions[index]))
    angles = [_azimuth(directions[index]) for index in order]
    gaps = [right - left for left, right in zip(angles, angles[1:] + [angles[0] + math.tau])]
    if min(gaps) < 1e-6 or max(gaps) >= math.pi:
        return None
    return angles, order


def _smoothstep(value):
    return value * value * (3 - 2 * value)


def _ring_weights(ring, direction):
    angles, order = ring
    azimuth = _azimuth(direction)
    right = bisect_right(angles, azimuth)
    left_angle = angles[right - 1] - (math.tau if right == 0 else 0)
    right_angle = angles[right % len(angles)] + (math.tau if right == len(angles) else 0)
    fraction = _smoothstep((azimuth - left_angle) / (right_angle - left_angle))
    return {order[right - 1]: 1 - fraction, order[right % len(angles)]: fraction}


def _angular_weights(direction, captures):
    distances = [math.acos(max(-1.0, min(1.0, direction.dot(view)))) for view in captures]
    ordered = sorted(distances)
    # The support follows local spacing. A small margin makes ties fade smoothly.
    cutoff = ordered[3] + math.radians(1) if len(ordered) > 3 else math.pi
    weights = {index: (cutoff - distance)**2 / (distance**2 + 1e-4)
               for index, distance in enumerate(distances) if distance < cutoff}
    total = sum(weights.values())
    return {index: weight / total for index, weight in weights.items()}


def _weights(direction, captures, ring):
    weights = _ring_weights(ring, direction) if ring else _angular_weights(direction, captures)
    if not weights:
        return {}
    nearest = max(direction.dot(captures[index]) for index in weights)
    # Fall back gradually as the shot moves outside the captured hemisphere.
    confidence = _smoothstep(max(0.0, min(1.0, nearest / math.cos(math.radians(55)))))
    return {index: weight * confidence for index, weight in weights.items() if weight * confidence > 1e-8}


def _photo_paths(capture_dir, size, views):
    paths = [capture_dir / view["file"] for view in views]
    for path in paths:
        with path.open("rb") as stream:
            header = stream.read(29)
        if (len(header) != 29 or header[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"
                or struct.unpack(">II", header[16:24]) != size or header[25] != 6):
            raise ValueError(f"Expected a {size[0]}×{size[1]} RGBA PNG: {path}")
    return paths


def _load_photos(paths, size, indices):
    photos = {}
    try:
        for index in indices:
            path = paths[index]
            photo = bpy.data.images.load(str(path), check_existing=False)
            photos[index] = photo
            if tuple(photo.size) != size or photo.channels != 4:
                raise ValueError(f"Cannot decode RGBA capture: {path}")
            photo.alpha_mode = "STRAIGHT"
            photo.pack()
    except (OSError, RuntimeError, ValueError):
        for photo in photos.values():
            bpy.data.images.remove(photo)
        raise
    return photos


def _bind(tree, value, socket):
    if isinstance(value, (float, int, tuple)):
        socket.default_value = value
    else:
        tree.links.new(value, socket)


def _math(tree, operation, left, right):
    node = tree.nodes.new("ShaderNodeMath")
    node.operation = operation
    _bind(tree, left, node.inputs[0])
    _bind(tree, right, node.inputs[1])
    return node.outputs[0]


def _vector(tree, operation, left, right=None):
    node = tree.nodes.new("ShaderNodeVectorMath")
    node.operation = operation
    _bind(tree, left, node.inputs[0])
    if right is not None:
        _bind(tree, right, node.inputs[3] if operation == "SCALE" else node.inputs[1])
    return node.outputs["Value" if operation == "DOT_PRODUCT" else "Vector"]


def _depth_visibility(tree, coordinates, depth, measurement, cosine):
    image, footprint = measurement
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.label = image.name
    texture.image = image
    texture.extension = "CLIP"
    texture.interpolation = "Closest"
    _bind(tree, coordinates, texture.inputs["Vector"])
    front_depth = texture.outputs["Color"]
    depth_gap = _math(tree, "SUBTRACT", depth, front_depth)
    pixel_span = _math(tree, "MULTIPLY", depth, 2 * footprint)
    tolerance = _math(tree, "DIVIDE", pixel_span, _math(tree, "MAXIMUM", cosine, 0.25))
    behind = _math(tree, "DIVIDE", depth_gap, tolerance)
    visible = _math(tree, "SUBTRACT", 2, behind)
    visible.node.use_clamp = True
    # A ray that missed the subject gives no depth evidence at this texel.
    return _math(tree, "MAXIMUM", visible, _math(tree, "LESS_THAN", front_depth, 1e-8))


def _sample(tree, geometry, view, photo, measurement):
    position = geometry.outputs["Position"]
    matrix = view["projection"]
    clip = [_math(tree, "ADD", _vector(tree, "DOT_PRODUCT", position, tuple(matrix[index][:3])), matrix[index][3])
            for index in (0, 1, 3)]
    coordinates = tree.nodes.new("ShaderNodeCombineXYZ")
    for index in (0, 1):
        projected = _math(tree, "DIVIDE", clip[index], clip[2])
        _bind(tree, _math(tree, "ADD", 0.5, _math(tree, "MULTIPLY", 0.5, projected)),
              coordinates.inputs[index])
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.label = view["file"]
    texture.image = photo
    texture.extension = "CLIP"
    _bind(tree, coordinates.outputs[0], texture.inputs["Vector"])
    towards = _vector(tree, "NORMALIZE", _vector(tree, "SUBTRACT", tuple(view["camera"]), position))
    # Smooth shading normals can hide parts of a visible low-poly face.
    cosine = _vector(tree, "DOT_PRODUCT", geometry.outputs["True Normal"], towards)
    front = _math(tree, "MULTIPLY", 6, cosine)
    front.node.use_clamp = True
    in_front_of_camera = _math(tree, "GREATER_THAN", clip[2], 0)
    alpha = _math(tree, "MULTIPLY", texture.outputs["Alpha"], in_front_of_camera)
    visibility = _depth_visibility(tree, coordinates.outputs[0], clip[2], measurement, cosine)
    alpha = _math(tree, "MULTIPLY", alpha, visibility)
    return texture.outputs["Color"], _math(tree, "MULTIPLY", front, alpha)


def _animate_weight(socket, index, first_frame, timeline):
    values = [weights.get(index, 0.0) for weights in timeline]
    socket.default_value = values[0]
    if all(value == values[0] for value in values):
        return
    for offset, value in enumerate(values):
        if 0 < offset < len(values) - 1 and values[offset - 1] == value == values[offset + 1]:
            continue
        socket.default_value = value
        socket.keyframe_insert("default_value", frame=first_frame + offset)


def _make_projection(scene, views, photos, depths, timeline):
    tree = bpy.data.node_groups.new("Open-Video Photo Projection", "ShaderNodeTree")
    tree.interface.new_socket(name="Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    tree.interface.new_socket(name="Coverage", in_out="OUTPUT", socket_type="NodeSocketFloat")
    output = tree.nodes.new("NodeGroupOutput")
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    total = 0.0
    rgb = (0.0, 0.0, 0.0)
    for index, photo in photos.items():
        color, mask = _sample(tree, geometry, views[index], photo, depths[index])
        weight = tree.nodes.new("ShaderNodeValue")
        weight.label = views[index]["file"]
        _animate_weight(weight.outputs[0], index, scene.frame_start, timeline)
        factor = _math(tree, "MULTIPLY", mask, weight.outputs[0])
        total = _math(tree, "ADD", total, factor)
        rgb = _vector(tree, "ADD", rgb, _vector(tree, "SCALE", color, factor))

    inverse = _math(tree, "DIVIDE", 1, _math(tree, "MAXIMUM", total, 0.0001))
    _bind(tree, _vector(tree, "SCALE", rgb, inverse), output.inputs["Color"])
    coverage = _math(tree, "MULTIPLY", 5, total)
    coverage.node.use_clamp = True
    _bind(tree, coverage, output.inputs["Coverage"])
    if tree.animation_data is not None:
        animation = tree.animation_data
        strip = animation.action.layers[0].strips[0]
        for curve in strip.channelbag(animation.action_slot).fcurves:
            for key in curve.keyframe_points:
                key.interpolation = "LINEAR"
    return tree


def _project_material(material, projection):
    had_nodes = material.use_nodes
    material.use_nodes = True
    tree = material.node_tree
    output = tree.get_output_node("CYCLES")
    if output is None:
        output = tree.nodes.new("ShaderNodeOutputMaterial")
    surface = output.inputs["Surface"]
    previous = surface.links[0].from_socket if surface.is_linked else None
    if not had_nodes:
        base = tree.nodes.new("ShaderNodeBsdfPrincipled")
        base.inputs["Base Color"].default_value = material.diffuse_color
        previous = base.outputs[0]
    group = tree.nodes.new("ShaderNodeGroup")
    group.node_tree = projection
    rays = tree.nodes.new("ShaderNodeLightPath")
    factor = _math(tree, "MULTIPLY", rays.outputs["Is Camera Ray"], group.outputs["Coverage"])
    emission = tree.nodes.new("ShaderNodeEmission")
    _bind(tree, group.outputs["Color"], emission.inputs["Color"])
    mix = tree.nodes.new("ShaderNodeMixShader")
    _bind(tree, factor, mix.inputs[0])
    if previous is not None:
        _bind(tree, previous, mix.inputs[1])
    _bind(tree, emission.outputs[0], mix.inputs[2])
    _bind(tree, mix.outputs[0], surface)


def apply_photo_projection(scene, subject, manifest_path, center):
    """Bake the shot's camera weights. Subject and captures share the Blender world pose."""
    coordinates = tuple(center)
    if not _numbers(coordinates, 3):
        raise ValueError("Expected a finite subject center in Blender world coordinates")
    center = Vector(coordinates)
    meshes = [obj for obj in subject.all_objects if obj.type == "MESH" and not obj.hide_render]
    if not meshes:
        raise ValueError("Photo projection needs a visible subject mesh")
    path = Path(manifest_path).resolve(strict=True)
    size, views, captures = _read_manifest(path, center)
    paths = _photo_paths(path.parent, size, views)
    trajectory = _scene_directions(scene, center)
    ring = _capture_ring(captures, trajectory)
    timeline = [_weights(direction, captures, ring) for direction in trajectory]
    indices = sorted({index for weights in timeline for index in weights})
    report = {
        "available_views": len(views),
        "active_views": [views[index]["file"] for index in indices],
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "weighting": "azimuth-ring" if ring else "spherical",
    }
    if not indices:
        return report

    photos = _load_photos(paths, size, indices)
    start = perf_counter()
    depths, dimensions = bake_depth_images(meshes, views, size, indices)
    report["depth_resolution"] = list(dimensions)
    report["depth_bake_seconds"] = round(perf_counter() - start, 3)
    projection = _make_projection(scene, views, photos, depths, timeline)
    materials = {}
    for obj in meshes:
        if not obj.material_slots:
            if obj.data.users > 1:
                obj.data = obj.data.copy()
            obj.data.materials.append(None)
        for slot in obj.material_slots:
            original = slot.material
            if original not in materials:
                material = original.copy() if original else bpy.data.materials.new("Open-Video Photo Projection Base")
                _project_material(material, projection)
                materials[original] = material
            slot.link = "OBJECT"
            slot.material = materials[original]
    scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)
    return report
