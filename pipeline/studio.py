"""Prepare a studio orbit around a collection from a Blender asset library."""

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import bpy
from mathutils import Vector

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.ressources import load_stone, relative_packed_paths
from pipeline.rendu import PROFILES, configure_cycles, render_frame, soften_camera_background, validate_recipe, write_json
from pipeline.scene import aim, blank_scene, color_material, light, stone_slab, world


def load_subject(scene, path, name):
    with bpy.data.libraries.load(str(path)) as (source, destination):
        if name not in source.collections:
            raise ValueError(f"Collection not found: {name}. Available: {', '.join(source.collections)}")
        destination.collections = [name]
    subject = destination.collections[0]
    scene.collection.children.link(subject)
    return subject


def linked_node(socket, kind):
    if socket.is_linked and socket.links[0].from_node.type == kind:
        return socket.links[0].from_node
    return None


def photo_contact_shadows(subject):
    """Restore contact shadows for the camera-only shader imported from glTF unlit."""
    materials = {material for obj in subject.all_objects if obj.type == "MESH"
                 for material in obj.data.materials if material and material.use_nodes}
    for material in materials:
        tree = material.node_tree
        output = next((node for node in tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output), None)
        if output is None:
            continue
        mix = linked_node(output.inputs["Surface"], "MIX_SHADER")
        if mix is None:
            continue
        alpha_surface = linked_node(mix.inputs[2], "MIX_SHADER")
        if linked_node(mix.inputs[1], "BSDF_TRANSPARENT") and alpha_surface:
            mix = alpha_surface
        ray = mix.inputs[0]
        if not (linked_node(ray, "LIGHT_PATH") and ray.links[0].from_socket.name == "Is Camera Ray"
                and linked_node(mix.inputs[1], "BSDF_TRANSPARENT") and linked_node(mix.inputs[2], "EMISSION")):
            continue
        diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.label = "Open-Video contact shadows"
        diffuse.inputs["Color"].default_value = (0.035, 0.035, 0.035, 1)
        tree.links.new(diffuse.outputs[0], mix.inputs[1])


def subject_bounds(collection):
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    corners = []
    for obj in collection.all_objects:
        if obj.type == "MESH" and not obj.hide_render:
            evaluated = obj.evaluated_get(graph)
            corners.extend(evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box)
    if not corners:
        raise ValueError("The studio subject needs a visible mesh")
    minimum = Vector(min(point[axis] for point in corners) for axis in range(3))
    maximum = Vector(max(point[axis] for point in corners) for axis in range(3))
    scale = max(maximum - minimum)
    if scale == 0:
        raise ValueError("The studio subject has no extent")
    return minimum, maximum, scale


def stage(center, minimum, size, extent, settings, stone_path):
    plinth_height = extent * 0.10 if settings.get("plinth", False) else 0
    if plinth_height:
        if stone_path is None:
            raise ValueError("A stone library is required when studio.plinth is enabled")
        stone = load_stone(stone_path)
        plinth = stone_slab((center.x, center.y, minimum.z - plinth_height / 2),
                            (max(size.x * 1.37, extent * 0.4), max(size.y * 1.30, extent * 0.4), plinth_height),
                            stone, extent * 0.021)
        plinth.name = "Open-Video stone plinth"
    floor = color_material("Studio floor", settings["floor_color"], 0.49)
    bpy.ops.mesh.primitive_plane_add(size=extent * 90,
                                    location=(center.x, center.y, minimum.z - plinth_height - extent * 0.0003))
    bpy.context.object.data.materials.append(floor)


def orbit(scene, center, scale, recipe):
    settings = recipe["studio"]
    camera_settings = settings["camera"]
    target = center + Vector(camera_settings.get("target_offset", [0, 0, 0])) * scale
    world(scene, settings["world_color"], settings["world_strength"])
    for lamp in settings["lights"]:
        position = center + Vector(lamp["offset"]) * scale
        light(scene, lamp["name"], "AREA", position, target, lamp["power"] * scale**2,
              lamp["color"], lamp["size"] * scale)
    scene.view_settings.exposure = settings.get("exposure", 0)

    data = bpy.data.cameras.new("Open-Video studio camera")
    camera = bpy.data.objects.new(data.name, data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    data.lens = camera_settings["lens_mm"]
    data.clip_start = scale * 0.01
    data.clip_end = scale * 300
    data.dof.use_dof = True
    data.dof.aperture_fstop = camera_settings["f_stop"]
    radius = camera_settings["distance"] * scale
    elevation = camera_settings["height"] * scale
    data.dof.focus_distance = math.hypot(radius, elevation)
    for index in range(recipe["frames"]):
        fraction = index / max(recipe["frames"] - 1, 1)
        eased = fraction * fraction * (3 - 2 * fraction)
        angle = math.radians(camera_settings["start_degrees"] + camera_settings["sweep_degrees"] * eased)
        camera.location = target + Vector((radius * math.sin(angle), -radius * math.cos(angle), elevation))
        previous_rotation = camera.rotation_euler.copy()
        aim(camera, target)
        camera.rotation_euler.make_compatible(previous_rotation)
        frame = recipe.get("first_frame", 1) + index
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--collection", help="Override studio.collection in the recipe")
    parser.add_argument("--captures", type=Path, help="Calibrated photos for camera-dependent projection")
    parser.add_argument("--stone", type=Path)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="reference")
    parser.add_argument("--lighting", choices=("reference", "fond-doux"), default="reference")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    validate_recipe(recipe)
    studio = recipe["studio"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    scene = blank_scene()
    subject = load_subject(scene, args.asset.resolve(strict=True), args.collection or studio["collection"])
    if studio.get("photo_shadows", False) or args.captures:
        photo_contact_shadows(subject)
    scene.frame_set(recipe.get("first_frame", 1))
    minimum, maximum, extent = subject_bounds(subject)
    center = (minimum + maximum) / 2
    stage(center, minimum, maximum - minimum, extent, studio, args.stone)
    orbit(scene, center, extent, recipe)
    devices = configure_cycles(scene, recipe, args.profile)
    projection = None
    if args.captures:
        from pipeline.projection_photos import apply_photo_projection

        projection = apply_photo_projection(scene, subject, args.captures.resolve(strict=True), center)
    frame = (scene.frame_start + scene.frame_end) // 2
    scene.frame_set(frame)
    bpy.ops.file.pack_all()
    relative_packed_paths(bpy.data.images)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "scene.blend"), compress=True, relative_remap=False)
    assembly = round(perf_counter() - start, 3)
    if args.lighting == "fond-doux":
        soften_camera_background(scene, recipe.get("camera_background_factor", 0.72))
    preview = render_frame(scene, frame, output / "apercu.png")
    report = {
        "blender": bpy.app.version_string,
        "gpu": devices,
        "backend": "Cycles OptiX",
        "assembly_seconds": assembly,
        "preview_frame": frame,
        "preview_render_seconds": preview,
        "collection": subject.name,
        "subject_extent_m": round(extent, 4),
        "profile": args.profile,
        "lighting": args.lighting,
        "orbit_degrees": studio["camera"]["sweep_degrees"] if recipe["frames"] > 1 else 0,
        "area_lights": len(studio["lights"]),
    }
    if projection is not None:
        report["photo_projection"] = projection
    write_json(output / "preparation.json", report)
    print("Open-Video STUDIO " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
