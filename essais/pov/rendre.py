"""Prepare or render the Open-Video lawn-level camera trial in background Blender."""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from time import perf_counter

import bpy
from mathutils import Vector

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.instances import scatter
from pipeline.ressources import load_stone, relative_packed_paths
from pipeline.rendu import PROFILES, configure_cycles, render_frames, soften_camera_background, validate_recipe, write_json
from pipeline.scene import aim, blank_scene, color_material, light, stone_slab


GRASS_VARIANTS = [f"grass_medium_02_{letter}_geo" for letter in "abcde"]


def packed_image(path, color_space=None):
    image = bpy.data.images.load(str(path), check_existing=True)
    if color_space:
        image.colorspace_settings.name = color_space
    image.pack()
    image.filepath = f"//{path.name}"
    return image


def add_lighting(scene, assets, settings):
    scene.world = bpy.data.worlds.new("Meadow daylight")
    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    links = scene.world.node_tree.links
    background = nodes.get("Background")
    background.inputs["Strength"].default_value = settings["sky_strength"]
    texture = nodes.new("ShaderNodeTexEnvironment")
    texture.image = packed_image(assets / "environment/meadow.hdr")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value.z = math.radians(settings["sky_rotation_degrees"])
    links.new(nodes.new("ShaderNodeTexCoord").outputs["Normal"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], background.inputs["Color"])
    light(scene, "Soft morning sun", "SUN", (-3.0, -4.0, 6.0), (0, 0, 0),
          settings["sun_energy"], (1.0, 0.91, 0.78), 0.105)
    light(scene, "Sky fill", "AREA", (-1.1, -3.4, 1.65), (-0.1, -1.3, 0),
          settings["fill_energy"], (0.81, 0.9, 1.0), 2.5)
    scene.view_settings.exposure = settings["exposure"]


def floor_height(x, y):
    return -0.016 + 0.006 * math.cos(y * 2.3 + x) * math.sin(x * 3.4)


def add_floor(assets):
    material = color_material("Grass path soil", (0.22, 0.16, 0.09), 0.85)
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = nodes.get("Principled BSDF")
    normal = nodes.new("ShaderNodeNormalMap")
    normal.inputs["Strength"].default_value = 1.0
    for channel, color_space, socket in (
        ("basecolor", "sRGB", shader.inputs["Base Color"]),
        ("roughness", "Non-Color", shader.inputs["Roughness"]),
        ("normal", "Non-Color", normal.inputs["Color"]),
    ):
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = packed_image(assets / "ground" / f"{channel}.jpg", color_space)
        links.new(texture.outputs["Color"], socket)
    links.new(normal.outputs["Normal"], shader.inputs["Normal"])

    columns, rows = 51, 91
    points = [(x / 2 - 14, y / 2 - 7, floor_height(x / 2 - 14, y / 2 - 7))
              for y in range(rows) for x in range(columns)]
    faces = [(base, base + 1, base + columns + 1, base + columns)
             for y in range(rows - 1) for x in range(columns - 1) for base in [y * columns + x]]
    mesh = bpy.data.meshes.new("Soil surface")
    mesh.from_pydata(points, [], faces)
    uv = mesh.uv_layers.new(name="Meters")
    for face in mesh.polygons:
        face.use_smooth = True
        for loop_index in face.loop_indices:
            uv.data[loop_index].uv = points[mesh.loops[loop_index].vertex_index][:2]
    mesh.materials.append(material)
    bpy.context.scene.collection.objects.link(bpy.data.objects.new("Soil", mesh))


def add_border(stone):
    for node in stone.node_tree.nodes:
        if node.type == "NORMAL_MAP":
            node.inputs["Strength"].default_value = 0.72
    for index in range(28):
        y = -4.5 + 0.585 * index
        block = stone_slab(
            (0.06 + 0.017 * math.sin(index * 1.8), y, 0.028), (0.46, 0.559, 0.108), stone, 0.014
        )
        block.rotation_euler.z = math.radians(0.85 * math.sin(index * 1.7))


def load_grass(assets):
    library = assets / "grass/grass_medium_02_1k.blend"
    with bpy.data.libraries.load(str(library)) as (available, imported):
        if not set(GRASS_VARIANTS).issubset(available.objects):
            raise RuntimeError("Grass library is missing a clump mesh")
        imported.objects = GRASS_VARIANTS
    prototypes = bpy.data.collections.new("Grass prototypes")
    for obj in imported.objects:
        prototypes.objects.link(obj)
        obj.location = (0, 0, 0)
    shader = imported.objects[0].data.materials[0].node_tree.nodes["Group"]
    shader.inputs["Value"].default_value = 0.78
    shader.inputs["Saturation"].default_value = 1.15
    for image in bpy.data.images:
        if image.name.startswith("grass_medium_02_"):
            path = assets / "grass/textures" / Path(image.filepath).name
            if not path.is_file():
                raise FileNotFoundError(path)
            image.filepath = str(path)
            image.reload()
            image.pack()
            image.filepath = f"//{path.name}"
    return prototypes


def add_grass(assets, seed):
    prototypes = load_grass(assets)
    rng = random.Random(seed)
    points, variants, rotations, scales = [], [], [], []
    # Keep the densest clumps near the tracking path.
    zones = [(-2.4, -0.185, -3.8, 6.4, 0.076), (-8.0, -2.4, -5.0, 6.5, 0.24),
             (-9.0, -0.18, 6.4, 22.0, 0.25), (1.25, 6.2, -1.5, 16.0, 0.28)]
    for x0, x1, y0, y1, spacing in zones:
        for column in range(math.ceil((x1 - x0) / spacing)):
            for row in range(math.ceil((y1 - y0) / spacing)):
                x = x0 + (column + rng.uniform(0.10, 0.90)) * spacing
                y = y0 + (row + rng.uniform(0.10, 0.90)) * spacing
                if x > x1 or y > y1:
                    continue
                height = 0.45 + 0.12 * math.sin(y * 3.1 + x) + rng.uniform(-0.075, 0.13)
                width = rng.uniform(1.04, 1.40)
                points.append((x, y, floor_height(x, y) - 0.012))
                variants.append(rng.randrange(len(GRASS_VARIANTS)))
                rotations.extend((rng.uniform(-0.09, 0.09), rng.uniform(-0.09, 0.09), rng.random() * math.tau))
                scales.extend((width, width, height))

    scatter(bpy.context.scene, "Instanced grass", prototypes, points, variants, rotations, scales)
    return len(points)


def add_camera(scene, recipe):
    settings = recipe["camera"]
    data = bpy.data.cameras.new("Lawn-level camera")
    camera = bpy.data.objects.new("Lawn-level camera", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    data.lens = settings["lens_mm"]
    data.dof.use_dof = True
    data.dof.focus_distance = settings["focus_m"]
    data.dof.aperture_fstop = settings["f_stop"]
    first_frame = recipe.get("first_frame", 1)
    for frame in range(first_frame, first_frame + recipe["frames"]):
        t = (frame - first_frame) / recipe["fps"]
        camera.location = Vector(settings["start"]) + Vector((
            0.004 * math.sin(t * 8), settings["speed_mps"] * t, 0.0025 * math.sin(t * 22) + 0.0015 * math.sin(t * 39),
        ))
        direction = Vector(settings["direction"]) + Vector((0.013 * math.sin(t * 6.5), 0, 0.008 * math.sin(t * 13)))
        aim(camera, camera.location + direction)
        camera.rotation_euler.y += 0.008 * math.sin(t * 11)
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)


def prepare(args, recipe):
    start = perf_counter()
    scene = blank_scene()
    stone = load_stone(args.stone)
    add_floor(args.assets)
    add_border(stone)
    clumps = add_grass(args.assets, recipe["seed"])
    add_lighting(scene, args.assets, recipe["light"])
    add_camera(scene, recipe)
    devices = configure_cycles(scene, recipe, args.profile)
    if args.lighting == "fond-doux":
        soften_camera_background(scene, recipe.get("camera_background_factor", 0.72))
    frame = min(scene.frame_start + 22, scene.frame_end)
    scene.frame_set(frame)
    args.scene.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    relative_packed_paths(bpy.data.images)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.scene), compress=True, relative_remap=False)
    assembly = round(perf_counter() - start, 3)
    start = perf_counter()
    scene.render.filepath = str(args.output / "apercu.png")
    bpy.ops.render.render(write_still=True)
    report = {"blender": bpy.app.version_string, "gpu": devices, "backend": "Cycles OptiX", "profile": args.profile,
              "lighting": args.lighting,
              "clump_instances": clumps, "assembly_seconds": assembly, "preview_frame": frame,
              "preview_render_seconds": round(perf_counter() - start, 3)}
    write_json(args.output / "preparation.json", report)
    print("Open-Video PREPARE " + json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "frames"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--stone", type=Path)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="reference")
    parser.add_argument("--lighting", choices=("reference", "fond-doux"), default="reference")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    validate_recipe(recipe)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        if not all((args.assets, args.stone, args.scene)):
            parser.error("prepare requires --assets, --stone, and --scene")
        args.assets, args.stone, args.scene = args.assets.resolve(), args.stone.resolve(), args.scene.resolve()
        prepare(args, recipe)
    else:
        render_frames(args, recipe)


if __name__ == "__main__":
    main()
