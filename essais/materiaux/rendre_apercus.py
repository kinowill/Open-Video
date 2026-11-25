"""Pack the PBR maps in a Blender library and load it for each still."""

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
from pipeline.ressources import STONE_MATERIAL_NAME, load_stone, relative_packed_paths
from pipeline.scene import aim, blank_scene, color_material, light, stone_slab, world


RESOLUTION = (720, 960)
SAMPLES = 32


def save_library(output):
    blank_scene()
    material = color_material(STONE_MATERIAL_NAME, (0.35, 0.31, 0.27), 0.75)
    material.asset_mark()
    material.asset_data.description = "Pierre procédurale avec trois cartes PBR incorporées."
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = nodes.get("Principled BSDF")
    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.location = (-290, -340)
    for index, (channel, color_space, target) in enumerate((
        ("basecolor", "sRGB", shader.inputs["Base Color"]),
        ("roughness", "Non-Color", shader.inputs["Roughness"]),
        ("normal", "Non-Color", normal_map.inputs["Color"]),
    )):
        texture = nodes.new("ShaderNodeTexImage")
        texture.location = (-620, 170 - index * 270)
        texture.label = channel
        texture.image = bpy.data.images.load(str(output / f"{channel}.png"))
        texture.image.colorspace_settings.name = color_space
        texture.image.pack()
        links.new(texture.outputs["Color"], target)
    links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])

    library = output / "stone.blend"
    relative_packed_paths(bpy.data.images)
    bpy.data.libraries.write(str(library), {material}, path_remap="NONE", fake_user=True, compress=True)
    return library


def camera(scene, position, target, lens, focus, aperture):
    data = bpy.data.cameras.new("Camera")
    view = bpy.data.objects.new("Camera", data)
    scene.collection.objects.link(view)
    view.location = position
    aim(view, target)
    data.lens = lens
    data.dof.use_dof = True
    data.dof.focus_distance = (Vector(focus) - view.location).length
    data.dof.aperture_fstop = aperture
    scene.camera = view


def ground(material, z):
    bpy.ops.mesh.primitive_plane_add(size=180, location=(0, 0, z))
    bpy.context.object.data.materials.append(material)


def grass():
    generator = random.Random(194)
    grass_materials = [color_material(f"Grass {index}", color, 0.72) for index, color in enumerate((
        (0.053, 0.118, 0.022), (0.086, 0.168, 0.026), (0.111, 0.191, 0.041), (0.16, 0.208, 0.061),
    ))]
    points, faces, indices = [], [], []
    for _ in range(11500):
        x = generator.uniform(-2.0, -0.255)
        y = generator.uniform(-3.4, 5.2)
        angle = generator.uniform(0, math.tau)
        dx, dy = math.cos(angle), math.sin(angle)
        h = generator.uniform(0.025, 0.145)
        w = generator.uniform(0.0021, 0.0050)
        lean = generator.uniform(0.03, 0.72) * h
        first = len(points)
        points.extend(((x - dy*w, y + dx*w, -0.018), (x + dy*w, y - dx*w, -0.018),
                       (x + dx*lean/3 + dy*w*0.58, y + dy*lean/3 - dx*w*0.58, h*0.56),
                       (x + dx*lean/3 - dy*w*0.58, y + dy*lean/3 + dx*w*0.58, h*0.56),
                       (x + dx*lean, y + dy*lean, h)))
        faces.extend(((first, first+1, first+2, first+3), (first+3, first+2, first+4)))
        indices.extend([generator.randrange(len(grass_materials))] * 2)
    mesh = bpy.data.meshes.new("Grass blades")
    mesh.from_pydata(points, [], faces)
    for material in grass_materials:
        mesh.materials.append(material)
    for face, index in zip(mesh.polygons, indices, strict=True):
        face.material_index = index
    bpy.context.scene.collection.objects.link(bpy.data.objects.new("Grass", mesh))


def garden(scene, material):
    ground(color_material("Soil", (0.072, 0.047, 0.027), 0.93), -0.016)
    grass()
    for index in range(11):
        stone = stone_slab((0, index*0.584 - 2.5, 0.036), (0.49, 0.562, 0.104), material, 0.018)
        stone.rotation_euler.z = math.radians((index % 3 - 1) * 1.3)
    world(scene, (0.55, 0.70, 1), 0.36)
    light(scene, "Afternoon sun", "SUN", (-3, -2, 4), (0, 0, 0), 2.5, (1, 0.81, 0.63), 0.09)
    camera(scene, (-0.43, -1.79, 0.24), (0.04, 0.6, -0.035), 40, (0.02, -0.80, 0.085), 4.0)


def studio(scene, material):
    ground(color_material("Studio floor", (0.027, 0.038, 0.049), 0.72), 0)
    sample = stone_slab((0, 0, 0.105), (1.44, 0.94, 0.2), material, 0.027)
    sample.rotation_euler.z = math.radians(-12)
    world(scene, (0.23, 0.29, 0.42), 0.22)
    light(scene, "Softbox", "AREA", (-2.2, -2, 3.8), (0, 0, 0), 520, (1, 0.86, 0.69), 2.9)
    light(scene, "Fill", "AREA", (2.0, 0.75, 2.2), (0, 0, 0.1), 320, (0.68, 0.81, 1), 1.8)
    light(scene, "Rim", "AREA", (-0.9, 2.8, 1.6), (0, 0, 0.09), 430, (0.86, 0.93, 1), 1.3)
    camera(scene, (1.86, -2.68, 1.71), (0, 0, 0.06), 52, (0, 0, 0.16), 8.0)


def render(output, library, name, compose):
    start = perf_counter()
    scene = blank_scene()
    material = load_stone(library)
    compose(scene, material)

    settings = bpy.context.preferences.addons["cycles"].preferences
    settings.compute_device_type = "OPTIX"
    settings.refresh_devices()
    devices = [device.name for device in settings.devices if device.type == "OPTIX"]
    if not devices:
        raise RuntimeError("No OptiX GPU available")
    for device in settings.devices:
        device.use = device.type == "OPTIX"
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output / f"{name}.png")
    assembly_seconds = round(perf_counter() - start, 3)
    start = perf_counter()
    bpy.ops.render.render(write_still=True)
    return {
        "file": f"{name}.png",
        "gpu": devices,
        "assembly_seconds": assembly_seconds,
        "render_seconds": round(perf_counter() - start, 3),
    }


def main():
    output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    start = perf_counter()
    library = save_library(output)
    library_seconds = round(perf_counter() - start, 3)
    report = {
        "blender": bpy.app.version_string,
        "backend": "Cycles OptiX",
        "resolution": RESOLUTION,
        "samples": SAMPLES,
        "material_library": library.name,
        "packed_maps": 3,
        "library_seconds": library_seconds,
        "stills": [render(output, library, "jardin", garden), render(output, library, "studio", studio)],
    }
    (output / "mesures.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
