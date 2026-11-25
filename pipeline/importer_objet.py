"""Package a GLB as a self-contained Blender asset collection."""

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import bpy
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.ressources import relative_packed_paths
from pipeline.rendu import write_json
from pipeline.scene import blank_scene


def import_collection(source, name):
    scene = blank_scene()
    collection = bpy.data.collections.new(name)
    if collection.name != name:
        raise ValueError(f"Collection name cannot be stored as requested: {name}")
    scene.collection.children.link(collection)
    result = bpy.ops.import_scene.gltf(filepath=str(source), import_pack_images=True,
                                       import_scene_as_collection=False)
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender could not import {source.name}")
    for obj in list(scene.objects):
        previous_collections = list(obj.users_collection)
        collection.objects.link(obj)
        for previous in previous_collections:
            previous.objects.unlink(obj)
        if obj.type in {"LIGHT", "CAMERA"}:
            obj.hide_render = True
    collection.asset_mark()
    collection.asset_data.description = "Collection importée depuis un GLB pour les scènes Open-Video."
    return collection


def mesh_dimensions(objects):
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    minima, maxima = [], []
    for obj in objects:
        evaluated = obj.evaluated_get(graph)
        vertices = evaluated.data.vertices
        if not vertices:
            continue
        positions = np.empty((len(vertices), 3), dtype=np.float32)
        vertices.foreach_get("co", positions.ravel())
        matrix = np.array(evaluated.matrix_world, dtype=np.float64)
        points = positions @ matrix[:3, :3].T + matrix[:3, 3]
        minima.append(points.min(axis=0))
        maxima.append(points.max(axis=0))
    if not minima:
        raise ValueError("The GLB has no vertices in its main scene")
    dimensions = np.max(maxima, axis=0) - np.min(minima, axis=0)
    if not np.isfinite(dimensions).all() or dimensions.max() <= 0:
        raise ValueError("The GLB needs finite, nonzero dimensions")
    return dimensions.tolist()


def scale_collection(collection, factor):
    roots = [obj for obj in collection.objects if obj.parent is None]
    scale_root = bpy.data.objects.new("Open-Video asset scale", None)
    collection.objects.link(scale_root)
    for obj in roots:
        obj.parent = scale_root
    scale_root.scale = (factor,) * 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dimension-m", type=float, help="Desired longest world-space side in metres")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if not args.collection.strip():
        parser.error("--collection needs a nonblank name")
    if args.dimension_m is not None and (not math.isfinite(args.dimension_m) or args.dimension_m <= 0):
        parser.error("--dimension-m must be finite and positive")

    source = args.asset.resolve(strict=True)
    output = args.output.resolve()
    report_path = args.report.resolve()
    if source.suffix.casefold() != ".glb" or output.suffix.casefold() != ".blend":
        parser.error("The input must be a .glb file and the output a .blend file")
    if report_path in (source, output):
        parser.error("--report must differ from the input and output files")
    with source.open("rb") as binary:
        if binary.read(4) != b"glTF":
            raise ValueError(f"Invalid GLB header: {source.name}")

    start = perf_counter()
    collection = import_collection(source, args.collection)
    objects = [obj for obj in collection.all_objects if obj.type == "MESH"]
    triangle_counts = {}
    for mesh in {obj.data for obj in objects}:
        mesh.calc_loop_triangles()
        triangle_counts[mesh] = len(mesh.loop_triangles)
        color = mesh.color_attributes.active_color
        if color is None and len(mesh.color_attributes) == 1:
            color = mesh.color_attributes[0]
        if color is not None:
            for material in mesh.materials:
                if material is not None and material.use_nodes:
                    for node in material.node_tree.nodes:
                        if node.type == "VERTEX_COLOR" and not node.layer_name:
                            node.layer_name = color.name
    triangles = sum(triangle_counts[obj.data] for obj in objects)
    if not triangles:
        raise ValueError("The GLB has no surface triangles in its main scene")
    source_dimensions = mesh_dimensions(objects)
    dimensions = source_dimensions
    factor = 1.0
    if args.dimension_m is not None:
        factor = args.dimension_m / max(source_dimensions)
        scale_collection(collection, factor)
        dimensions = mesh_dimensions(objects)
        if not math.isclose(max(dimensions), args.dimension_m, rel_tol=1e-5):
            raise ValueError("Could not represent the requested asset size in Blender")

    images = [image for image in bpy.data.images if image.source == "FILE"]
    missing = [image.name for image in images if not image.packed_files or not all(image.size)]
    if missing:
        raise ValueError(f"Could not pack the imported textures: {', '.join(missing)}")
    for image in images:
        if not Path(image.filepath).name:
            basename = bpy.path.clean_name(image.name)
            image.filepath_raw = f"//{basename}.{image.file_format.lower()}"
    relative_packed_paths(images)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.data.libraries.write(str(output), {collection}, path_remap="NONE", fake_user=True, compress=True)
    report = {
        "blender": bpy.app.version_string,
        "source": source.name,
        "output": output.name,
        "collection": collection.name,
        "mesh_objects": len(objects),
        "mesh_datablocks": len(triangle_counts),
        "triangles": triangles,
        "packed_images": len(images),
        "source_dimensions_m": source_dimensions,
        "dimensions_m": dimensions,
        "scale_factor": factor,
        "import_seconds": round(perf_counter() - start, 3),
    }
    write_json(report_path, report)
    print("Open-Video OBJET " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
