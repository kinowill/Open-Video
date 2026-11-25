"""Project a calibrated photo set onto a GLB as an embedded UV atlas."""

import argparse
import hashlib
import json
from math import ceil, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt, minimum_filter
from trimesh.exchange.gltf import export_glb


def project(points, projection, width, height):
    clip = points @ projection[:, :3].T + projection[:, 3]
    depth = clip[:, 3]
    divisor = np.where(depth > 0, depth, 1)
    xy = np.column_stack((0.5 + 0.5 * clip[:, 0] / divisor, 0.5 - 0.5 * clip[:, 1] / divisor))
    xy *= (width, height)
    valid = (depth > 0) & (xy >= 0).all(axis=1) & (xy < (width, height)).all(axis=1)
    return xy, depth, valid


def read_capture(directory, view, size):
    with Image.open(directory / view["file"]) as image:
        if image.size != size or "A" not in image.getbands():
            raise ValueError(f"Expected an RGBA capture at the stated resolution: {view['file']}")
        rgba = np.asarray(image.convert("RGBA"))
    mask = rgba[..., 3] >= 128
    if not mask.any() or mask.all():
        raise ValueError(f"Expected an object and a transparent background: {view['file']}")
    matrix = np.asarray(view["projection"], dtype=np.float32)
    camera = np.asarray(view["camera"], dtype=np.float32)
    if matrix.shape != (4, 4) or camera.shape != (3,) or not np.isfinite(matrix).all() or not np.isfinite(camera).all():
        raise ValueError(f"Invalid calibration for {view['file']}")
    return rgba[..., :3], mask, matrix, camera


def put_tile(atlas, rgb, mask, tile_x, tile_y, padding, size):
    # Dilate the colours under the alpha edge to prevent dark filtering fringes.
    nearest = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    opaque = rgb[nearest[0], nearest[1]]
    if size != (rgb.shape[1], rgb.shape[0]):
        opaque = np.asarray(Image.fromarray(opaque).resize(size, Image.Resampling.LANCZOS))
    tile = np.pad(opaque, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
    height, width = tile.shape[:2]
    atlas[tile_y:tile_y + height, tile_x:tile_x + width] = tile


def depth_map(points, projection, width, height):
    xy, depth, valid = project(points, projection, width, height)
    x, y = xy[valid].astype(np.int32).T
    buffer = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(buffer, (y, x), depth[valid])
    return minimum_filter(buffer, size=3)


def choose_views(source, scores):
    view_count = scores.shape[1]
    selected = scores.argmax(axis=1)
    covered = scores.max(axis=1) > 0
    selected[~covered] = view_count
    if view_count == 1 or not len(source.face_adjacency):
        return selected

    first, second = source.face_adjacency.T
    facing = np.einsum("ij,ij->i", source.face_normals[first], source.face_normals[second]).clip(0, 1)
    # Smooth camera seams across a broad surface and allow changes at sharp edges.
    neighbors = np.concatenate((second, first))
    owners = np.concatenate((first, second))
    influence = np.concatenate((facing, facing)) ** 2
    total = np.maximum(np.bincount(owners, weights=influence, minlength=len(scores)), 1e-8)
    for _ in range(7):
        support = np.zeros((len(scores), view_count + 1))
        np.add.at(support, (owners, selected[neighbors]), influence)
        merit = np.where(scores > 0, scores + 0.22 * support[:, :view_count] / total[:, None], -1)
        proposal = merit.argmax(axis=1)
        proposal[~covered] = view_count
        if np.array_equal(proposal, selected):
            break
        selected = proposal
    return selected


def build_atlas(source, manifest, captures, selection):
    # The source GLB is Y-up, but the recorded camera matrices use Blender Z-up.
    vertices = np.asarray(source.vertices, dtype=np.float32)[:, [0, 2, 1]] * [1, -1, 1]
    normals = np.asarray(source.vertex_normals, dtype=np.float32)[:, [0, 2, 1]] * [1, -1, 1]
    triangles = vertices[source.faces]
    centers = triangles.mean(axis=1)
    face_normals = normals[source.faces].mean(axis=1)
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-8)
    edge_points = (triangles + np.roll(triangles, 1, axis=1)) * 0.5
    surface_samples = np.concatenate((vertices, centers, edge_points.reshape(-1, 3)))
    radius = np.linalg.norm(np.ptp(vertices, axis=0))
    tolerance = max(radius * 0.024, 1e-5)
    views = [view for view in manifest["views"] if view.get("split", "train") in {"train", "texture"}]
    if not views:
        raise ValueError("The manifest contains no training photos")
    width, height = manifest["width"], manifest["height"]
    if width <= 0 or height <= 0:
        raise ValueError("The capture dimensions must be positive")
    padding = 4
    columns = ceil(sqrt(len(views) + 1))
    rows = ceil((len(views) + 1) / columns)
    scale = min(1, (8192 // columns - 2 * padding) / width, (8192 // rows - 2 * padding) / height)
    tile_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    tile_w, tile_h = tile_size[0] + 2 * padding, tile_size[1] + 2 * padding
    atlas = np.full((rows * tile_h, columns * tile_w, 3), 48, dtype=np.uint8)
    uv_scales = np.array([atlas.shape[1], atlas.shape[0]])
    scores = np.zeros((len(centers), len(views)), dtype=np.float32)
    directions = np.asarray([view["camera"] for view in views]) - vertices.mean(axis=0)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions_by_axis = np.concatenate((directions, -directions), axis=1)
    # Break near-ties consistently for large surfaces such as a flat lid.
    anchor_views = np.round(directions_by_axis, 2).argmax(axis=0)
    dominant_axes = np.abs(face_normals).argmax(axis=1)
    dominant_zones = dominant_axes + 3 * (face_normals[np.arange(len(face_normals)), dominant_axes] < 0)
    anchor = anchor_views[dominant_zones]
    coords = []
    for index, view in enumerate(views):
        rgb, mask, projection, camera = read_capture(captures, view, (width, height))
        tile_x, tile_y = index % columns * tile_w, index // columns * tile_h
        put_tile(atlas, rgb, mask, tile_x, tile_y, padding, tile_size)
        xy, depth, valid = project(centers, projection, width, height)
        point_xy, _, point_valid = project(vertices, projection, width, height)
        valid &= point_valid[source.faces].all(axis=1)
        candidates = np.flatnonzero(valid)
        x, y = xy[candidates].astype(np.int32).T
        rays = camera - centers[candidates]
        distance = np.linalg.norm(rays, axis=1)
        cosines = np.einsum("ni,ni->n", rays, face_normals[candidates]) / distance
        closest = depth_map(surface_samples, projection, width, height)
        visible = (cosines > 0.1) & mask[y, x] & (depth[candidates] <= closest[y, x] + tolerance)
        if selection == "coherent":
            strength = cosines + 0.25 * (index == anchor[candidates])
        else:
            strength = cosines / np.maximum(distance, 1e-8)
        scores[candidates, index] = np.where(visible, strength, 0)
        uv = (point_xy * (tile_size[0] / width, tile_size[1] / height)
              + (tile_x + padding, tile_y + padding)) / uv_scales
        uv[:, 1] = 1 - uv[:, 1]
        coords.append(uv)

    if selection == "coherent":
        selected = choose_views(source, scores)
    else:
        selected = scores.argmax(axis=1)
        selected[scores.max(axis=1) <= 0] = len(views)
    fallback = len(views)
    fallback_x = (fallback % columns + 0.5) * tile_w
    fallback_y = (fallback // columns + 0.5) * tile_h
    coords.append(np.broadcast_to((fallback_x / uv_scales[0], 1 - fallback_y / uv_scales[1]), (len(vertices), 2)))
    # Only split a vertex if adjacent faces sample different source cameras.
    pairs = np.column_stack((np.repeat(selected, 3), source.faces.ravel()))
    unique, indices = np.unique(pairs, axis=0, return_inverse=True)
    cameras, original_indices = unique.T
    uv = np.stack(coords)[cameras, original_indices]
    material = trimesh.visual.material.PBRMaterial(
        name="Open-Video calibrated photo atlas", baseColorTexture=Image.fromarray(atlas), roughnessFactor=0.75,
        metallicFactor=0,
    )
    result = trimesh.Trimesh(
        vertices=source.vertices[original_indices], faces=indices.reshape(-1, 3),
        vertex_normals=source.vertex_normals[original_indices],
        visual=trimesh.visual.texture.TextureVisuals(uv=uv, material=material), process=False,
    )
    coverage = np.bincount(selected, minlength=len(views) + 1)
    return result, atlas.shape[1::-1], {
        "faces_per_view": {view["file"]: int(count) for view, count in zip(views, coverage[:-1])},
        "uncovered_faces": int(coverage[-1]),
    }


def save_glb(mesh, destination, lighting):
    def apply_lighting(tree):
        if lighting == "capture":
            tree.setdefault("extensionsUsed", []).append("KHR_materials_unlit")
            tree.setdefault("extensionsRequired", []).append("KHR_materials_unlit")
            for material in tree["materials"]:
                material.setdefault("extensions", {})["KHR_materials_unlit"] = {}

    glb = export_glb(mesh.scene(), include_normals=True, tree_postprocessor=apply_lighting)
    destination.write_bytes(glb)
    verified = trimesh.load_scene(destination, process=False).to_mesh()
    # Trimesh's to_mesh may discard visuals but the mesh geometry must stay intact.
    if len(verified.faces) != len(mesh.faces) or not np.allclose(verified.bounds, mesh.bounds, atol=1e-6):
        raise ValueError("Exported GLB did not preserve the input geometry")
    return hashlib.sha256(glb).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lighting", choices=("capture", "studio"), default="capture",
                        help="Use capture for lit source photos and studio for diffuse colour maps")
    parser.add_argument("--selection", choices=("coherent", "direct"), default="coherent",
                        help="Prefer continuous source views across neighboring faces")
    args = parser.parse_args()
    start = perf_counter()
    if args.asset.suffix.lower() != ".glb":
        parser.error("Expect a binary GLB asset")
    if args.asset.resolve() == (args.output / "objet.glb").resolve():
        parser.error("Use a separate output directory for the textured asset")
    source = trimesh.load_scene(args.asset, process=False).to_mesh()
    if not len(source.faces) or not np.isfinite(source.vertices).all():
        raise ValueError("Expected a finite triangular mesh")
    manifest = json.loads(args.captures.read_text(encoding="utf-8"))
    mesh, size, coverage = build_atlas(source, manifest, args.captures.parent, args.selection)
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".texture-", dir=args.output) as directory:
        stage = Path(directory)
        digest = save_glb(mesh, stage / "objet.glb", args.lighting)
        report = {
            "project": "Open-Video",
            "lighting": args.lighting,
            "selection": args.selection,
            "texture_resolution": list(size),
            "triangles": len(mesh.faces),
            "vertices": len(mesh.vertices),
            "total_seconds": round(perf_counter() - start, 3),
            "sha256": digest,
            **coverage,
        }
        (stage / "texture.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for name in ("objet.glb", "texture.json"):
            (stage / name).replace(args.output / name)
    print(f"Open-Video PHOTO UV {json.dumps(report)}", flush=True)


if __name__ == "__main__":
    main()
