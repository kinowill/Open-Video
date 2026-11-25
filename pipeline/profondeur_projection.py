"""Bake the first subject surface seen by each calibrated perspective photo."""

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree


MAX_DEPTH_RESOLUTION = 512


def _subject_bvh(meshes):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices, triangles = [], []
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            offset = len(vertices)
            world = evaluated.matrix_world
            vertices.extend(world @ vertex.co for vertex in mesh.vertices)
            triangles.extend(tuple(offset + index for index in triangle.vertices)
                             for triangle in mesh.loop_triangles)
        finally:
            evaluated.to_mesh_clear()
    if not triangles:
        raise ValueError("Photo projection needs a subject with evaluated mesh faces")
    return BVHTree.FromPolygons(vertices, triangles, all_triangles=True)


def _depth_image(bvh, view, size, index):
    width, height = size
    rows = np.asarray(view["projection"], dtype=np.float64)[[0, 1, 3]]
    try:
        inverse = np.linalg.inv(rows[:, :3])
    except np.linalg.LinAlgError as error:
        raise ValueError(f"Singular perspective projection: {view['file']}") from error
    # Inverse clip coordinates give an unnormalized ray for each pixel center.
    # Blender image pixels start at the lower left, as do OpenGL clip coordinates.
    x = (np.arange(width) + 0.5) * (2 / width) - 1
    y = (np.arange(height) + 0.5) * (2 / height) - 1
    rays = (inverse[:, 2] + x[None, :, None] * inverse[:, 0]
            + y[:, None, None] * inverse[:, 1]).reshape(-1, 3)
    origin = view["camera"]
    directions = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    depth_row = rows[2, :3].tolist()
    depth_scale = directions @ rows[2, :3]
    origin_depth = sum(coefficient * coordinate for coefficient, coordinate in zip(depth_row, origin)) + rows[2, 3]
    depths = np.zeros(width * height, dtype=np.float32)
    for pixel, direction in enumerate(directions.tolist()):
        _, _, _, distance = bvh.ray_cast(origin, direction)
        if distance is not None:
            depths[pixel] = origin_depth + distance * depth_scale[pixel]

    # The stored depth and shader both use clip W. The pixel footprint controls
    # the tolerance when adjacent pixels see different depths on a sloped face.
    footprint = 2 * np.linalg.norm(rows[2, :3]) * max(np.linalg.norm(inverse[:, 0]) / width,
                                                   np.linalg.norm(inverse[:, 1]) / height)
    image = bpy.data.images.new(f"Open-Video depth {view['file']}", width, height, alpha=False,
                                float_buffer=True, is_data=True)
    image.filepath = f"//open-video-photo-depth-{index:04d}.exr"
    pixels = np.ones((width * height, 4), dtype=np.float32)
    pixels[:, :3] = depths[:, None]
    image.pixels.foreach_set(pixels.ravel())
    image.update()
    image.pack()
    return image, float(footprint)


def bake_depth_images(meshes, views, size, indices):
    scale = min(1, MAX_DEPTH_RESOLUTION / max(size))
    dimensions = tuple(max(1, round(side * scale)) for side in size)
    bvh = _subject_bvh(meshes)
    return {index: _depth_image(bvh, views[index], dimensions, index) for index in indices}, dimensions
