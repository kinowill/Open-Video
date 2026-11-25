"""Sample Blender Bullet collisions independently of the production scene."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector


@dataclass(frozen=True)
class BodyProxy:
    """An active convex body with world-space kinematic poses before release."""

    vertices: Sequence
    motion: Mapping[int, Matrix]
    release_frame: int
    mass: float = 1.0
    friction: float = .5
    restitution: float = .15


@dataclass(frozen=True)
class BoxCollider:
    """A passive box in world metres with optional animated centre poses."""

    center: Sequence[float]
    dimensions: Sequence[float]
    friction: float = .8
    restitution: float = .1
    motion: Mapping[int, Matrix] | None = None


def _vector(values):
    point = Vector(values)
    if len(point) != 3 or not all(isfinite(component) for component in point):
        raise ValueError("Expected three finite world coordinates")
    return point


def _surface(friction, restitution):
    if not 0 <= friction <= 1 or not 0 <= restitution <= 1:
        raise ValueError("Friction and restitution must be in [0, 1]")


def _pose(matrix):
    if len(matrix) != 4 or len(matrix[0]) != 4 or not all(
        isfinite(value) for row in matrix for value in row
    ):
        raise ValueError("Expected a finite 4×4 world pose")
    location, rotation, scale = matrix.decompose()
    if any(abs(axis - 1) > 1e-4 for axis in scale):
        raise ValueError("Apply body scale to its local vertices before simulation")
    return location, rotation


def _hull(mesh, vertices):
    if len(vertices) < 4:
        raise ValueError("A body needs at least four vertices for its convex hull")
    bm = bmesh.new()
    try:
        for point in vertices:
            bm.verts.new(_vector(point))
        hull = bmesh.ops.convex_hull(bm, input=list(bm.verts))
        bmesh.ops.delete(bm, geom=hull["geom_unused"] + hull["geom_interior"], context="VERTS")
        if len(bm.faces) < 4:
            raise ValueError("A body needs a nondegenerate three-dimensional convex hull")
        bm.to_mesh(mesh)
    finally:
        bm.free()


def _box(mesh, dimensions):
    x, y, z = _vector(dimensions) * .5
    if min(x, y, z) <= 0:
        raise ValueError("Box collider dimensions must be positive")
    mesh.from_pydata(
        [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
         (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)],
        [],
        [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    )


def _key_transforms(obj, motion, first_frame, last_frame):
    if first_frame not in motion or any(type(frame) is not int for frame in motion):
        raise ValueError("Motion needs an integer pose at the first simulation frame")

    obj.rotation_mode = "QUATERNION"
    previous = None
    for frame in sorted(motion):
        if not first_frame <= frame <= last_frame:
            continue
        location, rotation = _pose(motion[frame])
        if previous is not None and previous.dot(rotation) < 0:
            rotation.negate()
        obj.location, obj.rotation_quaternion = location, rotation
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        previous = rotation


def _set_interpolation(obj):
    animation = obj.animation_data
    strip = animation.action.layers[0].strips[0]
    for curve in strip.channelbag(animation.action_slot).fcurves:
        mode = "CONSTANT" if curve.data_path == "rigid_body.kinematic" else "LINEAR"
        for key in curve.keyframe_points:
            key.interpolation = mode


def _animate(obj, body, first_frame, last_frame):
    release = body.release_frame
    if type(release) is not int or not first_frame <= release <= last_frame:
        raise ValueError("The body's release frame must be inside the simulation range")
    _key_transforms(obj, body.motion, first_frame, release)
    rigid = obj.rigid_body
    if release > first_frame:
        rigid.kinematic = True
        rigid.keyframe_insert(data_path="kinematic", frame=first_frame)
    rigid.kinematic = False
    rigid.keyframe_insert(data_path="kinematic", frame=release)
    _set_interpolation(obj)


def _rigid_bodies(objects, kind):
    if not objects:
        return
    bpy.context.view_layer.update()
    with bpy.context.temp_override(
        object=objects[0], active_object=objects[0],
        selected_objects=objects, selected_editable_objects=objects,
    ):
        if bpy.ops.rigidbody.objects_add(type=kind) != {"FINISHED"}:
            raise RuntimeError(f"Could not create {kind.lower()} rigid bodies")
    if any(obj.rigid_body is None for obj in objects):
        raise RuntimeError(f"Blender did not create all {kind.lower()} rigid bodies")


def simulate_bodies(bodies, colliders, first_frame, last_frame, fps) -> list[list[tuple[int, Vector, Quaternion]]]:
    """Return integer-frame Bullet poses from each body's release to the end."""
    if any(type(value) is not int for value in (first_frame, last_frame, fps)):
        raise ValueError("The frame range and frame rate must use integers")
    if first_frame > last_frame or fps < 1:
        raise ValueError("Expected an ordered frame range and positive frame rate")
    bodies, colliders = tuple(bodies), tuple(colliders)
    if not bodies:
        return []

    window = bpy.context.window
    if window is None:
        raise RuntimeError("Blender needs a window context to evaluate rigid bodies")
    previous_scene = window.scene
    previous_layer = window.view_layer
    scene = bpy.data.scenes.new("Open-Video · Calcul des collisions")
    objects, meshes = [], []
    world = None

    def mesh_object(name):
        mesh = bpy.data.meshes.new(name)
        meshes.append(mesh)
        obj = bpy.data.objects.new(name, mesh)
        objects.append(obj)
        scene.collection.objects.link(obj)
        return obj

    try:
        window.scene = scene
        scene.frame_start, scene.frame_end = first_frame, last_frame
        scene.render.fps = fps
        if scene.render.fps != fps:
            raise ValueError("The frame rate exceeds Blender's supported range")
        bpy.ops.rigidbody.world_add()
        world = scene.rigidbody_world
        world.substeps_per_frame = 8
        world.solver_iterations = 25
        world.point_cache.frame_start, world.point_cache.frame_end = first_frame, last_frame
        if (world.point_cache.frame_start, world.point_cache.frame_end) != (first_frame, last_frame):
            raise ValueError("The frame range exceeds Blender's rigid body cache limits")

        for collider in colliders:
            _surface(collider.friction, collider.restitution)
            obj = mesh_object("Obstacle fixe")
            _box(obj.data, collider.dimensions)
            obj.location = _vector(collider.center)
        passive = objects[:]
        _rigid_bodies(passive, "PASSIVE")
        for obj, collider in zip(passive, colliders):
            rigid = obj.rigid_body
            rigid.collision_shape = "BOX"
            rigid.friction = collider.friction
            rigid.restitution = collider.restitution
            if collider.motion is not None:
                rigid.kinematic = True
                _key_transforms(obj, collider.motion, first_frame, last_frame)
                _set_interpolation(obj)

        for body in bodies:
            if not isfinite(body.mass) or body.mass <= 0:
                raise ValueError("Body mass must be finite and positive")
            _surface(body.friction, body.restitution)
            _hull(mesh_object("Volume de collision").data, body.vertices)
        active = objects[len(passive):]
        _rigid_bodies(active, "ACTIVE")
        for obj, body in zip(active, bodies):
            rigid = obj.rigid_body
            rigid.collision_shape = "CONVEX_HULL"
            rigid.mass = body.mass
            rigid.friction = body.friction
            rigid.restitution = body.restitution
            rigid.use_margin = True
            rigid.collision_margin = min(.004, min(obj.dimensions) * .02)
            rigid.use_deactivation = True
            _animate(obj, body, first_frame, last_frame)

        tracks = [[] for _ in active]
        for frame in range(first_frame, last_frame + 1):
            scene.frame_set(frame)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            for obj, body, track in zip(active, bodies, tracks):
                if frame < body.release_frame:
                    continue
                evaluated = obj.evaluated_get(depsgraph).matrix_world
                location, rotation, _ = evaluated.decompose()
                if not all(isfinite(value) for value in (*location, *rotation)):
                    raise RuntimeError(f"Non-finite body pose at frame {frame}")
                if track and track[-1][2].dot(rotation) < 0:
                    rotation.negate()
                track.append((frame, location, rotation))
        return tracks
    finally:
        temporary_collections = (world.collection, world.constraints) if world else ()
        actions = [obj.animation_data.action for obj in objects if obj.animation_data]
        window.scene = previous_scene
        window.view_layer = previous_layer
        bpy.data.scenes.remove(scene)
        for obj in objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in meshes:
            bpy.data.meshes.remove(mesh)
        for action in actions:
            if action and action.users == 0:
                bpy.data.actions.remove(action)
        for collection in temporary_collections:
            if collection and collection.users == 0:
                bpy.data.collections.remove(collection)
