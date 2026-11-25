"""Exercise the shared Bullet sampler in Blender's background mode."""

from itertools import product
from pathlib import Path
import sys

import bpy
from mathutils import Matrix

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.corps_rigides import BodyProxy, BoxCollider, simulate_bodies


def verify_collisions():
    cube = tuple(product((-.15, .15), repeat=3))
    floor = BoxCollider((0, 0, -.5), (20, 20, 1))

    def approach(sign):
        motion = {frame: Matrix.Translation((sign * (.7 - .1 * (frame - 1)), 0, 1.4)) for frame in range(1, 5)}
        return BodyProxy(cube, motion, 5, friction=.42, restitution=.75)

    left, right = approach(-1), approach(1)
    tracks = simulate_bodies([left, right], [floor], 1, 90, 24)
    control, = simulate_bodies([left], [floor], 1, 90, 24)
    assert tracks[0][0][1].x > left.motion[4].translation.x
    assert all(abs(track[-1][1].z - .15) < .02 for track in tracks)
    difference = abs(tracks[0][-1][1].x - control[-1][1].x)
    assert difference > .5
    print(f"Open-Video BULLET pair shifts one body by {difference:.3f} m")


def verify_moving_collider():
    parcel = tuple(product((-.17, .17), (-.14, .14), (-.115, .115)))
    bodies = [BodyProxy(parcel, {1: Matrix.Translation((1.12 + .42 * i, 0, 1.39))}, 1, friction=.54)
              for i in range(3)]
    floor = BoxCollider((0, 0, -.1), (20, 12, .2))
    dimensions = (2.9, 1.4, .11)
    pivot, offset = Matrix.Translation((0, 0, 1.2)), Matrix.Translation((1.45, 0, 0))
    motion = {frame: pivot @ Matrix.Rotation(-.73 * min(1, max(0, (frame - 7) / 22)), 4, 'Y') @ offset
              for frame in range(1, 30)}
    ramp = BoxCollider((1.45, 0, 1.2), dimensions, .54, motion=motion)
    platform = BoxCollider((1.45, 0, 1.2), dimensions, .54)
    dynamic = simulate_bodies(bodies, [floor, ramp], 1, 70, 24)
    control = simulate_bodies(bodies, [floor, platform], 1, 70, 24)
    for track, body in zip(dynamic, bodies, strict=True):
        assert (track[0][1] - body.motion[1].translation).length < 1e-5
        assert track[-1][1].z < .26
    assert all(track[-1][1].z > 1.35 for track in control)
    contact_frames = [next(frame for frame, position, _ in track if position.z < .22) for track in dynamic]
    print(f"Open-Video BULLET ramp contacts ground at frames {contact_frames}")


if __name__ == '__main__':
    scene = bpy.context.scene
    layer = bpy.context.view_layer
    scene.frame_set(28)
    blocks = [bpy.data.scenes, bpy.data.objects, bpy.data.meshes, bpy.data.actions, bpy.data.collections]
    counts = tuple(map(len, blocks))
    verify_collisions()
    verify_moving_collider()
    assert bpy.context.scene == scene and bpy.context.view_layer == layer and scene.frame_current == 28
    assert tuple(map(len, blocks)) == counts
