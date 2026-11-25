"""Régularisation du mouvement apparent d'une caméra Blender."""

from bisect import bisect_right
from math import floor, isfinite

from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


CAMERA_TRANSFORMS = {'location', 'rotation_euler', 'rotation_quaternion', 'rotation_axis_angle'}
CAMERA_OPTICS = {'lens', 'ortho_scale', 'shift_x', 'shift_y', 'dof.focus_distance'}


def _keyed_curves(block, paths, first, last):
    animation = block.animation_data
    if animation is None:
        return []
    if animation.drivers or animation.nla_tracks:
        raise ValueError('La caméra doit être animée directement, sans drivers ni pistes NLA.')
    action = animation.action
    if action is None:
        return []
    if len(action.layers) != 1 or len(action.layers[0].strips) != 1:
        raise ValueError('L’action de la caméra doit avoir une seule bande.')
    slot = animation.action_slot
    if slot is None:
        return []
    if len(slot.users()) > 1:
        raise ValueError('Les courbes animées doivent appartenir à cette seule caméra.')
    strip = action.layers[0].strips[0]
    bag = strip.channelbag(slot)
    if bag is None:
        return []
    curves = [curve for curve in bag.fcurves if curve.data_path in paths and not curve.mute]
    for curve in curves:
        if curve.modifiers or curve.sampled_points:
            raise ValueError(f'La courbe {curve.data_path} doit contenir des clés sans modificateurs.')
        if any(not first <= key.co.x <= last for key in curve.keyframe_points):
            raise ValueError(f'La courbe {curve.data_path} possède des clés hors du plan {first}–{last}.')
    return [curve for curve in curves if curve.keyframe_points]


def set_linear_keys(curve, samples, first):
    """Remplace les clés d'une courbe par un échantillon linéaire par image."""
    keys = curve.keyframe_points
    keys.clear()
    keys.add(len(samples))
    for frame, (key, value) in enumerate(zip(keys, samples), start=first):
        key.co = frame, value
        key.interpolation = 'LINEAR'
    curve.update()


def _projected_arc(scene, camera, anchors, first, last):
    render = scene.render
    scale = render.resolution_percentage / 100
    width = max(1, floor(render.resolution_x * scale)) * render.pixel_aspect_x
    height = max(1, floor(render.resolution_y * scale)) * render.pixel_aspect_y
    arc = [0.]
    previous = None
    for index in range(4 * (last - first) + 1):
        frame, quarter = divmod(index, 4)
        scene.frame_set(first + frame, subframe=quarter / 4)
        points = [world_to_camera_view(scene, camera, point) for point in anchors]
        if any(point.z <= 0 or not all(map(isfinite, point)) for point in points):
            raise ValueError('Les repères doivent rester devant la caméra pendant le plan.')
        if previous is not None:
            distance = sum(((new.x - old.x) * width) ** 2 + ((new.y - old.y) * height) ** 2
                           for new, old in zip(points, previous)) ** .5
            arc.append(arc[-1] + distance)
        previous = points
    return arc


def _ease_speed(position, length):
    unit = min(1., position / length) if length else 1.
    return unit * unit * (3 - 2 * unit)


def regularize_camera(scene, camera, first, last, anchors, blend=1., ramp=(.1, .15)):
    """Redistribue le déplacement projeté de repères fixes au fil d'un plan.

    Les extrémités et l'image de travail sont conservées. ``blend`` règle la
    part du nouveau rythme et ``ramp`` les fractions d'entrée et de sortie.
    Les courbes de pose et d'optique sont échantillonnées à chaque image.
    La caméra doit porter ses propres clés, incluses dans le plan, et rester
    indépendante d'un parent ou de contraintes. Renvoie True si elle a changé.
    """
    if (not isinstance(first, int) or not isinstance(last, int) or first > last
            or not 0 <= blend <= 1 or len(ramp) != 2 or not all(0 <= fraction <= 1 for fraction in ramp)):
        raise ValueError('Intervalle, mélange ou rampes de rééchantillonnage invalides.')
    if not blend or last - first < 2:
        return False
    if (camera.type != 'CAMERA' or camera.data.type not in {'PERSP', 'ORTHO'}
            or camera.parent or camera.constraints):
        raise ValueError('Choisir une caméra perspective ou orthographique indépendante.')
    curves = _keyed_curves(camera, CAMERA_TRANSFORMS, first, last)
    optics = _keyed_curves(camera.data, CAMERA_OPTICS, first, last)
    if optics and camera.data.users > 1 + camera.data.use_fake_user:
        raise ValueError('Les données optiques animées doivent appartenir à cette seule caméra.')
    curves.extend(optics)
    if not curves:
        return False
    anchors = [Vector(point).to_3d() for point in anchors]
    if not anchors:
        raise ValueError('Fournir au moins un repère fixe devant la caméra.')

    current = scene.frame_current, scene.frame_subframe, scene.camera
    try:
        arc = _projected_arc(scene, camera, anchors, first, last)
        if arc[-1] <= 1e-6:
            return False
        duration = last - first
        progress = [0.]
        for index in range(duration):
            t = (index + .5) / duration
            speed = _ease_speed(t, ramp[0]) * _ease_speed(1 - t, ramp[1])
            progress.append(progress[-1] + speed)
        source_frames = [first]
        for index in range(1, duration):
            distance = progress[index] / progress[-1] * arc[-1]
            slot = bisect_right(arc, distance) - 1
            weight = (distance - arc[slot]) / (arc[slot + 1] - arc[slot])
            remapped = first + (slot + weight) / 4
            source_frames.append((first + index) * (1 - blend) + remapped * blend)
        source_frames.append(last)

        samples = [[curve.evaluate(frame) for frame in source_frames] for curve in curves]
        for curve, values in zip(curves, samples):
            set_linear_keys(curve, values, first)
        return True
    finally:
        frame, subframe, active_camera = current
        scene.frame_set(frame, subframe=subframe)
        if scene.camera != active_camera:
            scene.camera = active_camera
