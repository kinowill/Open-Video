"""Render a prepared scene or compare Cycles and lighting settings."""

import argparse
import json
import sys
from math import isfinite
from pathlib import Path
from time import perf_counter

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.reprise import atomic_json, frame_hash, read_checkpoint, render_fingerprint, verified_frame


PROFILES = json.loads(Path(__file__).with_name("profils.json").read_text(encoding="utf-8"))


def write_json(path, contents):
    path.write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")


def validate_recipe(recipe):
    width, height = recipe["resolution"]
    values = (width, height, recipe["fps"], recipe["frames"], recipe.get("first_frame", 1),
              recipe.get("source_fps", recipe["fps"]))
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Resolution, fps, source_fps, frames and first_frame must be positive integers")
    if recipe["frames"] > 1 and (width % 2 or height % 2):
        raise ValueError("H.264 video needs even width and height")
    for name in ("denoising_gpu", "persistent_data"):
        if name in recipe and type(recipe[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    for name, minimum, maximum in (("motion_blur_shutter", 0, 3.4e38), ("pixel_filter_width", .01, 10)):
        if name in recipe:
            value = recipe[name]
            if type(value) not in (int, float) or not isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be a finite number in [{minimum}, {maximum}]")


def source_sample(recipe, index):
    fps = recipe["fps"]
    source_fps = recipe.get("source_fps", fps)
    offset, remainder = divmod((index - 1) * source_fps, fps)
    return recipe.get("first_frame", 1) + offset, remainder / fps


def configure_cycles(scene, recipe, profile="reference"):
    settings = bpy.context.preferences.addons["cycles"].preferences
    settings.compute_device_type = "OPTIX"
    settings.refresh_devices()
    devices = [device.name for device in settings.devices if device.type == "OPTIX"]
    if not devices:
        raise RuntimeError("No OptiX device found")
    for device in settings.devices:
        device.use = device.type == "OPTIX"
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True
    scene.cycles.denoising_use_gpu = recipe.get("denoising_gpu", scene.cycles.denoising_use_gpu)
    scene.render.use_persistent_data = recipe.get("persistent_data", True)
    scene.render.use_motion_blur = recipe.get("motion_blur", recipe["frames"] > 1)
    scene.render.motion_blur_shutter = recipe.get("motion_blur_shutter", 0.32)
    scene.cycles.filter_width = recipe.get("pixel_filter_width", scene.cycles.filter_width)
    scene.render.resolution_x, scene.render.resolution_y = recipe["resolution"]
    scene.render.resolution_percentage = 100
    source_fps = recipe.get("source_fps", recipe["fps"])
    scene.render.fps = source_fps
    if scene.render.fps != source_fps:
        raise ValueError(f"Unsupported Blender timeline frame rate: {source_fps}")
    if "source_fps" in recipe:
        scene.render.fps_base = 1.0
    scene.frame_start = recipe.get("first_frame", 1)
    scene.frame_end, _ = source_sample(recipe, recipe["frames"])
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    set_profile(scene, recipe, profile)
    return devices


def set_profile(scene, recipe, profile):
    values = PROFILES[profile].copy()
    if profile == "reference" and "samples" in recipe:
        values["samples"] = recipe["samples"]
    values.update(recipe.get("cycles", {}).get(profile, {}))
    for name, value in values.items():
        if name not in PROFILES[profile]:
            raise ValueError(f"Unsupported Cycles profile property: {name}")
        setattr(scene.cycles, name, value)
    return values


def soften_camera_background(scene, factor=0.72):
    if factor < 0:
        raise ValueError("Camera background strength must be non-negative")
    if not scene.world or not scene.world.use_nodes:
        return False
    nodes, links = scene.world.node_tree.nodes, scene.world.node_tree.links
    output = next((node for node in nodes if node.type == "OUTPUT_WORLD" and node.is_active_output), None)
    if output is None or not output.inputs["Surface"].is_linked:
        return False
    incoming = output.inputs["Surface"].links[0]
    environment = incoming.from_node
    if environment.type != "BACKGROUND":
        return False
    visible = nodes.new("ShaderNodeBackground")
    visible.label = "Open-Video - Camera background"
    visible.inputs["Strength"].default_value = environment.inputs["Strength"].default_value * factor
    color = environment.inputs["Color"]
    if color.is_linked:
        links.new(color.links[0].from_socket, visible.inputs["Color"])
    else:
        visible.inputs["Color"].default_value = color.default_value
    path = nodes.new("ShaderNodeLightPath")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(path.outputs["Is Camera Ray"], mix.inputs[0])
    links.new(environment.outputs[0], mix.inputs[1])
    links.new(visible.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])
    return True


def render_frame(scene, frame, output, *, subframe=0.0):
    scene.frame_set(frame, subframe=subframe)
    scene.render.filepath = str(output)
    start = perf_counter()
    bpy.ops.render.render(write_still=True)
    return round(perf_counter() - start, 3)


def render_dependencies():
    packed_paths, external_paths = set(), set()
    for source in (*bpy.data.images, *bpy.data.fonts, *bpy.data.sounds, *bpy.data.libraries):
        if not source.filepath or source.filepath == "<builtin>":
            continue
        path = bpy.path.abspath(source.filepath, library=source.library)
        if source.packed_file:
            packed_paths.add(path)
        else:
            external_paths.add(path)
    reported_paths = set(bpy.utils.blend_paths(absolute=True, packed=True))
    # Blender can still return file names held in packed image and font data.
    return external_paths | (reported_paths - packed_paths - {"", "<builtin>"})


def resume_signature(scene, recipe, args, devices):
    for image in bpy.data.images:
        if image.source in {"SEQUENCE", "TILED"}:
            return None, image.filepath or image.name
    for clip in bpy.data.movieclips:
        if clip.source == "SEQUENCE":
            return None, clip.filepath or clip.name
    if scene.sequence_editor:
        for strip in scene.sequence_editor.strips_all:
            if strip.type == "IMAGE" and len(strip.elements) > 1:
                return None, strip.name
    settings = {
        "version": bpy.app.version_string,
        "build": bpy.app.build_hash.decode("ascii"),
        "devices": sorted(devices),
        "resolution": recipe["resolution"],
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "source_fps": scene.render.fps,
        "fps_base": scene.render.fps_base,
        "output_fps": recipe["fps"],
        "output_frames": recipe["frames"],
        "motion_blur": scene.render.use_motion_blur,
        "motion_blur_shutter": scene.render.motion_blur_shutter,
        "pixel_filter_width": scene.cycles.filter_width,
        "persistent_data": scene.render.use_persistent_data,
        "denoising_gpu": scene.cycles.denoising_use_gpu,
        "profile": args.profile,
        "lighting": args.lighting,
        "background_factor": recipe.get("camera_background_factor", 0.72) if args.lighting == "fond-doux" else None,
        "cycles": {name: getattr(scene.cycles, name) for name in PROFILES[args.profile]},
    }
    scripts = [Path(__file__).with_name(name) for name in ("rendu.py", "reprise.py", "profils.json")]
    return render_fingerprint(bpy.data.filepath, render_dependencies(), settings, scripts)


def render_frames(args, recipe):
    scene = bpy.context.scene
    devices = configure_cycles(scene, recipe, args.profile)
    resampled = scene.render.fps != recipe["fps"]
    first = 1 if resampled else scene.frame_start
    all_frames = range(first, first + recipe["frames"])
    selected = set(args.images or all_frames)
    if any(frame not in all_frames for frame in selected):
        units = "output images" if resampled else "Blender frames"
        raise ValueError(f"Selected images must be within {units} {first}–{all_frames[-1]}")
    resume = args.resume or bool(args.images)
    if args.lighting == "fond-doux" and not soften_camera_background(scene, recipe.get("camera_background_factor", 0.72)):
        raise RuntimeError("fond-doux requires an active World Background shader")
    signature, unresolved = resume_signature(scene, recipe, args, devices)
    if resume and unresolved:
        print(f"Open-Video REPRISE unresolved source, rendering requested frames: {unresolved}", flush=True)
    journal = args.output / "reprise.json"
    saved = read_checkpoint(journal, signature) if resume else {}
    checkpoint = {"signature": signature, "frames": saved}
    complete_report = args.output / "frames.json"
    complete_report.unlink(missing_ok=True)
    report = args.output / "controle-images.json" if args.images else complete_report
    report.unlink(missing_ok=True)
    atomic_json(journal, checkpoint)

    measurements = []
    reused = 0
    previous_frame = scene.frame_start - 1
    for index, number in enumerate(range(first, max(selected) + 1), 1):
        frame, subframe = source_sample(recipe, index)
        for skipped in range(previous_frame + 1, frame):
            scene.frame_set(skipped)
        previous_frame = frame
        if number not in selected:
            scene.frame_set(frame, subframe=subframe)
            continue
        image = args.output / f"{index:04d}.png"
        key = str(index)
        entry = saved.get(key)
        sample = {"source_frame": frame}
        if resampled:
            sample["source_subframe"] = subframe
        if verified_frame(entry, image, frame, recipe["resolution"], source_subframe=subframe):
            # Advance the timeline for scenes with live simulations.
            scene.frame_set(frame, subframe=subframe)
            measurements.append({"frame": index, **sample, "seconds": entry["seconds"]})
            reused += 1
            print(f"Open-Video FRAME {index}/{recipe['frames']} reused", flush=True)
            continue

        temporary = args.output / f".{index:04d}.part.png"
        temporary.unlink(missing_ok=True)
        try:
            elapsed = render_frame(scene, frame, temporary, subframe=subframe)
            digest = frame_hash(temporary, recipe["resolution"])
            if digest is None:
                raise RuntimeError(f"Rendered PNG is missing or incomplete for frame {frame}")
            temporary.replace(image)
        finally:
            temporary.unlink(missing_ok=True)
        measurements.append({"frame": index, **sample, "seconds": elapsed})
        saved[key] = {**sample, "seconds": elapsed, "sha256": digest}
        atomic_json(journal, checkpoint)
        print(f"Open-Video FRAME {index}/{recipe['frames']} {elapsed:.3f}s", flush=True)
    atomic_json(report, {"gpu": devices, "profile": args.profile, "lighting": args.lighting,
                         "samples": scene.cycles.samples, "frames_reused": reused, "frames": measurements})
    if resume:
        print(f"Open-Video REPRISE {reused}/{len(selected)} frames reused", flush=True)
    if args.images:
        print(f"Open-Video CONTROLE {len(measurements)} images checked", flush=True)


def render_ao(scene, frame, output):
    bpy.context.view_layer.use_pass_ambient_occlusion = True
    scene.use_nodes = True
    tree = scene.node_tree
    layers = tree.nodes.new("CompositorNodeRLayers")
    passes = tree.nodes.new("CompositorNodeOutputFile")
    passes.base_path = str(output)
    passes.format.file_format = "PNG"
    passes.file_slots[0].path = "occlusion-"
    tree.links.new(layers.outputs["AO"], passes.inputs[0])
    scene.frame_set(frame)
    bpy.ops.render.render()
    path = output / f"occlusion-{frame:04d}.png"
    if not path.is_file():
        raise RuntimeError("Ambient occlusion output missing")
    return path.name


def audit(args, recipe):
    scene = bpy.context.scene
    devices = configure_cycles(scene, recipe)
    middle = (scene.frame_start + scene.frame_end) // 2
    frames = sorted({scene.frame_start, middle, scene.frame_end})
    scene.frame_set(middle)
    bpy.ops.render.render()
    results = []
    profiles = list(PROFILES)
    values = {}
    for index, frame in enumerate(frames):
        for profile in profiles[index:] + profiles[:index]:
            values[profile] = set_profile(scene, recipe, profile)
            filename = f"{profile}-{frame:04d}.png"
            elapsed = render_frame(scene, frame, args.output / filename)
            results.append({"frame": frame, "profile": profile, "file": filename, "seconds": elapsed})
            print(f"Open-Video AUDIT {profile} {frame} {elapsed:.3f}s", flush=True)
    set_profile(scene, recipe, "reference")
    lighting = None
    factor = recipe.get("camera_background_factor", 0.72)
    if soften_camera_background(scene, factor):
        filename = f"fond-doux-{middle:04d}.png"
        render_frame(scene, middle, args.output / filename)
        lighting = {"original": f"reference-{middle:04d}.png", "adjusted": filename, "camera_strength_factor": factor}
    occlusion = render_ao(scene, middle, args.output)
    write_json(args.output / "audit.json", {"gpu": devices, "resolution": recipe["resolution"],
               "profiles": values, "renders": results, "lighting": lighting, "occlusion": occlusion})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("frames", "audit"))
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="reference")
    parser.add_argument("--lighting", choices=("reference", "fond-doux"), default="reference")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--images", nargs="+", type=int, metavar="FRAME")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.images and args.mode != "frames":
        parser.error("--images requires frames mode")
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    validate_recipe(recipe)
    if not bpy.context.scene.camera:
        raise RuntimeError("The input scene needs an active camera")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "frames":
        render_frames(args, recipe)
    else:
        audit(args, recipe)


if __name__ == "__main__":
    main()
