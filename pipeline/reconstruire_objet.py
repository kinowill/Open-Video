"""Reconstruct a coloured GLB from one masked object photograph on the CPU."""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import ModuleType


PROFILES = {"apercu": 128, "standard": 192, "detail": 256}
SOURCE_REVISION = "107cefdc244c39106fa830359024f6a2f1c78871"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="apercu")
    return parser.parse_args()


def runtime_paths(runtime):
    source = runtime / "source" / f"TripoSR-{SOURCE_REVISION}"
    weights = runtime / "weights"
    for file in (source / "tsr" / "system.py", weights / "config.yaml",
                 weights / "model.ckpt", weights / "dino-config.json"):
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f"TripoSR runtime file missing or empty: {file}")
    return source, weights


def prepare_photo(path):
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        if image.format != "PNG" or image.n_frames != 1:
            raise ValueError("Expected a single PNG image with a transparent background")
        rgba = ImageOps.exif_transpose(image).convert("RGBA")
    alpha = rgba.getchannel("A")
    alpha_minimum, alpha_maximum = alpha.getextrema()
    if alpha_minimum != 0 or alpha_maximum == 0:
        raise ValueError("PNG alpha must contain both the subject and its transparent background")

    subject = rgba.crop(alpha.getbbox())
    scale = 435 / max(subject.size)
    size = tuple(max(1, round(length * scale)) for length in subject.size)
    subject = subject.resize(size, Image.Resampling.LANCZOS)
    photo = Image.new("RGB", (512, 512), (128, 128, 128))
    photo.paste(subject, ((512 - subject.width) // 2, (512 - subject.height) // 2), subject.getchannel("A"))
    return photo


def load_triposr(source, weights):
    import torch
    from skimage.measure import marching_cubes

    def extract_surface(volume, level):
        """Preserve the coordinate convention of the upstream CPU helper."""
        vertices, faces, _, _ = marching_cubes(volume.detach().cpu().numpy(), level, allow_degenerate=False)
        return torch.from_numpy(vertices[:, ::-1].copy()), torch.from_numpy(faces.copy()).long()

    def local_dino_config(repo_id, filename):
        if (repo_id, filename) != ("facebook/dino-vitb16", "config.json"):
            raise ValueError(f"Unexpected image tokenizer artifact: {repo_id}/{filename}")
        return str(weights / "dino-config.json")

    # Background removal is unnecessary because the input already supplies a mask.
    sys.modules.setdefault("rembg", ModuleType("rembg"))
    isosurface = ModuleType("torchmcubes")
    isosurface.marching_cubes = extract_surface
    sys.modules["torchmcubes"] = isosurface
    sys.path.insert(0, str(source))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    from tsr.models.tokenizers import image as tokenizer
    from tsr.system import TSR

    tokenizer.hf_hub_download = local_dino_config
    return TSR


def export_object(mesh, photo, output, report):
    import numpy as np
    import trimesh
    from trimesh.exchange.gltf import export_glb

    if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all() or mesh.visual.kind != "vertex":
        raise ValueError("TripoSR did not generate a finite coloured mesh")
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()
    # TripoSR works in Z-up coordinates and glTF stores the model in Y-up.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))

    # The input image and the model's predicted colours are sRGB; glTF's COLOR_0 is linear.
    colors = mesh.visual.vertex_colors.copy()
    srgb = colors[:, :3].astype(np.float32) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    colors[:, :3] = np.rint(linear * 255).astype(np.uint8)
    mesh.visual.vertex_colors = colors

    def apply_diffuse_material(tree):
        materials = tree.setdefault("materials", [])
        material_index = len(materials)
        materials.append({
            "name": "Couleurs reconstruites",
            "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.65},
        })
        for geometry in tree["meshes"]:
            for primitive in geometry["primitives"]:
                primitive["material"] = material_index

    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".objet-", dir=output) as directory:
        temporary = Path(directory)
        destination = temporary / "objet.glb"
        destination.write_bytes(export_glb(mesh.scene(), include_normals=True,
                                           tree_postprocessor=apply_diffuse_material))
        verified = trimesh.load_mesh(destination, process=False, skip_materials=True)
        if (len(verified.faces) != len(mesh.faces) or verified.visual.kind != "vertex"
                or not np.array_equal(verified.visual.vertex_colors, colors)
                or not np.isfinite(verified.vertices).all()):
            raise ValueError("GLB round-trip did not preserve the coloured mesh")
        photo.save(temporary / "entree.png")
        with destination.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report.update({
            "vertices": len(verified.vertices),
            "triangles": len(verified.faces),
            "vertex_colors": True,
            "vertex_color_space": "linear sRGB",
            "smooth_normals": True,
            "dimensions_y_up_unscaled": [round(float(value), 6) for value in verified.extents],
            "watertight": bool(verified.is_watertight),
            "filesize_bytes": destination.stat().st_size,
            "sha256": digest,
            "output": destination.name,
        })
        (temporary / "reconstruction.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for name in ("objet.glb", "entree.png", "reconstruction.json"):
            (temporary / name).replace(output / name)


def main():
    args = arguments()
    start = perf_counter()
    photo = prepare_photo(args.photo.resolve(strict=True))
    prepared = perf_counter()
    source, weights = runtime_paths(args.runtime.resolve())
    import torch

    threads = min(8, os.cpu_count() or 1)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    triposr = load_triposr(source, weights)
    ready = perf_counter()
    print("Open-Video OBJET chargement TripoSR", flush=True)
    model = triposr.from_pretrained(str(weights), "config.yaml", "model.ckpt").eval()
    model.renderer.set_chunk_size(32768)
    loaded = perf_counter()
    resolution = PROFILES[args.profile]
    with torch.inference_mode():
        codes = model([photo], device="cpu")
        inferred = perf_counter()
        mesh = model.extract_mesh(codes, has_vertex_color=True, resolution=resolution)[0]
    extracted = perf_counter()
    report = {
        "project": "Open-Video",
        "engine": "TripoSR",
        "source_revision": SOURCE_REVISION,
        "device": "CPU",
        "torch": str(torch.__version__),
        "threads": threads,
        "source_photo": args.photo.name,
        "profile": args.profile,
        "grid_resolution": resolution,
        "preparation_seconds": round(prepared - start, 3),
        "load_seconds": round(loaded - ready, 3),
        "runtime_init_seconds": round(ready - prepared, 3),
        "inference_seconds": round(inferred - loaded, 3),
        "extraction_seconds": round(extracted - inferred, 3),
        "processing_seconds": round(extracted - start, 3),
    }
    export_object(mesh, photo, args.output.resolve(), report)
    print("Open-Video OBJET " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
