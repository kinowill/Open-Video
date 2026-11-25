"""Build an original material with MaterialCraft and render two Blender scenes."""

import argparse
import json
import random
import struct
import subprocess
import zlib
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter


TEXTURE_SIZE = 512
MAP_NAMES = ("basecolor", "roughness", "normal")


def png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def write_height_map(path):
    randomizer = random.Random(14)
    octaves = []
    for frequency, weight in ((5, 0.36), (13, 0.29), (33, 0.23), (99, 0.12)):
        grid = [[randomizer.random() for _ in range(frequency)] for _ in range(frequency)]
        octaves.append((frequency, weight, grid))

    rows = bytearray()
    for y in range(TEXTURE_SIZE):
        rows.append(0)
        for x in range(TEXTURE_SIZE):
            height = 0.0
            for frequency, weight, grid in octaves:
                u = x * frequency / TEXTURE_SIZE
                v = y * frequency / TEXTURE_SIZE
                i, j = int(u), int(v)
                du, dv = u - i, v - j
                u = du * du * (3 - 2 * du)
                v = dv * dv * (3 - 2 * dv)
                next_i, next_j = (i + 1) % frequency, (j + 1) % frequency
                a = grid[j][i] * (1 - u) + grid[j][next_i] * u
                b = grid[next_j][i] * (1 - u) + grid[next_j][next_i] * u
                height += weight * (a * (1 - v) + b * v)
            rows.extend(struct.pack(">H", round(height * 65535)))

    header = struct.pack(">IIBBBBB", TEXTURE_SIZE, TEXTURE_SIZE, 16, 0, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(rows))
        + png_chunk(b"IEND", b"")
    )


def build_graph(materialcraft, scratch, output):
    seed = scratch / "open-video-height.png"
    blank_graph = scratch / "blank.sbs"
    working_graph = scratch / "stone.sbs"
    batch_path = scratch / "batch.json"
    write_height_map(seed)
    subprocess.run(
        [str(materialcraft), "new", "empty", str(blank_graph), "--size", str(TEXTURE_SIZE), "--name", "OpenVideo_Stone"],
        check=True,
        capture_output=True,
    )

    steps = [{"id": "file.importBitmap", "params": {"path": str(seed), "name": "open-video_height"}}]
    nodes = [
        ("bitmap", None, {"resource": "open-video_height", "colour_space": 1, "color_mode": False}),
        ("gradient_map", 1, {"gradient": [[0.0, "#4d443b"], [0.45, "#82776a"], [0.67, "#a69c90"], [1.0, "#cebda7"]]}),
        ("gradient_map", 1, {"gradient": [[0.0, "#929292"], [1.0, "#e9e9e9"]], "color_mode": False}),
        ("normal", 1, {"intensity": 2.3, "opengl": True}),
        ("output", 2, {"identifier": "basecolor", "usage": 0}),
        ("output", 3, {"identifier": "roughness", "usage": 2}),
        ("output", 4, {"identifier": "normal", "usage": 1}),
    ]
    for index, (kind, source, parameters) in enumerate(nodes):
        arguments = {"def": kind, "params": parameters, "at": [(index // 3) * 230, (index % 3) * 175]}
        if source is not None:
            arguments["connectFrom"] = {"node": source}
        steps.append({"id": "graph.node.add", "params": arguments})

    batch = {"open": str(blank_graph), "steps": steps, "save": str(working_graph)}
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    built = subprocess.run([str(materialcraft), "batch", str(batch_path)], capture_output=True, text=True, encoding="utf-8")
    if built.returncode or not working_graph.is_file():
        raise RuntimeError(f"MaterialCraft did not save its graph: {built.stderr or built.stdout}")

    start = perf_counter()
    subprocess.run(
        [str(materialcraft), "export", str(working_graph), str(output), "--format", "png", "--depth", "16", "--pattern", "$(identifier)"],
        check=True,
    )
    export_seconds = round(perf_counter() - start, 3)
    for name in MAP_NAMES:
        path = output / f"{name}.png"
        data = path.read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", data[16:24]) != (TEXTURE_SIZE, TEXTURE_SIZE) or data[24] != 16:
            raise RuntimeError(f"Expected a 16-bit {TEXTURE_SIZE}×{TEXTURE_SIZE} texture in {path}")
    return export_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialcraft", type=Path, required=True, help="Path to materialcraft-cli.exe")
    parser.add_argument("--blender", type=Path, required=True, help="Path to blender.exe")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "cache/rendus/materiau")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    materialcraft = args.materialcraft.resolve(strict=True)
    version = subprocess.check_output([str(materialcraft), "--version"], text=True, encoding="utf-8").strip()

    with TemporaryDirectory(prefix="open-video-materiaux-", dir=output) as temporary_directory:
        scratch = Path(temporary_directory)
        start = perf_counter()
        exported_seconds = build_graph(materialcraft, scratch, output)
        preparation_seconds = round(perf_counter() - start - exported_seconds, 3)
        subprocess.run(
            [str(args.blender.resolve(strict=True)), "--background", "--factory-startup", "--disable-autoexec",
             "--python-exit-code", "1", "--python", str(Path(__file__).with_name("rendre_apercus.py")), "--", str(output)],
            check=True,
            env={**environ, "BLENDER_USER_CONFIG": str(scratch / "blender-config")},
        )

    report = output / "mesures.json"
    measurements = json.loads(report.read_text(encoding="utf-8"))
    measurements["materialcraft"] = {
        "version": version,
        "texture_size": TEXTURE_SIZE,
        "png_bit_depth": 16,
        "maps": list(MAP_NAMES),
        "preparation_seconds": preparation_seconds,
        "export_seconds": exported_seconds,
    }
    report.write_text(json.dumps(measurements, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
