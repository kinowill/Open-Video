"""Calibrate photographs with COLMAP on the CPU and train a Brush splat scene."""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter


PROFILES = {"apercu": 400, "standard": 1200, "detail": 3600}
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png"}
COLMAP_SETTINGS = {"feature_max_px": 1600, "undistort_max_px": 1440, "threads": min(8, os.cpu_count() or 1)}
BRUSH_SETTINGS = {"max_resolution_px": 960, "max_splats": 100000, "seed": 42}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def photos_in(directory):
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in PHOTO_EXTENSIONS)


@contextmanager
def output_lock(path):
    with path.open("a+b") as file:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another reconstruction is using this output directory") from error
        yield


def run(tool, arguments, directory, log_path, capture=False):
    command = [str(tool), *map(str, arguments)]
    print(f"Open-Video DECOR {tool.stem} {arguments[0] if tool.stem == 'colmap' else 'training'}", flush=True)
    start = perf_counter()
    with log_path.open("a+b") as log:
        log.write(("> " + subprocess.list2cmdline(command) + "\n").encode("utf-8"))
        log.flush()
        offset = log.tell()
        result = subprocess.run(command, cwd=directory, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            raise RuntimeError(f"{tool.name} failed ({result.returncode}). See {log_path}")
        output = ""
        if capture:
            log.seek(offset)
            output = log.read().decode("utf-8", errors="replace")
        log.write(b"\n")
    return round(perf_counter() - start, 3), output


def calibration_metrics(output):
    def value(field, pattern, convert):
        match = re.search(rf"\] {field}: ({pattern})", output)
        if match is None:
            raise RuntimeError(f"Could not read {field} from COLMAP")
        return convert(match[1])

    return {
        "registered_images": value("Registered images", r"\d+", int),
        "points": value("Points", r"\d+", int),
        "mean_reprojection_error_px": value("Mean reprojection error", r"[\d.]+(?:[eE][+-]?\d+)?", float),
    }


def calibrate(colmap, photo_dir, photos, settings, stage, log_path):
    photo_list = stage / "photos.txt"
    photo_list.write_text("".join(f"{photo.name}\n" for photo in photos), encoding="utf-8")
    database = stage / "photos.db"
    models = stage / "models"
    models.mkdir()
    package = stage / "calibration"
    dataset = package / "dataset"
    dataset.mkdir(parents=True)

    times = {}

    def colmap_step(step, *arguments, capture=False):
        seconds, output = run(colmap, [step, "--log_target", "stderr", "--log_color", 0, *arguments],
                              stage, log_path, capture)
        times[step] = round(times.get(step, 0) + seconds, 3)
        return output

    threads = settings["threads"]
    feature_args = ["--database_path", database, "--image_path", photo_dir, "--image_list_path", photo_list,
                    "--FeatureExtraction.use_gpu", 0, "--FeatureExtraction.num_threads", threads,
                    "--FeatureExtraction.max_image_size", settings["feature_max_px"]]
    if settings["single_camera"]:
        feature_args.extend(["--ImageReader.single_camera", 1])
    colmap_step("feature_extractor", *feature_args)
    matcher = {"exhaustif": "exhaustive_matcher", "sequentiel": "sequential_matcher"}[settings["matching"]]
    colmap_step(matcher, "--database_path", database, "--FeatureMatching.use_gpu", 0, "--FeatureMatching.num_threads", threads)
    colmap_step("mapper", "--database_path", database, "--image_path", photo_dir, "--output_path", models,
                "--Mapper.num_threads", threads)

    candidates = []
    for model in sorted(models.iterdir()):
        if (model / "images.bin").is_file():
            info = calibration_metrics(colmap_step("model_analyzer", "--path", model, capture=True))
            candidates.append((model, info))
    if not candidates:
        raise RuntimeError(f"COLMAP did not reconstruct any cameras. See {log_path}")
    best, stats = max(candidates, key=lambda item: (item[1]["registered_images"], item[1]["points"],
                                                    -item[1]["mean_reprojection_error_px"]))
    if stats["registered_images"] < 3 or not stats["points"]:
        raise RuntimeError(f"COLMAP reconstructed too few cameras or no 3D points. See {log_path}")
    colmap_step("image_undistorter", "--image_path", photo_dir, "--input_path", best, "--output_path", dataset,
                "--output_type", "COLMAP", "--max_image_size", settings["undistort_max_px"])
    if not valid_calibration(dataset, stats["registered_images"]):
        raise RuntimeError(f"COLMAP did not export the reconstructed images and cameras. See {log_path}")

    return package, {
        **stats,
        "models": len(candidates),
        "coverage": round(stats["registered_images"] / len(photos), 3),
        "calibration_phase_seconds": times,
    }


def valid_calibration(dataset, count):
    try:
        sparse = dataset / "sparse"
        model_files = ("cameras.bin", "images.bin", "points3D.bin")
        photos = [path for path in (dataset / "images").rglob("*")
                  if path.suffix.lower() in PHOTO_EXTENSIONS and path.is_file() and path.stat().st_size]
        return len(photos) == count and all((sparse / name).stat().st_size for name in model_files)
    except OSError:
        return False


def cached_calibration(package, cache_key):
    try:
        data = json.loads((package / "calibrage.json").read_text(encoding="utf-8"))
        if data["key"] == cache_key and valid_calibration(package / "dataset", data["registered_images"]):
            return data
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def inspect_splats(path):
    with path.open("rb") as file:
        lines = []
        line = ""
        for raw in iter(lambda: file.readline(65537), b""):
            if file.tell() > 65536:
                break
            line = raw.strip().decode("ascii", errors="replace")
            if line == "end_header":
                break
            lines.append(line)
        else:
            line = ""
        if line != "end_header" or lines[:2] != ["ply", "format binary_little_endian 1.0"]:
            raise RuntimeError("Brush did not export a binary PLY")
        elements = [line for line in lines if line.startswith("element ")]
        properties = {line.removeprefix("property float ") for line in lines if line.startswith("property float ")}
        required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
                    "f_dc_0", "f_dc_1", "f_dc_2"}
        if len(elements) != 1 or not elements[0].startswith("element vertex ") or not required <= properties:
            raise RuntimeError("Brush PLY is missing its Gaussian splat properties")
        vertices = int(elements[0].split()[-1])
        size = path.stat().st_size
        if vertices < 1 or file.tell() + vertices * len(properties) * 4 != size:
            raise RuntimeError("Brush PLY is empty or truncated")
    return {"file": path.name, "splats": vertices, "bytes": size, "sha256": file_hash(path)}


def reconstruct(args, output):
    photo_dir = args.photos.resolve(strict=True)
    colmap = args.colmap.resolve(strict=True)
    brush = args.brush.resolve(strict=True)
    photos = photos_in(photo_dir)
    if len(photos) < 3:
        raise ValueError("Supply at least three overlapping JPG, JPEG or PNG photographs in the input directory")
    log = output / "reconstruction.log"
    log.write_text("", encoding="utf-8")
    total_start = perf_counter()
    settings = {**COLMAP_SETTINGS, "single_camera": args.single_camera, "matching": args.matching}
    photo_digests = {path.name: file_hash(path) for path in photos}
    cache_input = {"photos": photo_digests, "settings": settings, "colmap_sha256": file_hash(colmap)}
    cache_key = hashlib.sha256(json.dumps(cache_input, sort_keys=True).encode("utf-8")).hexdigest()
    calibration_cache = output / ".calibrage" / cache_key
    brush_wd = output / ".brush"
    brush_wd.mkdir(exist_ok=True)

    with TemporaryDirectory(prefix=".reconstruction-", dir=output) as temp:
        stage = Path(temp)
        calibration_start = perf_counter()
        package = calibration_cache
        info = cached_calibration(package, cache_key)
        reused = info is not None
        if not reused:
            package, info = calibrate(colmap, photo_dir, photos, settings, stage, log)
            if {path.name: file_hash(path) for path in photos_in(photo_dir)} != photo_digests:
                raise RuntimeError("Source photos changed during calibration. Run the reconstruction again.")
            info["key"] = cache_key
            write_json(package / "calibrage.json", info)
            if not calibration_cache.exists():
                if not calibration_cache.resolve().is_relative_to(output):
                    raise RuntimeError("Calibration cache is outside the output directory")
                calibration_cache.parent.mkdir(exist_ok=True)
                package.rename(calibration_cache)
                package = calibration_cache
        calibration_seconds = round(perf_counter() - calibration_start, 3)
        print(f"Open-Video DECOR {info['registered_images']}/{len(photos)} camera frames, calibration reused={reused}", flush=True)

        steps = PROFILES[args.profile]
        settings_brush = {**BRUSH_SETTINGS, "steps": steps,
                          "subsample_points": math.ceil(info["points"] / BRUSH_SETTINGS["max_splats"])}
        brush_seconds, _ = run(brush, [package / "dataset", "--total-steps", steps,
                                      "--max-resolution", settings_brush["max_resolution_px"],
                                      "--max-splats", settings_brush["max_splats"], "--seed", settings_brush["seed"],
                                      "--subsample-points", settings_brush["subsample_points"],
                                      "--export-every", steps, "--export-path", stage, "--export-name", "scene.ply"],
                               brush_wd, log)
        ply = stage / "scene.ply"
        if not ply.is_file():
            raise RuntimeError(f"Brush did not export scene.ply. See {log}")
        result = inspect_splats(ply)
        report = {
            "project": "Open-Video",
            "profile": args.profile,
            "source_photos": len(photos),
            "colmap": {"settings": settings, "reused": reused, "seconds": calibration_seconds, **info},
            "brush": {"settings": settings_brush, "seconds": brush_seconds},
            "scene": result,
            "seconds": round(perf_counter() - total_start, 3),
        }
        write_json(stage / "rapport-decor.json", report)
        ply.replace(output / "scene.ply")
        (stage / "rapport-decor.json").replace(output / "rapport-decor.json")
        print(f"Open-Video DECOR {result['splats']} splats, {report['seconds']} s, {(output / 'scene.ply')}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--colmap", type=Path, required=True)
    parser.add_argument("--brush", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="standard")
    parser.add_argument("--single-camera", action="store_true")
    parser.add_argument("--matching", choices=("exhaustif", "sequentiel"), default="exhaustif")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=True)
        with output_lock(output / ".reconstruction.lock"):
            reconstruct(args, output)
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f"Open-Video DECOR: {error}")


if __name__ == "__main__":
    main()
