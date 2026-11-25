"""Compare the rendered audit frames without mixing the light and speed tests."""

import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path


def similarity(ffmpeg, first, second, foreground_only=False):
    filters = "ssim"
    if foreground_only:
        filters = "[0:v]crop=iw:ih/2:0:ih/2[a];[1:v]crop=iw:ih/2:0:ih/2[b];[a][b]ssim"
    result = subprocess.run([
        str(ffmpeg), "-hide_banner", "-i", str(first), "-i", str(second), "-lavfi", filters,
        "-frames:v", "1", "-f", "null", "-",
    ], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    match = re.search(r"\bAll:([0-9.]+)", result.stderr)
    if not match:
        raise RuntimeError("SSIM measurement missing")
    return float(match[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve(strict=True)
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    reference = {row["frame"]: row for row in audit["renders"] if row["profile"] == "reference"}
    baseline = statistics.median(row["seconds"] for row in reference.values())
    profiles = {}
    for name, settings in audit["profiles"].items():
        frames = [row for row in audit["renders"] if row["profile"] == name]
        times = [row["seconds"] for row in frames]
        comparisons = []
        if name != "reference":
            for row in frames:
                comparisons.append({"frame": row["frame"], "ssim": similarity(
                    args.ffmpeg, root / reference[row["frame"]]["file"], root / row["file"]
                )})
        median = statistics.median(times)
        profiles[name] = {"settings": settings, "median_seconds": median,
                          "gain_percent": round(100 * (1 - median / baseline), 1),
                          "frames": [{"frame": row["frame"], "seconds": row["seconds"]} for row in frames],
                          "ssim": comparisons}

    lighting = audit["lighting"]
    if lighting:
        first, second = root / lighting["original"], root / lighting["adjusted"]
        lighting["foreground_ssim"] = similarity(args.ffmpeg, first, second, foreground_only=True)
        lighting["comparison"] = "comparaison-lumiere.png"
        subprocess.run([
            str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(first), "-i", str(second),
            "-filter_complex", "[0:v]scale=360:-2[a];[1:v]scale=360:-2[b];[a][b]hstack",
            "-frames:v", "1", "-update", "1", str(root / lighting["comparison"]),
        ], check=True)
        lighting = {key: lighting[key] for key in ("camera_strength_factor", "foreground_ssim", "comparison")}

    report = {"gpu": audit["gpu"], "resolution": audit["resolution"], "profiles": profiles,
              "lighting": lighting, "occlusion": audit["occlusion"]}
    (root / "comparaison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, profile in profiles.items():
        print(f"{name}: {profile['median_seconds']:.3f}s median, {profile['gain_percent']:+.1f}% time saved")


if __name__ == "__main__":
    main()
