"""Validate a rendered still or encode a frame sequence with optional audio."""

import argparse
import json
import statistics
import struct
import subprocess
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter


HERE = Path(__file__).resolve().parent
POV = HERE.parent / "essais/pov"


def inspect_cadence(video, frames, recipe):
    fps = recipe["fps"]
    time_base = Fraction(video["time_base"])
    ticks_per_frame = 1 / (fps * time_base)
    timestamps = [frame.get("pts") for frame in frames if frame["stream_index"] == video["index"]]
    if len(timestamps) != recipe["frames"] or any(pts is None for pts in timestamps):
        raise RuntimeError("Missing video presentation timestamps")

    # Constant cadence can require rounding when the MP4 timescale is not a multiple of the frame rate.
    first_pts = timestamps[0]
    steps = [current - previous for previous, current in zip(timestamps, timestamps[1:])]
    for index, step in enumerate(steps, start=1):
        elapsed = timestamps[index] - first_pts
        if step <= 0 or abs(step - ticks_per_frame) >= 1 or abs(elapsed - index * ticks_per_frame) >= 1:
            raise RuntimeError(f"Irregular video cadence at frame {index + 1}: PTS {timestamps[index]}, "
                               f"expected {first_pts + index * ticks_per_frame} ticks")

    duration = video.get("duration_ts")
    if duration is None or abs(duration - recipe["frames"] * ticks_per_frame) >= 1:
        raise RuntimeError(f"Unexpected video stream duration: {duration} ticks at {time_base}")
    return {
        "fps": fps,
        "time_base": str(time_base),
        "frames_checked": len(timestamps),
        "first_pts": first_pts,
        "last_pts": timestamps[-1],
        "minimum_step_ticks": min(steps, default=None),
        "maximum_step_ticks": max(steps, default=None),
        "duration_ticks": duration,
    }


def inspect_video(ffprobe, movie, recipe, has_audio):
    raw = subprocess.check_output([
        str(ffprobe), "-v", "error", "-count_frames", "-of", "json", "-show_entries",
        "stream=index,codec_name,codec_type,width,height,avg_frame_rate,nb_read_frames,channels,sample_rate,time_base,duration_ts"
        ":frame=stream_index,pts:format=duration,size",
        str(movie),
    ])
    report = json.loads(raw)
    streams = {stream["codec_type"]: stream for stream in report["streams"]}
    video, audio = streams["video"], streams.get("audio")
    if (video["codec_name"] != "h264" or [video["width"], video["height"]] != recipe["resolution"]
            or int(video["nb_read_frames"]) != recipe["frames"]):
        raise RuntimeError(f"Unexpected video stream: {video}")
    if bool(audio) != has_audio or audio and (
        audio["codec_name"] != "aac" or audio["channels"] != 2 or int(audio["sample_rate"]) != 48000
    ):
        raise RuntimeError(f"Unexpected audio stream: {audio}")
    report["cadence"] = inspect_cadence(video, report.pop("frames"), recipe)
    return report


def audio_settings(recipe, recipe_path, override):
    audio = recipe.get("audio")
    offset = recipe.get("audio_offset_seconds", 0)
    if override:
        return override.resolve(strict=True), offset
    if audio:
        return (recipe_path.parent / audio["path"]).resolve(strict=True), audio.get("offset_seconds", 0)
    if "audio_offset_seconds" in recipe:
        raise ValueError("Set audio.path or pass --audio when specifying an audio offset")
    return None, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=POV / "recette.json")
    parser.add_argument("--output", type=Path, default=POV.parent.parent / "cache/rendus")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--gpu-report", type=Path)
    args = parser.parse_args()

    ffmpeg = args.ffmpeg.resolve(strict=True)
    ffprobe = ffmpeg.with_name("ffprobe" + ffmpeg.suffix).resolve(strict=True)
    frames = args.frames.resolve(strict=True)
    output = args.output.resolve()
    recipe_path = args.recipe.resolve(strict=True)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    preparation_path = output / "preparation.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8")) if preparation_path.is_file() else None
    render_report = json.loads((frames / "frames.json").read_text(encoding="utf-8"))
    frame_range = list(range(1, recipe["frames"] + 1))
    if [frame["frame"] for frame in render_report["frames"]] != frame_range or any(
        not (frames / f"{frame:04d}.png").is_file() for frame in frame_range
    ):
        raise RuntimeError("The rendered sequence is incomplete")

    stem = recipe.get("name", "Open-Video")
    if stem in ("", ".", "..") or Path(stem).name != stem:
        raise ValueError("The recipe name must be a filename stem")
    frame_times = [frame["seconds"] for frame in render_report["frames"]]
    report = {
        "preparation": preparation,
        "render": {
            "profile": render_report.get("profile", "reference"), "lighting": render_report.get("lighting", "reference"),
            "gpu": render_report["gpu"], "samples_max": render_report.get("samples", recipe.get("samples")),
            "frame_seconds_sum": round(sum(frame_times), 3),
            "median_frame_seconds": statistics.median(frame_times), "max_frame_seconds": max(frame_times),
            "frame_times_seconds": frame_times,
        },
    }
    if args.gpu_report:
        report["gpu_telemetry"] = json.loads(args.gpu_report.read_text(encoding="utf-8-sig"))

    output.mkdir(parents=True, exist_ok=True)
    if recipe["frames"] == 1:
        image = frames / "0001.png"
        with image.open("rb") as data:
            header = data.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or list(struct.unpack(">II", header[16:24])) != recipe["resolution"]:
            raise RuntimeError("Unexpected still image dimensions")
        subprocess.run([str(ffmpeg), "-v", "error", "-xerror", "-i", str(image), "-f", "null", "-"], check=True)
        (output / (stem + ".png")).write_bytes(image.read_bytes())
        report["still"] = {"resolution": recipe["resolution"], "file": stem + ".png"}
        (output / "mesures.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"1 frame checked, render {frame_times[0]:.3f}s")
        return

    audio, offset = audio_settings(recipe, recipe_path, args.audio)
    duration = recipe["frames"] / recipe["fps"]
    inputs = ["-framerate", str(recipe["fps"]), "-start_number", "1", "-i", str(frames / "%04d.png")]
    audio_output = []
    if audio:
        inputs.extend(["-ss", str(offset), "-i", str(audio)])
        fade_in, fade_out = min(0.08, duration / 4), min(0.16, duration / 4)
        audio_filter = f"afade=t=in:d={fade_in},afade=t=out:st={duration - fade_out}:d={fade_out}"
        audio_output = ["-map", "1:a:0", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", "-af", audio_filter]
        report["audio"] = {"source": audio.name, "offset_seconds": offset}

    with TemporaryDirectory(prefix="open-video-encodage-", dir=output) as temp:
        movie = Path(temp) / (stem + ".mp4")
        start = perf_counter()
        subprocess.run([
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            *inputs, "-map", "0:v:0", *audio_output, "-t", f"{duration:.6f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-vf", "scale=out_color_matrix=bt709,format=yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-map_metadata", "-1",
            "-metadata", f"title={recipe.get('title', 'Open-Video')}", "-movflags", "+faststart", str(movie),
        ], check=True)
        report["encoding_seconds"] = round(perf_counter() - start, 3)
        video_report = inspect_video(ffprobe, movie, recipe, bool(audio))
        if abs(float(video_report["format"]["duration"]) - duration) > 1 / recipe["fps"]:
            raise RuntimeError("Unexpected clip duration")
        subprocess.run([str(ffmpeg), "-v", "error", "-xerror", "-i", str(movie), "-f", "null", "-"], check=True)
        report["ffprobe"] = video_report
        final_report = Path(temp) / "mesures.json"
        final_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        movie.replace(output / movie.name)
        final_report.replace(output / final_report.name)

    print(f"{recipe['frames']} frames checked, median {report['render']['median_frame_seconds']:.3f}s, encoding {report['encoding_seconds']:.3f}s")


if __name__ == "__main__":
    main()
