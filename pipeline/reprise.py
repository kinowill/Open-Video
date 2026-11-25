"""Atomic checkpoints for interrupted frame sequences."""

import hashlib
import json
import math
import struct
from pathlib import Path
from tempfile import NamedTemporaryFile


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def render_fingerprint(scene, dependencies, settings, scripts):
    files = []
    for filename in sorted({str(scene), *dependencies, *(str(script) for script in scripts)}):
        path = Path(filename)
        try:
            files.append((filename, file_hash(path)))
        except OSError:
            return None, filename
    payload = json.dumps({"files": files, "settings": settings}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), None


def atomic_json(path, contents):
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False, mode="w", encoding="utf-8") as stream:
            temporary = Path(stream.name)
            json.dump(contents, stream, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_checkpoint(path, signature):
    if signature is None:
        return {}
    try:
        contents = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    if isinstance(contents, dict) and contents.get("signature") == signature:
        frames = contents.get("frames")
        if isinstance(frames, dict):
            return frames
    return {}


def frame_hash(path, resolution):
    try:
        with path.open("rb") as stream:
            header = stream.read(24)
            if (len(header) != 24 or header[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                    or struct.unpack(">II", header[16:]) != tuple(resolution)):
                return None
            stream.seek(-12, 2)
            if stream.read(12) != b"\x00\x00\x00\x00IEND\xaeB`\x82":
                return None
            stream.seek(0)
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return None


def verified_frame(entry, path, source_frame, resolution, *, source_subframe=0.0):
    if (not isinstance(entry, dict) or entry.get("source_frame") != source_frame
            or entry.get("source_subframe", 0.0) != source_subframe):
        return False
    seconds = entry.get("seconds")
    digest = entry.get("sha256")
    return (type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0
            and isinstance(digest, str) and frame_hash(path, resolution) == digest)
