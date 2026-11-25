"""Save the studio shot with the same packed Open-Video material as the POV."""

import argparse
import sys
from pathlib import Path

import bpy

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.ressources import load_stone, relative_packed_paths
from pipeline.scene import blank_scene
from rendre_apercus import studio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    scene = blank_scene()
    studio(scene, load_stone(args.stone.resolve(strict=True)))
    relative_packed_paths(bpy.data.images)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=True, relative_remap=False)


if __name__ == "__main__":
    main()
