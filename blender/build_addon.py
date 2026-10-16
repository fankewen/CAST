"""Build the standalone local Blender addon from its shared source modules."""

import argparse
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', zipfile.ZIP_DEFLATED) as archive:
        archive.write(source/'cast_live.py', 'cast_live/__init__.py')
        archive.write(source/'rig_adapter.py', 'cast_live/rig_adapter.py')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
