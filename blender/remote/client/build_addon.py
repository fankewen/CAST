"""Build only the independent remote addon and its external Python runtime."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
RUNTIME_FILES = (
    "client_config.py", "config.example.json", "requirements.txt", "README.md",
    "transport.py", "motion_worker.py", "pipeline_client.py", "create_session.py",
    "camera_standalone.py", "camera_capture.py", "local_frame_stream.py",
    "fps_control.py", "lens.py", "calibrate_from_video.py",
)


def build(output: Path) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(ROOT / "cast_remote_live" / "__init__.py", "cast_remote_live/__init__.py")
        for name in RUNTIME_FILES:
            archive.write(ROOT / name, "cast_remote_live/" + name)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(build(parser.parse_args().output))
