"""Create an immutable rig session for the Blender remote addon."""
import argparse
import json
from pathlib import Path

from client_config import default_config_path, load_config, write_json
from pipeline_client import PipelinedMotionClient


def create_session(config_path: Path, rig_path: Path, output: Path) -> dict:
    config = load_config(config_path)
    rig = json.loads(rig_path.read_text(encoding="utf-8"))
    client = PipelinedMotionClient(config["server_url"], config["api_key"],
                                   window=int(config.get("pipeline_depth", 4)),
                                   timeout=float(config.get("request_timeout_seconds", 120)))
    try:
        meta = client.start(rig, config.get("target_mode", "human"))
        if meta.get("joint_names") != rig.get("joint_names"):
            raise RuntimeError("Server joint order differs from the exported Blender rig")
        meta["server_url"] = config["server_url"]
        meta["target_mode"] = config.get("target_mode", "human")
        write_json(output, meta)
    except BaseException:
        client.close()
        raise
    client.close(delete_session=False)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--rig-file", type=Path, required=True)
    parser.add_argument("--active-session", type=Path, required=True)
    args = parser.parse_args()
    meta = create_session(args.config, args.rig_file, args.active_session)
    print(f"Session ready: {meta['session_id']}, joints={meta['joints']}")
