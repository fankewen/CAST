"""Portable configuration shared by the remote addon and external processes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def default_config_path() -> Path:
    return Path(os.environ.get("CAST_REMOTE_CLIENT_HOME", Path.home() / ".cast_remote")) / "config.json"


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def initialize(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        data = json.loads(Path(__file__).with_name("config.example.json").read_text(encoding="utf-8"))
        write_json(path, data)
    return path


def load_config(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["api_key"] = os.environ.get("CAST_API_KEY", data.get("api_key", ""))
    parsed = urlsplit(data.get("server_url", ""))
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("server_url must be an http(s) URL")
    if data.get("target_mode", "human") not in ("human", "generic"):
        raise ValueError("target_mode must be human or generic")
    if not 1 <= float(data.get("fps", 24)) <= 120:
        raise ValueError("fps must be 1..120")
    if not 1 <= int(data.get("pipeline_depth", 4)) <= 10:
        raise ValueError("pipeline_depth must be 1..10")
    ports = [int(data.get(key, default)) for key, default in
             (("pose_port", 5599), ("control_port", 5601), ("input_port", 5602))]
    if len(set(ports)) != len(ports) or any(not 1024 <= port <= 65535 for port in ports):
        raise ValueError("pose_port, control_port and input_port must be distinct ports in 1024..65535")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a remote client config without overwriting existing settings")
    parser.add_argument("--config", type=Path, default=default_config_path())
    args = parser.parse_args()
    print(initialize(args.config))
