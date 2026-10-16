#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${CAST_PYTHON:-python}"
command -v "$PYTHON" >/dev/null || { echo "Python is not available; activate your Python environment" >&2; exit 1; }
exec "$PYTHON" -u run_origin.py
