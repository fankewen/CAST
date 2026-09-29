"""Constants and field adapters for the cache on-disk format."""

from __future__ import annotations

from typing import Any, Mapping


CACHE_TAR_INVENTORY_FILENAME = "cache_rgb_mask_tar.json"


def skeleton_id_from_static(static: Mapping[str, Any]) -> str:
    """Return the skeleton identity stored in static metadata."""
    value = static["skeleton_id"]
    array = value if hasattr(value, "item") else None
    return str(array.item() if array is not None and array.ndim == 0 else value)
