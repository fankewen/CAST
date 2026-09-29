"""Small, explicit temporal KV-cache primitives.

The update/trim contract follows Hugging Face Transformers' cache API: every
attention layer owns one cache, ``update`` appends the projected key/value for
the current position, and a sliding window keeps only the visible suffix.
Unlike a forward-hook or a monkey-patch, the cache is an explicit argument to
the streaming forward path and is therefore safe to reuse across frames.

Shape stability
---------------
A capped cache (``max_length`` set) always returns exactly ``max_length`` slots
in chronological order.  The write position, the frame counter and the absolute
position are device tensors, and a slot that no frame has been written to yet is
reported through ``relative_delta`` / ``valid_mask`` instead of by shortening
the returned slice.  Nothing that a traced graph can see therefore changes while
the window fills: one compiled streaming step serves frame 1 through frame N.
Without a cap the cache grows and returns the whole history, which is what
training and offline inference expect.
"""

from __future__ import annotations

import torch

# Distance given to a slot that is still empty. Larger than any temporal window,
# so ``relative_delta <= temporal_window`` already excludes it.
_UNWRITTEN_DELTA = 4096


class TemporalKVCache:
    """Per-layer cache for tensors shaped ``[batch_joints, heads, time, dim]``."""

    def __init__(self, max_length: int | None = None):
        if max_length is not None and (isinstance(max_length, bool) or max_length <= 0):
            raise ValueError("max_length must be a positive integer or None")
        self.max_length = max_length
        self.key: torch.Tensor | None = None
        self.value: torch.Tensor | None = None
        self.start_pos: int | torch.Tensor = 0
        self._length = 0
        self._write_pos: torch.Tensor | None = None
        self._ring_indices: torch.Tensor | None = None
        self._relative_delta: torch.Tensor | None = None
        # Frames written since the last reset and how many of them the window
        # holds; both stay on the device (see ``_allocate``).
        self._total: torch.Tensor | None = None
        self._filled: torch.Tensor | None = None

    @property
    def length(self) -> int | torch.Tensor:
        """Slots currently holding a frame (a device tensor once streaming)."""
        return self._filled if self._filled is not None else self._length

    @property
    def next_position(self) -> int | torch.Tensor:
        """Absolute position of the frame that ``update`` will write next."""
        if self._total is not None:
            return self._total
        return self.start_pos + self.length

    def relative_delta(self, device=None) -> torch.Tensor:
        """Chronological causal distances for the visible window, oldest first.

        A capped cache reports ``max_length`` distances even while it fills;
        the slots that are still empty get ``_UNWRITTEN_DELTA + age``, so a
        ``delta <= temporal_window`` mask drops them on its own.
        """
        if self.max_length is None:
            target = device if device is not None else (
                self.key.device if self.key is not None else "cpu"
            )
            if self._relative_delta is None or self._relative_delta.device != target:
                self._relative_delta = torch.arange(
                    self.length - 1, -1, -1, device=target, dtype=torch.long
                )
            return self._relative_delta
        # Deliberately rebuilt on every call: this runs inside the compiled step,
        # and a Python-level cache here is a Dynamo guard that re-traces the
        # whole graph as soon as it flips (it cost 30 s on the first frame).
        target = device if device is not None else self.key.device
        ages = torch.arange(
            self.max_length - 1, -1, -1, device=target, dtype=torch.long
        )
        return torch.where(ages < self._filled.to(target), ages, ages + _UNWRITTEN_DELTA)

    def valid_mask(self, device=None) -> torch.Tensor:
        """True for the slots that already hold a frame, oldest first.

        Only needed by attention layers without a temporal window; with one, the
        distances from ``relative_delta`` already exclude the empty slots.
        """
        device = device if device is not None else (
            self.key.device if self.key is not None else "cpu"
        )
        if self._filled is None:
            length = 0 if self.key is None else self.key.shape[2]
            return torch.ones(length, dtype=torch.bool, device=device)
        return self.relative_delta(device) < self.max_length

    def update(self, key: torch.Tensor, value: torch.Tensor):
        if key.ndim != 4 or value.shape != key.shape:
            raise ValueError("cache update expects key/value [N,H,T,D] with equal shapes")
        if key.shape[2] != 1:
            raise ValueError("streaming cache update accepts exactly one frame")
        if self.max_length is None:
            return self._append(key, value)
        if self.key is None:
            self._allocate(key)
        elif self.key.shape[:2] + self.key.shape[3:] != key.shape[:2] + key.shape[3:]:
            raise ValueError("cache batch/head/head_dim shape changed between frames")
        # One physical slot per frame: the ring never shifts the window, so the
        # returned slice keeps its shape from the first frame on.
        self.key.index_copy_(2, self._write_pos.view(1), key)
        self.value.index_copy_(2, self._write_pos.view(1), value)
        self._write_pos.add_(1).remainder_(self.max_length)
        self._total.add_(1)
        self._filled.copy_(self._total.clamp(max=self.max_length))
        self.start_pos.copy_((self._total - self.max_length).clamp(min=0))
        # The next write position is the oldest visible frame, so the window in
        # chronological order starts there.
        indices = self._ring_indices.index_select(
            0, self._write_pos.view(1)
        ).squeeze(0)
        return self.key.index_select(2, indices), self.value.index_select(2, indices)

    def _allocate(self, key: torch.Tensor) -> None:
        self.key = torch.zeros(
            key.shape[0], key.shape[1], self.max_length, key.shape[3],
            dtype=key.dtype, device=key.device,
        )
        self.value = torch.zeros_like(self.key)
        # Keep every changing counter on device. A Python int that changes
        # between frames becomes a Dynamo value guard and re-traces the graph.
        self.start_pos = torch.zeros((), dtype=torch.long, device=key.device)
        self._write_pos = torch.zeros((), dtype=torch.long, device=key.device)
        self._total = torch.zeros((), dtype=torch.long, device=key.device)
        self._filled = torch.zeros((), dtype=torch.long, device=key.device)
        self._ring_indices = (
            torch.arange(self.max_length, device=key.device)[:, None]
            + torch.arange(self.max_length, device=key.device)[None, :]
        ) % self.max_length

    def _append(self, key: torch.Tensor, value: torch.Tensor):
        """Uncapped cache: keep every frame and return the whole history."""
        if self.key is None:
            self.key = key
            self.value = value
        else:
            if self.key.shape[:2] + self.key.shape[3:] != key.shape[:2] + key.shape[3:]:
                raise ValueError("cache batch/head/head_dim shape changed between frames")
            self.key = torch.cat((self.key, key), dim=2)
            self.value = torch.cat((self.value, value), dim=2)
        self._length = self.key.shape[2]
        self._relative_delta = None
        return self.key, self.value

    def reset_streaming(self) -> None:
        """Empty the window in place, keeping the tensors a graph captured."""
        if self.key is None or self.max_length is None:
            self.clear()
            return
        self.key.zero_()
        self.value.zero_()
        self.start_pos.zero_()
        self._write_pos.zero_()
        self._total.zero_()
        self._filled.zero_()
        self._relative_delta = None

    def clear(self):
        self.key = None
        self.value = None
        self.start_pos = 0
        self._length = 0
        self._write_pos = None
        self._ring_indices = None
        self._relative_delta = None
        self._total = None
        self._filled = None


class TemporalKVCacheState(dict):
    """Container used by model streaming APIs; kept dict-like for checkpointing."""
