"""Batch-scoped packing metadata, shared by layers and checkpoint recomputation."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

_cache = ContextVar('attention_layout_cache', default=None)
_persistent_cache = None
_persistent_cache_readonly = False
_context_cache_active = False


def set_attention_layout_cache(cache, *, readonly=True):
    """Install an inference cache outside the compiled streaming graph."""
    global _persistent_cache, _persistent_cache_readonly
    _persistent_cache = cache
    _persistent_cache_readonly = bool(readonly)


@contextmanager
def attention_layout_scope(cache):
    global _context_cache_active
    previous_active = _context_cache_active
    _context_cache_active = True
    token = _cache.set(cache)
    try:
        yield cache
    finally:
        _cache.reset(token)
        _context_cache_active = previous_active


def with_attention_layouts(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        # A new forward owns its cache; checkpoint closures keep it until backward.
        with attention_layout_scope({}):
            return fn(*args, **kwargs)
    return wrapped


def checkpoint(function, *args, **kwargs):
    cache = _cache.get()
    if cache is None:
        return torch_checkpoint(function, *args, **kwargs)
    @wraps(function)
    def recomputable(*inputs):
        with attention_layout_scope(cache):
            return function(*inputs)
    return torch_checkpoint(recomputable, *args, **kwargs)


def _signature(tensor):
    if tensor is None:
        return None
    # Streaming topology tensors are immutable for the lifetime of a state.
    # Do not query inference-mode state here: that non-Tensor Python op is a
    # fullgraph compiler break.  Per-forward training caches are fresh, so a
    # version field is not needed for correctness either.
    version = None
    return (tensor.device, tensor.dtype, tensor.data_ptr(), tensor.shape,
            tensor.stride(), version)


def _get(key, tensors, build):
    # The persistent branch is used by streaming inference and avoids tracing
    # ContextVar.get (unsupported by fullgraph Dynamo).  Training still uses
    # the dynamic context-local cache.
    context_cache_active = _context_cache_active
    cache = _persistent_cache
    if context_cache_active:
        cache = _cache.get()
    if cache is not None and key in cache:
        return cache[key][1]
    # Metadata must not become part of the trainable autograd graph.
    with torch.no_grad():
        result = build()
    if cache is not None and (
        context_cache_active or not _persistent_cache_readonly
    ):
        # Keep source storage alive: allocator address reuse must not hit stale keys.
        cache[key] = (tensors, result)
    return result


def cached_tensor_metadata(namespace, tensors, parameters, build):
    """Reuse non-differentiable tensor metadata inside one model forward.

    Training gets a fresh cache from ``with_attention_layouts`` for every
    batch, while activation-checkpoint recomputation shares that same cache.
    Streaming may install a longer-lived cache for its immutable topology.
    """
    tensors = tuple(tensors)
    key = (
        namespace,
        tuple(_signature(tensor) for tensor in tensors),
        tuple(parameters),
    )
    return _get(key, tensors, build)


def packed_layout(q_mask, k_mask, k_length):
    key = ('packed', _signature(q_mask), _signature(k_mask), k_length)
    def build():
        q_valid = q_mask.bool()
        k_valid = (torch.ones((q_valid.shape[0], k_length), dtype=torch.bool,
                              device=q_valid.device) if k_mask is None else k_mask.bool())
        q_lengths = q_valid.sum(-1, dtype=torch.int32)
        k_lengths = k_valid.sum(-1, dtype=torch.int32)
        active = (q_lengths > 0) & (k_lengths > 0)
        q_indices = (q_valid & active[:, None]).flatten().nonzero().flatten()
        k_indices = (k_valid & active[:, None]).flatten().nonzero().flatten()
        q_lengths = q_lengths[active]
        k_lengths = k_lengths[active]
        return dict(q_indices=q_indices, k_indices=k_indices,
                    cu_q=F.pad(q_lengths.cumsum(0, dtype=torch.int32), (1, 0)),
                    cu_k=F.pad(k_lengths.cumsum(0, dtype=torch.int32), (1, 0)),
                    max_q=int(q_lengths.max().item()) if q_lengths.numel() else 0,
                    max_k=int(k_lengths.max().item()) if k_lengths.numel() else 0)
    return _get(key, (q_mask, k_mask), build)


def temporal_layout(mask, batch, frames, joints, device):
    key = ('temporal', _signature(mask), batch, frames, joints, device)
    def build():
        if mask is None:
            valid = torch.ones((batch, frames, joints), dtype=torch.bool, device=device)
        elif mask.ndim == 2:
            valid = mask.bool()[:, None].expand(-1, frames, -1)
        else:
            valid = mask.bool()
        valid = valid.permute(0, 2, 1).reshape(batch * joints, frames)
        indices = valid.flatten().nonzero().flatten()
        lengths = valid.sum(-1, dtype=torch.int32)
        lengths = lengths[lengths > 0]
        return dict(indices=indices, positions=indices.remainder(frames), lengths=lengths,
                    cu=F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0)),
                    max_length=int(lengths.max().item()) if lengths.numel() else 0)
    return _get(key, (mask,), build)


def active_token_rows(mask):
    """Reuse valid query rows, keeping a singleton axis for image tokens."""
    if mask is None:
        return None
    if _persistent_cache_readonly and _persistent_cache is not None:
        key = ('active_rows', _signature(mask))
        cached = _persistent_cache.get(key)
        if cached is not None:
            return cached[1]
        # Fullgraph-safe fallback for an unexpected cache miss.  It keeps the
        # semantics but avoids dynamic nonzero output shapes.
        return mask.bool().any(dim=-1, keepdim=True)
    return _get(('active_rows', _signature(mask)), (mask,),
                lambda: mask.bool().any(dim=-1, keepdim=True))


def masked_pointwise(x, mask, function, *, residual=False):
    """Run token-wise normalization/MLPs only on valid rows, then scatter.

    Do not use for operations mixing tokens (attention/convolution/BatchNorm).
    Empty selections still run the module, preserving zero parameter gradients
    and DDP's used-parameter contract. Dropout probabilities are unchanged;
    packing changes which seeded random numbers are assigned to valid tokens.
    """
    if mask is None:
        update = function(x)
        return x + update if residual else update
    shape = x.shape[:-1]
    if _persistent_cache_readonly and _persistent_cache is not None:
        key = ('pointwise', _signature(mask), shape)
        cached = _persistent_cache.get(key)
        if cached is None:
            update = function(x)
            return x + update if residual else update
    def build():
        valid = mask.bool()
        if valid.shape == shape:
            pass
        elif x.ndim == 4 and valid.shape == (x.shape[0], x.shape[2]):
            valid = valid[:, None, :].expand(shape)
        elif valid.ndim == len(shape) and all(
            size == 1 or size == target for size, target in zip(valid.shape, shape)
        ):
            # Explicit [B,T,1] / [B,1] image-row masks avoid confusing a
            # frame axis with a joint axis when their lengths happen to match.
            valid = valid.expand(shape)
        elif valid.shape == shape[:valid.ndim]:
            trailing_axes = (1,) * (len(shape) - valid.ndim)
            valid = valid.reshape(*valid.shape, *trailing_axes).expand(shape)
        else:
            raise ValueError(f"mask {tuple(valid.shape)} does not match tokens {tuple(shape)}")
        return valid.reshape(-1).nonzero().flatten()
    indices = _get(('pointwise', _signature(mask), shape), (mask,), build)
    flat = x.reshape(-1, x.shape[-1])
    if indices.numel() == flat.shape[0]:
        update = function(x)
        return x + update if residual else update
    packed = flat.index_select(0, indices)
    update = function(packed)
    if residual:
        update = packed + update
    output = update.new_zeros(flat.shape[0], update.shape[-1])
    output = output.index_copy(0, indices, update)
    return output.reshape(*shape, update.shape[-1])
