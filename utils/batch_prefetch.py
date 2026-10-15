"""Bounded, ordered CPU preparation and CUDA transfer prefetch."""

from concurrent.futures import ThreadPoolExecutor
from collections import deque

import torch


def _record_stream(value, stream):
    if isinstance(value, torch.Tensor) and value.is_cuda:
        value.record_stream(stream)
    elif isinstance(value, dict):
        for child in value.values():
            _record_stream(child, stream)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _record_stream(child, stream)


class DeviceBatchPrefetcher:
    """Keep a bounded number of future batches, including CPU preparation.

    Construct the underlying DataLoader iterator on the main thread before
    creating this object, so worker processes are not forked by this thread.
    The consumer waits on a per-batch CUDA event, not the whole copy stream.
    record_stream protects allocations until consumer kernels finish.
    """

    def __init__(self, source, move, device, depth=1):
        if type(depth) is not int or depth < 1:
            raise ValueError('prefetch depth must be a positive integer')
        self.source = iter(source)
        self.move = move
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.stream = (torch.cuda.Stream(device=self.device)
                       if self.device.type == 'cuda' else None)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='batch-prefetch')
        self.closed = False
        self.futures = deque(self.executor.submit(self._load) for _ in range(depth))

    def _load(self):
        # CUDA's current device is thread-local. Set it before CPU collation
        # too, because that code may allocate pinned memory.
        if self.stream is not None:
            torch.cuda.set_device(self.device)
        try:
            cpu_batch = next(self.source)
        except StopIteration:
            return None
        if self.stream is None:
            return self.move(cpu_batch), None, cpu_batch
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            batch = self.move(cpu_batch)
            ready = torch.cuda.Event()
            ready.record(self.stream)
        # Retain the source through enqueue/hand-off. Pinned allocations also
        # have copy events tracked by PyTorch's caching host allocator.
        return batch, ready, cpu_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        try:
            result = self.futures.popleft().result()
        except BaseException:
            self.close()
            raise
        if result is None:
            self.close()
            raise StopIteration
        batch, ready, _source = result
        if ready is not None:
            consumer = torch.cuda.current_stream(self.device)
            consumer.wait_event(ready)
            _record_stream(batch, consumer)
        self.futures.append(self.executor.submit(self._load))
        return batch

    def close(self):
        if self.closed:
            return
        self.closed = True
        for future in self.futures:
            future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.stream is not None:
            self.stream.synchronize()
        self.futures.clear()
        close_source = getattr(self.source, 'close', None)
        if close_source is not None:
            close_source()
