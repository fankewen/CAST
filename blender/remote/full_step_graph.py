"""Fixed-rig, batch=1 graph for the already compiled DINOv3 + motion step.

No weights, dtypes, resolution, cache capacity or preprocessing changes. Capture
mutates temporal KV caches during warmup; snapshots restore those tensors in
place. New rig => destroy and rebuild this graph. Reset/rebind => use the model's
existing in-place reset_streaming/bind_reference APIs. One GPU caller at a time.
"""
import time
import torch
from inference.deploy_realtime_video2motion import _snapshot_caches, _restore_caches


class FullStepGraph:
    @torch.inference_mode()
    def __init__(self, model, warmup=3, verify=True):
        if model.device.type != 'cuda' or model.runner is None:
            raise ValueError('prepare and compile the fixed skeleton before capture')
        if model.use_cudagraphs:
            raise ValueError('full-step graph requires triton.cudagraphs=False; no nested graph mode')
        start=time.perf_counter()
        self.model=model
        self.state=model.state
        self.runner=model.runner
        self.static=torch.zeros((3,model.image_size,model.image_size),
                                dtype=model.neural_dtype,device=model.device)
        saved=_snapshot_caches(model.state)
        side=torch.cuda.Stream(device=model.device)
        side.wait_stream(torch.cuda.current_stream(model.device))
        with torch.cuda.stream(side):
            for _ in range(warmup):model.step(self.static)
        torch.cuda.current_stream(model.device).wait_stream(side)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output=model.step(self.static)
        _restore_caches(model.state,saved)
        torch.cuda.synchronize(model.device)
        self.max_validation_error=None
        if verify:
            # Real per-rig numerical comparison. Same reference, same cache start,
            # same three frames; excludes initialization/reference work from replay.
            inputs=[torch.full_like(self.static,value) for value in (-.5,0.,.75)]
            expected=[model.step(frame).detach().float().clone() for frame in inputs]
            _restore_caches(model.state,saved)
            actual=[self.step(frame).detach().float().clone() for frame in inputs]
            error=torch.stack([(a-b).abs().max() for a,b in zip(actual,expected)]).max().item()
            finite=all(torch.isfinite(a).all().item() for a in actual)
            _restore_caches(model.state,saved)
            self.max_validation_error=error
            if not finite or error>1e-5:
                raise RuntimeError(f'full-step graph validation failed: finite={finite}, max_abs_error={error}')
        self.capture_seconds=time.perf_counter()-start

    @torch.inference_mode()
    def step(self, frame):
        if self.model.state is not self.state or self.model.runner is not self.runner:
            raise RuntimeError('skeleton/runner changed; recapture full-step graph')
        if not isinstance(frame,torch.Tensor) or tuple(frame.shape)!=tuple(self.static.shape):
            raise ValueError(f'expected normalized single frame {tuple(self.static.shape)}')
        self.static.copy_(frame,non_blocking=True)
        self.graph.replay()
        # Output storage is reused by the next replay. Caller must consume/copy it
        # before calling step again (service immediately copies float motion to CPU).
        return self.output

    def info(self):
        return {'engine':'full_step_cuda_graph','enabled':True,
                'joints':self.model.joint_count,'image_size':self.model.image_size,
                'capture_seconds':round(self.capture_seconds,3),
                'per_rig_max_abs_error':self.max_validation_error,
                'scope':'DINOv3 + streaming mocap, fixed skeleton',
                'weight_or_precision_change':False}
