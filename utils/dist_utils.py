### dist_utils.py ###
import torch.distributed as dist
import torch
import numpy as np
import os

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2 ** 32))

# ========= Distributed-training helpers =========
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        # Bind NCCL to the process-local CUDA device explicitly.  Relying only
        # on set_device leaves the rank→GPU mapping unknown to ProcessGroupNCCL
        # during its first barrier and can hang before the first forward.
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://", device_id=device
        )
        dist.barrier(device_ids=[local_rank])
        if is_main_process():
            print(f"[INFO] Initialized distributed training: rank {rank}/{world_size} (GPU {local_rank})")
        return True, rank, world_size, local_rank
    else:
        if is_main_process():
            print("[INFO] Distributed training is off; running on a single GPU.")
        return False, 0, 1, 0

def reduce_sum_scalar(x, device):
    t = torch.tensor(x, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def reduce_sum_tensor(tensor, device):
    """All-reduce one double tensor once for a fixed ordered accumulator set."""
    t = torch.as_tensor(tensor, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t
