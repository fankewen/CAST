from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from settings import SETTINGS
import torch
import flash_attn
import xformers
import xformers.ops
from ops.fused_depth_geometry import fused_axis_angle_to_matrix
import json

q=torch.randn(1,128,8,64,device='cuda',dtype=torch.bfloat16)
with torch.inference_mode():
    f=flash_attn.flash_attn_func(q,q,q,causal=True)
    x=xformers.ops.memory_efficient_attention(q,q,q)
    ref=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2).float(),q.transpose(1,2).float(),q.transpose(1,2).float()).transpose(1,2)
    assert torch.allclose(x.float(),ref,atol=.025,rtol=.025)
    assert torch.isfinite(f).all()
    r=fused_axis_angle_to_matrix(torch.randn(16,3,device='cuda')*.1)
    assert torch.allclose(r@r.transpose(-1,-2),torch.eye(3,device='cuda')[None],atol=1e-5)
    torch.cuda.synchronize()
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
                  'flash_attn':flash_attn.__version__,'xformers':xformers.__version__,
                  'custom_cuda_kernel':'PASS','attention_kernels':'PASS'},indent=2))
