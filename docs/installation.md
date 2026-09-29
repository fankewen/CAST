# Installation

[Back to README](../README.md)

Linux is the default installation target. See [Windows](#windows) for
platform-specific settings. Run commands from the repository root unless
noted otherwise.

## Verified Stack

Other versions may work, but the reference setup uses:

| Component | Version |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.7.0+cu128 |
| TorchVision | 0.22.0+cu128 |
| FlashAttention | 2.8.3 |
| xFormers | 0.0.30 |
| CUDA toolkit | 12.8 |

## Linux

### 1. Create the Environment

```bash
conda create -n cast python=3.10 -y
conda activate cast
```

### 2. Prepare the CUDA Toolkit and Build Tools

If Ninja and a host C++ compiler are missing, install them
(Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y ninja-build gcc g++
```

If CUDA 12.8 is not installed, install it into the `cast` environment:

```bash
conda install -n cast -c nvidia cuda-toolkit=12.8 -y

export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
```

If using an existing CUDA toolkit, set `CUDA_HOME` to its installation directory.
Prepare the toolkit and compiler before building FlashAttention or the custom
CUDA operator.

### 3. Install Python Dependencies

```bash
python -m pip install \
  torch==2.7.0 \
  torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install xformers==0.0.30
```

FlashAttention enables optimized attention paths; the code provides a fallback
when it is unavailable. Use builds compatible with your PyTorch, CUDA, Python,
and GPU versions.

### 4. Compile the Custom CUDA Operator

Set the target GPU architecture:

```bash
export TORCH_CUDA_ARCH_LIST="8.9"
# RTX 40-series  8.9
# A100           8.0
# H100           9.0
```

Build the extension in place:

```bash
python ops/setup.py build_ext --inplace
```

The compiled module is written to
`ops/cast_fused_depth_geometry.<python-extension-suffix>`.

## Backbone Assets

CAST supports DINOv2 or DINOv3. Each backbone requires a local checkout of its
official repository and pretrained weights. The helper script fetches both
source checkouts and the DINOv2 weights:

```bash
bash scripts/setup_backbones.sh
```

The script pins the source revisions, verifies the DINOv2 weights by SHA-256,
and prints the configuration paths.

| Backbone | Source checkout | Weights | `data.image_size` |
| --- | --- | --- | --- |
| DINOv2 ViT-L/14 | `third_party/dinov2` | `third_party/weights/dinov2_vitl14_pretrain.pth` | 224 |
| DINOv3 ViT-L/16 | `third_party/dinov3` | `third_party/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | 256 |

**The script does not download DINOv3 weights.** Request access at
[facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
and place the checkpoint at the path above.

Choose the backbone that matches the CAST checkpoint. Set `data.image_size`
and `image_backbone_cfg` together: the settings below produce a 16×16 token
grid for each backbone. DINOv2 also takes `input_size: 224`.

```yaml
# DINOv2
data:
  image_size: 224
model:
  params:
    image_backbone_cfg:
      target: models.frozen_dinov2.FrozenDinoV2Encoder
      params:
        repo: third_party/dinov2
        weights: third_party/weights/dinov2_vitl14_pretrain.pth
        input_size: 224
```

```yaml
# DINOv3
data:
  image_size: 256
model:
  params:
    image_backbone_cfg:
      target: models.frozen_dinov3.FrozenDinoV3Encoder
      params:
        repo: third_party/dinov3
        weights: third_party/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

Backbone code and weights are obtained from their original projects and are
subject to the respective license terms:
[DINOv2](https://github.com/facebookresearch/dinov2) and
[DINOv3](https://github.com/facebookresearch/dinov3).

## Optional: SAM2 Mask Generation

SAM2 is used by the optional mask-generation workflow.

- Official repository: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- Checkpoint: [SAM 2.1 Hiera Large](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
- Model config: `configs/sam2.1/sam2.1_hiera_l.yaml` (provided by SAM2)

Install the official repository, then return to the CAST repository root:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
python -m pip install -e .
cd ..
```

Download the checkpoint linked above, then pass its path and model config to
`inference/offline_mask.py`:

```bash
python inference/offline_mask.py \
  --input-video /path/to/input.mp4 \
  --output-mask-video /path/to/mask.mp4 \
  --checkpoint /path/to/sam2.1_hiera_large.pt \
  --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml
```

## Paths to Configure

Replace `/path/to/...` and `/absolute/path/to/...` placeholders in the chosen
configuration before use. Backbone settings below are under
`model.params.image_backbone_cfg.params`.

| Setting | Path |
| --- | --- |
| `repo` | `third_party/dinov2` or `third_party/dinov3` |
| `weights` | Matching pretrained backbone checkpoint from the table above |
| `compile_cache_dir` | Writable directory for the `torch.compile` cache |
| `data.sources.*.dataset_root` | Pre-rendered RGB cache, only for dataset-based evaluation |

Run from the repository root so relative paths in the examples resolve correctly.

## Windows

Use the version list above as the reference environment, with Windows-compatible
builds of the packages. The Linux shell commands are not directly executable
in PowerShell; the Windows-specific settings are listed below. The backbone
setup script requires Bash (for example, Git Bash).

Install Visual Studio 2022 Build Tools with:

```text
Microsoft.VisualStudio.Workload.VCTools
Microsoft.VisualStudio.Component.VC.Tools.x86.x64
Microsoft.VisualStudio.Component.Windows11SDK.26100
```

Open the `x64 Native Tools Command Prompt for VS 2022`, start `powershell`
inside it, and activate the environment before applying these settings:

```powershell
conda activate cast
python -m pip install ninja==1.13.0
python -m pip install triton-windows==3.3.1.post21
$env:TORCHINDUCTOR_COMPILE_THREADS = "1"
```

For a Conda-installed CUDA toolkit, set the Windows CUDA paths:

```powershell
$env:CUDA_HOME = "$env:CONDA_PREFIX\Library"
$env:CUDA_PATH = $env:CUDA_HOME
```

Create the x64 CUDA library location expected by PyTorch's extension builder:

```powershell
$cudaLib = "$env:CONDA_PREFIX\Library\lib"
New-Item -ItemType Directory -Force "$cudaLib\x64" | Out-Null
Copy-Item "$cudaLib\cudart.lib" "$cudaLib\x64\cudart.lib" -Force
Copy-Item "$cudaLib\cudart_static.lib" "$cudaLib\x64\cudart_static.lib" -Force
```

Build from the repository root in the same PowerShell session, adjusting the
GPU architecture as needed:

```powershell
$env:TORCH_CUDA_ARCH_LIST = "8.9"
python ops\setup.py build_ext --inplace
```
