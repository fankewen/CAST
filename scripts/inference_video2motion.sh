#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Select one checkpoint; its configuration has the same filename stem.
# model=CAST_B_dinov2
# model=CAST_B_dinov2_no_grace
# model=CAST_B_dinov3
# model=CAST_B_dinov3_no_grace
model=CAST_L_dinov3

input_video="/path/to/input.mp4"
# Rest skeleton from the target cache.
static_path="/path/to/static.npy"
# Original character FBX (Zoo/Mixamo), or Mobjaverse raw_data.npz.
mesh_source="/path/to/character.fbx"
# mesh_source="/path/to/Mobjaverse/mobjaverse/<asset-id>/raw_data.npz"
blender="/path/to/blender"
export BLENDER_EXECUTABLE="$blender"

# Select one lighting preset. Zoo is recommended for other FBX meshes.
lighting_preset="zoo"  # zoo | mobj | mixamo
# The final video background is separate from the world light contribution.
background="white"
# Optional light-energy override. Leave empty to use the selected preset.
# Zoo: point, spot 1, spot 2, spot 3, area (1000, 500, 500, 500, 700).
# Mixamo: key, fill, rim (44, 28.6, 44), each multiplied by the clip-wide
# world-space mesh AABB maximum edge squared; fixed throughout the clip.
# Mobj: one directional light; default 7.5 textured / 1.5 untextured.
# Example for a brighter Zoo area light: light_energies=(1000 500 500 500 900)
light_energies=()

# Optional Blender-only adjustments for Zoo/Mixamo.
# Exposure is in stops: +1 doubles displayed brightness.
exposure="0"
# World-light override. Empty uses the selected preset:
# Mixamo 0.385 at RGB 0.17; Zoo 1.0 at RGB 0.050876.
world_strength=""
lighting_args=(--lighting-preset "$lighting_preset" --background "$background")
if (( ${#light_energies[@]} )); then
  lighting_args+=(--light-energies "${light_energies[@]}")
fi
case "$lighting_preset" in
  zoo|mixamo)
    lighting_args+=(--exposure "$exposure")
    if [[ -n "$world_strength" ]]; then
      lighting_args+=(--world-strength "$world_strength")
    fi
    ;;
  mobj) ;;
  *) echo "Unknown lighting_preset: $lighting_preset" >&2; exit 2 ;;
esac
sam2_checkpoint="/path/to/sam2.1_hiera_large.pt"
sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"
checkpoint="$repo_root/checkpoints/${model}.pt"
output_dir="$repo_root/results/inference/${model}"
prediction_dir="$output_dir/prediction"
mesh_asset_dir="$output_dir/mesh_asset"
mesh_path="$mesh_asset_dir/mesh_asset.npz"

# 1. Extract the target's skinned mesh directly from its original source file.
python -m scripts.build_mesh_asset \
  --source "$mesh_source" \
  --static "$static_path" \
  --output-dir "$mesh_asset_dir" \
  --blender "$blender"

# Set this to an existing mask video to skip SAM2 generation.
input_mask_video=""
if [[ -z "$input_mask_video" ]]; then
  input_mask_video="$output_dir/preprocess/$(basename "${input_video%.*}")_sam2_mask.mkv"
  mkdir -p "$(dirname "$input_mask_video")"
  # 2. Generate a foreground mask video with SAM2.
  python -m inference.offline_mask \
    --input-video "$input_video" \
    --output-mask-video "$input_mask_video" \
    --checkpoint "$sam2_checkpoint" \
    --model-cfg "$sam2_model_cfg" \
    --device cuda:0
fi

mkdir -p "$prediction_dir"
# 3. Predict the selected skeleton's joint rotations from the masked video.
python -m inference.video2motion_unlabeled \
  --input-video "$input_video" \
  --input-mask-video "$input_mask_video" \
  --static "$static_path" \
  --config "configs/experiment/${model}.yaml" \
  --checkpoint "$checkpoint" \
  --output-dir "$prediction_dir" \
  --window-overlap 61 \
  --device cuda:0

# 4. Drive the extracted mesh and render the comparison video.
python -m inference.video2motion_visualize \
  --input-video "$input_video" \
  --prediction-dir "$prediction_dir" \
  --static "$static_path" \
  --mesh-path "$mesh_path" \
  --output-dir "$output_dir" \
  --checkpoint "$checkpoint" \
  --panel-size 512 \
  --center-crop-scale 0.62 \
  "${lighting_args[@]}"
