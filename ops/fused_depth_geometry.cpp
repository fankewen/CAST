#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> fused_depth_geometry_cuda(
    torch::Tensor provisional,
    torch::Tensor rest_direction,
    torch::Tensor target,
    torch::Tensor target_has_direction,
    torch::Tensor confidence,
    double eps);
torch::Tensor axis_angle_to_matrix_cuda(torch::Tensor axis_angle);
torch::Tensor matrix_to_axis_angle_cuda(torch::Tensor matrix);
torch::Tensor minimal_axis_angle_cuda(
    torch::Tensor source, torch::Tensor target, double eps);
std::vector<torch::Tensor> confidence_weighted_kabsch_cuda(
    torch::Tensor source, torch::Tensor target,
    torch::Tensor confidence, double eps);
torch::Tensor visual_gate_cuda(
    torch::Tensor static_hidden, torch::Tensor dynamic,
    torch::Tensor dynamic_weight, torch::Tensor hidden_weight,
    torch::Tensor hidden_bias, torch::Tensor output_weight,
    torch::Tensor output_bias, torch::Tensor token_mask);
void fixed_visual_depth_cuda(
    torch::Tensor baseline, torch::Tensor rest, torch::Tensor target,
    torch::Tensor has_direction, torch::Tensor confidence,
    torch::Tensor aggregate_confidence, torch::Tensor static_hidden,
    torch::Tensor dynamic_weight, torch::Tensor hidden_weight,
    torch::Tensor hidden_bias, torch::Tensor output_weight,
    torch::Tensor output_bias, torch::Tensor depth_flat,
    torch::Tensor parent_flat, torch::Tensor child_count,
    torch::Tensor root_mask, torch::Tensor valid_parent,
    torch::Tensor eligible, torch::Tensor frame_mask,
    torch::Tensor corrected_global, torch::Tensor corrected_local,
    torch::Tensor gate_out, torch::Tensor observable_out,
    double fixed_gate_scale, int64_t gate_override, double eps);

std::vector<torch::Tensor> forward(
    torch::Tensor provisional,
    torch::Tensor rest_direction,
    torch::Tensor target,
    torch::Tensor target_has_direction,
    torch::Tensor confidence,
    double eps) {
  TORCH_CHECK(provisional.is_cuda(), "provisional must be CUDA");
  TORCH_CHECK(provisional.scalar_type() == torch::kFloat32, "provisional must be float32");
  TORCH_CHECK(rest_direction.scalar_type() == torch::kFloat32, "rest_direction must be float32");
  TORCH_CHECK(target.scalar_type() == torch::kFloat32, "target must be float32");
  TORCH_CHECK(confidence.scalar_type() == torch::kFloat32, "confidence must be float32");
  TORCH_CHECK(target_has_direction.scalar_type() == torch::kBool, "target_has_direction must be bool");
  TORCH_CHECK(provisional.dim() == 4 && provisional.size(2) == 3 && provisional.size(3) == 3,
              "provisional must be [N,T,3,3]");
  TORCH_CHECK(rest_direction.dim() == 3 && rest_direction.size(2) == 3,
              "rest_direction must be [N,K,3]");
  TORCH_CHECK(target.dim() == 4 && target.size(0) == provisional.size(0) &&
              target.size(1) == provisional.size(1) && target.size(2) == rest_direction.size(1) &&
              target.size(3) == 3, "target shape mismatch");
  const std::vector<int64_t> expected_shape = {
      provisional.size(0), provisional.size(1), rest_direction.size(1)};
  TORCH_CHECK(target_has_direction.sizes().vec() == expected_shape,
              "target_has_direction shape mismatch");
  TORCH_CHECK(confidence.sizes() == target_has_direction.sizes(), "confidence shape mismatch");
  return fused_depth_geometry_cuda(provisional, rest_direction, target,
                                   target_has_direction, confidence, eps);
}

torch::Tensor axis_angle_to_matrix(torch::Tensor axis_angle) {
  TORCH_CHECK(axis_angle.is_cuda(), "axis_angle must be CUDA");
  TORCH_CHECK(axis_angle.scalar_type() == torch::kFloat32, "axis_angle must be float32");
  TORCH_CHECK(axis_angle.dim() >= 1 && axis_angle.size(-1) == 3,
              "axis_angle must end in dimension 3");
  return axis_angle_to_matrix_cuda(axis_angle.contiguous());
}

torch::Tensor matrix_to_axis_angle(torch::Tensor matrix) {
  TORCH_CHECK(matrix.is_cuda(), "matrix must be CUDA");
  TORCH_CHECK(matrix.scalar_type() == torch::kFloat32, "matrix must be float32");
  TORCH_CHECK(matrix.dim() >= 2 && matrix.size(-2) == 3 && matrix.size(-1) == 3,
              "matrix must end in [3,3]");
  return matrix_to_axis_angle_cuda(matrix.contiguous());
}

torch::Tensor minimal_axis_angle(
    torch::Tensor source, torch::Tensor target, double eps) {
  TORCH_CHECK(source.is_cuda() && target.is_cuda(),
              "source and target must be CUDA");
  TORCH_CHECK(source.scalar_type() == torch::kFloat32 &&
              target.scalar_type() == torch::kFloat32,
              "source and target must be float32");
  TORCH_CHECK(source.sizes() == target.sizes() && source.size(-1) == 3,
              "source and target shape mismatch");
  return minimal_axis_angle_cuda(
      source.contiguous(), target.contiguous(), eps);
}

std::vector<torch::Tensor> confidence_weighted_kabsch(
    torch::Tensor source, torch::Tensor target,
    torch::Tensor confidence, double eps) {
  TORCH_CHECK(source.is_cuda() && target.is_cuda() && confidence.is_cuda(),
              "Kabsch inputs must be CUDA");
  TORCH_CHECK(source.scalar_type() == torch::kFloat32 &&
              target.scalar_type() == torch::kFloat32 &&
              confidence.scalar_type() == torch::kFloat32,
              "Kabsch inputs must be float32");
  TORCH_CHECK(source.sizes() == target.sizes() && source.size(-1) == 3,
              "source and target shape mismatch");
  TORCH_CHECK(confidence.sizes().vec() ==
              std::vector<int64_t>(source.sizes().begin(), source.sizes().end() - 1),
              "confidence shape mismatch");
  return confidence_weighted_kabsch_cuda(
      source.contiguous(), target.contiguous(), confidence.contiguous(), eps);
}

torch::Tensor visual_gate(
    torch::Tensor static_hidden, torch::Tensor dynamic,
    torch::Tensor dynamic_weight, torch::Tensor hidden_weight,
    torch::Tensor hidden_bias, torch::Tensor output_weight,
    torch::Tensor output_bias, torch::Tensor token_mask) {
  TORCH_CHECK(static_hidden.is_cuda(), "visual gate inputs must be CUDA");
  TORCH_CHECK(static_hidden.scalar_type() == torch::kFloat32 &&
              dynamic.scalar_type() == torch::kFloat32,
              "visual gate inputs must be float32");
  TORCH_CHECK(static_hidden.size(-1) <= 256 && dynamic.size(-1) == 2,
              "fused visual gate supports hidden <=256 and dynamic dim 2");
  return visual_gate_cuda(
      static_hidden.contiguous(), dynamic.contiguous(),
      dynamic_weight.contiguous(), hidden_weight.contiguous(),
      hidden_bias.contiguous(), output_weight.contiguous(),
      output_bias.contiguous(), token_mask.contiguous());
}

void fixed_visual_depth(
    torch::Tensor baseline, torch::Tensor rest, torch::Tensor target,
    torch::Tensor has_direction, torch::Tensor confidence,
    torch::Tensor aggregate_confidence, torch::Tensor static_hidden,
    torch::Tensor dynamic_weight, torch::Tensor hidden_weight,
    torch::Tensor hidden_bias, torch::Tensor output_weight,
    torch::Tensor output_bias, torch::Tensor depth_flat,
    torch::Tensor parent_flat, torch::Tensor child_count,
    torch::Tensor root_mask, torch::Tensor valid_parent,
    torch::Tensor eligible, torch::Tensor frame_mask,
    torch::Tensor corrected_global, torch::Tensor corrected_local,
    torch::Tensor gate_out, torch::Tensor observable_out,
    double fixed_gate_scale, int64_t gate_override, double eps) {
  TORCH_CHECK(baseline.is_cuda(), "fixed depth inputs must be CUDA");
  TORCH_CHECK(baseline.scalar_type()==torch::kFloat32 &&
              rest.scalar_type()==torch::kFloat32 &&
              target.scalar_type()==torch::kFloat32 &&
              confidence.scalar_type()==torch::kFloat32 &&
              aggregate_confidence.scalar_type()==torch::kFloat32 &&
              static_hidden.scalar_type()==torch::kFloat32,
              "fixed depth floating inputs must be float32");
  TORCH_CHECK(depth_flat.scalar_type()==torch::kInt64 &&
              parent_flat.scalar_type()==torch::kInt64 &&
              child_count.scalar_type()==torch::kInt64,
              "fixed depth indices must be int64");
  TORCH_CHECK(rest.size(2)<=16, "fixed depth kernel supports <=16 child slots");
  TORCH_CHECK(static_hidden.size(-1)<=256,
              "fixed depth kernel supports hidden <=256");
  fixed_visual_depth_cuda(
      baseline.contiguous(),rest.contiguous(),target.contiguous(),
      has_direction.contiguous(),confidence.contiguous(),
      aggregate_confidence.contiguous(),static_hidden.contiguous(),
      dynamic_weight.contiguous(),hidden_weight.contiguous(),
      hidden_bias.contiguous(),output_weight.contiguous(),
      output_bias.contiguous(),depth_flat.contiguous(),parent_flat.contiguous(),
      child_count.contiguous(),root_mask.contiguous(),valid_parent.contiguous(),
      eligible.contiguous(),frame_mask.contiguous(),corrected_global,
      corrected_local,gate_out,observable_out,fixed_gate_scale,gate_override,eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Fused depth geometry forward (CUDA)");
  m.def("axis_angle_to_matrix", &axis_angle_to_matrix,
        "Fused axis-angle to matrix (CUDA)");
  m.def("matrix_to_axis_angle", &matrix_to_axis_angle,
        "Fused matrix to axis-angle (CUDA)");
  m.def("minimal_axis_angle", &minimal_axis_angle,
        "Fused minimal rotation axis-angle (CUDA)");
  m.def("confidence_weighted_kabsch", &confidence_weighted_kabsch,
        "Fused confidence-weighted Kabsch (CUDA)");
  m.def("visual_gate", &visual_gate, "Fused visual gate (CUDA)");
  m.def("fixed_visual_depth", &fixed_visual_depth,
        "Fixed-topology visual correction depth (CUDA)");
}
