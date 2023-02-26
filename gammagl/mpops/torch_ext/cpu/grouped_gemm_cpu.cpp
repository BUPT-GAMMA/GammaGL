#include "grouped_gemm.h"

// Performs matrix multiplication across list of elements.
void grouped_matmul_out_kernel_at_impl(const std::vector<at::Tensor> input,
                                       const std::vector<at::Tensor> other,
                                       std::vector<at::Tensor> out) {
  for (size_t i = 0; i < out.size(); ++i) {
    at::matmul_out(const_cast<at::Tensor &>(out[i]), input[i], other[i]);
  }
}

std::vector<torch::Tensor>
grouped_matmul_kernel(const torch::TensorList input,
                      const torch::TensorList other) {
  const auto n_matrices = input.size();

  std::vector<torch::Tensor> input_contig;
  std::vector<torch::Tensor> other_contig;
  std::vector<torch::Tensor> out;

  input_contig.reserve(n_matrices);
  other_contig.reserve(n_matrices);
  out.reserve(n_matrices);

  for (size_t i = 0; i < n_matrices; ++i) {
    input_contig.emplace_back(input[i].contiguous());
    other_contig.emplace_back(other[i].contiguous());
    out.emplace_back(input_contig[i].new_empty(
        {input_contig[i].size(0), other_contig[i].size(-1)}));
  }

  grouped_matmul_out_kernel_at_impl(input_contig, other_contig, out);

  return out;
}

at::Tensor size_from_ptr(const at::Tensor &ptr) {
  return ptr.narrow(/*dim=*/0, /*start=*/1, /*length=*/ptr.numel() - 1) -
         ptr.narrow(/*dim=*/0, /*start=*/0, /*length=*/ptr.numel() - 1);
}

at::Tensor segment_matmul_kernel_cpu(const at::Tensor &input, const at::Tensor &ptr,
                                 const at::Tensor &other) {
  const auto size = size_from_ptr(ptr).cpu();
  const auto sizes = at::IntArrayRef(size.data_ptr<int64_t>(), size.numel());
  const auto input_contig = input.contiguous();
  const auto other_contig = other.contiguous();
  auto out = input_contig.new_empty({input.size(0), other.size(-1)});

  // AT_DISPATCH_ALL_TYPES(
  // input_contig.scalar_type(), "segment_matmul_kernel_cpu", [&] {
  const auto n = other_contig.size(-1);
  const auto k = input_contig.size(-1);
  // if (mkl_path_available<scalar_t>() && mkl_path_possible(sizes, n, k)) {
  //   segment_matmul_out_kernel_mkl_impl(input_contig, other_contig, out,
  //                                      sizes);
  // } else {
  auto outs = out.split_with_sizes(/*split_size=*/sizes, /*dim=*/0);
  for (auto &out_part : outs) {
    out_part.unsqueeze_(0);
  }

  grouped_matmul_out_kernel_at_impl(
      input_contig.split_with_sizes(/*split_size=*/sizes, /*dim=*/0),
      other_contig.split(/*split_size=*/1, /*dim=*/0), outs);
  // }
  // });

  return out;
}
