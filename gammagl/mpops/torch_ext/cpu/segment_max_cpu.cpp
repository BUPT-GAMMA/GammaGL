#include "segment_max_cpu.h"

#include <assert.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <limits>
#include <vector>
std::tuple<torch::Tensor, torch::Tensor> segment_max_cpu_forward(
    torch::Tensor& x, torch::Tensor& index, int64_t& N) {
  TORCH_CHECK(x.device().is_cpu(), "x must be CPU tensor");
  TORCH_CHECK(index.device().is_cpu(), "index must be CPU tensor");
  TORCH_CHECK_INDEX(
      index.dim() == 1, "index dimension should be 1, but got ", index.dim());
  TORCH_CHECK_INDEX(
      x.size(0) == index.size(0),
      "fisrt dimension of x and index should be same");

  x = x.contiguous();  // torch Tensor my not be contiguous.
  index = index.contiguous();

  auto sizes = x.sizes().vec();
  sizes[0] = N;
  torch::Tensor out = torch::zeros(sizes, x.options());
  torch::Tensor arg_out = torch::full_like(out, out.size(0), index.options());
  if (x.numel() == 0) {
    return std::make_tuple(out, arg_out);
  }

  auto E = x.size(0);
  auto K = x.numel() / E;
  auto index_data = index.data_ptr<int64_t>();
  auto arg_out_data = arg_out.data_ptr<int64_t>();

  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "segment_mean_cpu_forward", [&]() {
      out.fill_(std::numeric_limits<scalar_t>::lowest());
      auto x_data = x.data_ptr<scalar_t>();
      auto out_data = out.data_ptr<scalar_t>();

      int64_t idx;
    #ifdef COMPILE_WITH_OMP
    #pragma omp parallel for private(idx)
    #endif
      for (auto e = 0; e < E; ++e) {
        idx = index_data[e];
        TORCH_CHECK_INDEX(idx < N, "Index out of bounds: ", idx, " >= ", N);
        for (auto k = 0; k < K; ++k) {
          scalar_t current_val = x_data[e * K + k];
          scalar_t& max_val = out_data[idx * K + k];
          int64_t& max_idx = arg_out_data[idx * K + k];
    #ifdef COMPILE_WITH_OMP
    #pragma omp critical
    #endif
          {
            if (max_val < current_val) {
              max_val = current_val;
              max_idx = e;
            }
          }
        }
      }

  });

  return std::make_tuple(out, arg_out);
}
