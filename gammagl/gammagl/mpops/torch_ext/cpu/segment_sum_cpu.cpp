#include "segment_sum_cpu.h"

#include <ATen/ATen.h>
#include <assert.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <vector>
torch::Tensor segment_sum_cpu_forward(
    torch::Tensor& x, torch::Tensor& index, int64_t& N) {
  TORCH_CHECK(x.device().is_cpu(), "x must be CPU tensor");
  TORCH_CHECK(index.device().is_cpu(), "index must be CPU tensor");
  TORCH_CHECK_INDEX(
      index.dim() == 1, "index dimension should be 1, but got ", index.dim());
  TORCH_CHECK_INDEX(
      x.size(0) == index.size(0),
      "fisrt dimension of x and index should be same");

  x = x.contiguous();  // Make sure x is contiguous.
  index = index.contiguous();

  auto sizes = x.sizes().vec();
  sizes[0] = N;
  torch::Tensor out = torch::zeros(sizes, x.options());

  if (x.numel() == 0) {
    return out;
  }

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "segment_sum_cpu_forward", [&]() {
        // Get data pointers for index, arg_out, and x.
        auto index_data = index.data_ptr<int64_t>();
        auto x_data = x.data_ptr<scalar_t>();  // Assuming x is of type float.
        auto out_data = out.data_ptr<scalar_t>();

        auto E = index.size(0);          // Number of elements to process.
        auto K = x.numel() / x.size(0);  // Size of the inner dimension.

#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
        // Iterate over each element in x.
        for (auto e = 0; e < E; ++e) {
          auto idx = index_data[e];
          // Handle accumulation for different dimensions.
          for (auto k = 0; k < K; ++k) {
#ifdef COMPILE_WITH_OMP
#pragma omp critical
#endif
            out_data[idx * K + k] += x_data[e * K + k];
          }
        }
      });

  return out;
}
