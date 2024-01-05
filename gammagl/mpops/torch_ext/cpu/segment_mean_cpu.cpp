#include "segment_mean_cpu.h"

#include <assert.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <vector>
torch::Tensor segment_mean_cpu_forward(
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
  torch::Tensor arg_out = torch::full_like(out, 0., index.options());
  if (x.numel() == 0) {
    return out;
  }

  auto E = x.size(0);
  auto K = x.numel() / x.size(0);
  auto index_data = index.data_ptr<int64_t>();
  auto arg_out_data = arg_out.data_ptr<int64_t>();

  // AT_DISPATCH_ALL_TYPES(x.scalar_type(), "__ops_name", [&] {
  using scalar_t = float;
  auto x_data = x.data_ptr<scalar_t>();
  auto out_data = out.data_ptr<scalar_t>();

  torch::Tensor degree = torch::zeros({1, index.size(0)}, x.options());
  auto degree_data = degree.data_ptr<scalar_t>();

#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
  for (auto e = 0; e < E; ++e) {
    auto idx = index_data[e];
    degree_data[idx] += 1;
    for (auto k = 0; k < K; ++k) {
#ifdef COMPILE_WITH_OMP
#pragma omp critical
#endif
      out_data[idx * K + k] += x_data[e * K + k];
      arg_out_data[idx * K + k] = e;
    }
  }
  // });
  out = out.contiguous();
  degree = degree.contiguous();

#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
  for (auto e = 0; e < E; ++e) {
    if (degree_data[e] > 1) {
      for (auto k = 0; k < K; ++k) {
#ifdef COMPILE_WITH_OMP
#pragma omp critical
#endif
        out_data[e * K + k] /= degree_data[e];
      }
    }
  }

  return out;
}
