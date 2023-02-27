#include "segment_max_cpu.h"
#include <torch/torch.h>
#include <assert.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <vector>
std::tuple<torch::Tensor, torch::Tensor> segment_max_cpu_forward(torch::Tensor& x,
                                                     torch::Tensor& index,
                                                     int64_t& N) {
  TORCH_CHECK(x.device().is_cpu(), "x must be CPU tensor");
  TORCH_CHECK(index.device().is_cpu(), "index must be CPU tensor");
  TORCH_CHECK_INDEX(x.dim() == 2, "x dimension should be 2, but got ",
                    x.dim());
  TORCH_CHECK_INDEX(index.dim() == 1, "index dimension should be 1, but got ",
                    index.dim());
  TORCH_CHECK_INDEX(x.size(0) == index.size(0),
                    "fisrt dimension of x and index should be same");

  x = x.contiguous(); // torch Tensor my not be contiguous.

  std::vector<int64_t> sizes = {N, x.size(1)};
  // sizes[0] = N > *index.max().data_ptr<int64_t>()
  //                ? N
  //                : *index.max().data_ptr<int64_t>();
  torch::Tensor out = torch::empty(sizes, x.options());
  torch::Tensor arg_out = torch::full_like(out, 0., index.options());
  if (x.numel() == 0) {
    out.fill_(0.);
    return std::make_tuple(out, arg_out);
  }

  out.fill_(std::numeric_limits<int64_t>::lowest());
  auto E = x.size(0);
  auto K = x.size(1);
  auto index_data = index.data_ptr<int64_t>();
  auto arg_out_data = arg_out.data_ptr<int64_t>();

  // AT_DISPATCH_ALL_TYPES(x.scalar_type(), "__ops_name", [&] {
  using scalar_t = float;
  auto x_data = x.data_ptr<scalar_t>();
  auto out_data = out.data_ptr<scalar_t>();

  int64_t idx;
#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
  for (auto e = 0; e < E; ++e) {
    idx = index_data[e];
    for (auto k = 0; k < K; ++k) {
#ifdef COMPILE_WITH_OMP
#pragma omp critical
#endif
      if (out_data[idx * K + k] < x_data[e * K + k]) {
        out_data[idx * K + k] = x_data[e * K + k];
        arg_out_data[idx * K + k] = e;
      }
    }
  }
  out.masked_fill_(out == std::numeric_limits<int64_t>::lowest(), (scalar_t)0);
  // });

  return std::make_tuple(out, arg_out);
}
