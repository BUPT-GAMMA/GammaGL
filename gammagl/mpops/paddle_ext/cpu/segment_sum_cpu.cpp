#include <paddle/extension.h>
#include "segment_sum_cpu.h"
#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

std::vector<paddle::Tensor> segment_sum_cpu_forward(const paddle::Tensor& x,
                                                 const paddle::Tensor& index,
                                                 int64_t n) {
  CHECK_INPUT(x);
  CHECK_INPUT(index);
  // NOTE: paddle tensor seems to be contiguous
  std::vector<int64_t> sizes = {
      n, x.shape()[1]};  // TODO: maybe need max(max(index), N)
  auto out = paddle::full(sizes, 0., x.dtype(), x.place());
  if (x.numel() == 0) {
    return {out};
  }

  auto E = x.shape()[0];
  auto K = x.shape()[1];
  auto index_data = index.data<int64_t>();

  // PD_DISPATCH_FLOATING_TYPES(x.type(), "cpu_segment_kernel", ([&] {
  using data_t = float;
  auto x_data = x.data<data_t>();
  auto out_data = out.data<data_t>();

  int64_t idx;
#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
  for (auto e = 0; e < E; ++e) {
    idx = index_data[e];
    for (auto k = 0; k < K; ++k) {
#ifdef COMPILE_WITH_OMP
#pragma omp atomic
#endif
      out_data[idx * K + k] = out_data[idx * K + k] + x_data[e * K + k];
    }
  }
  // }));

  return {out};
}