#include <paddle/extension.h>
#include "segment_sum_cpu.h"
#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

std::vector<paddle::Tensor> segment_sum_cpu_forward(const paddle::Tensor& src,
                                                 const paddle::Tensor& index,
                                                 int64_t n) {
  CHECK_INPUT(src);
  CHECK_INPUT(index);
  // NOTE: paddle tensor seems to be contiguous
  std::vector<int64_t> sizes = {
      n, src.shape()[1]};  // TODO: maybe need max(max(index), N)
  auto out = paddle::full(sizes, 0., src.dtype(), src.place());
  if (src.numel() == 0) {
    return {out};
  }

  auto E = src.shape()[0];
  auto K = src.shape()[1];
  auto index_data = index.data<int64_t>();

  // PD_DISPATCH_FLOATING_TYPES(src.type(), "cpu_segment_kernel", ([&] {
  using data_t = float;
  auto src_data = src.data<data_t>();
  auto out_data = out.data<data_t>();

  int64_t idx;
#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
  for (auto e = 0; e < E; ++e) {
    idx = index_data[e];
    for (auto k = 0; k < K; ++k) {
#ifdef COMPILE_WITH_OMP
#pragma omp atomic write
#endif
      out_data[idx * K + k] += src_data[e * K + k];
    }
  }
  // }));

  return {out};
}