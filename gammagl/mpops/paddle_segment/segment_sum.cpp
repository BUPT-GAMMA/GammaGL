#include "paddle/extension.h"

#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

std::vector<paddle::Tensor> SegmentSumCPUForward(const paddle::Tensor& src,
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

std::vector<paddle::Tensor> SegmentSumCPUBackward(
    // const paddle::Tensor& src,
    const paddle::Tensor& index,
    const paddle::Tensor& grad_out) {
  CHECK_INPUT(index);
  CHECK_INPUT(grad_out);

  auto grad_src = paddle::gather(grad_out, index, 0);

  return {grad_src};
}

// To support static graph mode
std::vector<std::vector<int64_t>> SegmentInferShape(
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& index_shape,
    const int64_t& N) {
  return {{N, src_shape[1]}};
}
std::vector<std::vector<int64_t>> SegmentBackwardShape(
    const std::vector<int64_t> src_shape,
    const std::vector<int64_t>& index_shape,
    const std::vector<int64_t>& grad_out_shape) {
  return {src_shape};
}
std::vector<paddle::DataType> SegmentInferDtype(paddle::DataType src_dtype,
                                                paddle::DataType index_dtype) {
  return {src_dtype};
}

PD_BUILD_OP(unsorted_segment_sum)
    .Inputs({"Src", "Index"})
    .Outputs({"Out"})
    .Attrs({"N: int64_t"})
    .SetKernelFn(PD_KERNEL(SegmentSumCPUForward))
    .SetInferShapeFn(PD_INFER_SHAPE(SegmentInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SegmentInferDtype));

PD_BUILD_GRAD_OP(unsorted_segment_sum)
    .Inputs({"Index", paddle::Grad("Out")})
    .Outputs({paddle::Grad("Src")})
    .SetKernelFn(PD_KERNEL(SegmentSumCPUBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(SegmentBackwardShape));
