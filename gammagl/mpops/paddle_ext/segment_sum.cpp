#include <paddle/extension.h>
#include <vector>
#include "cpu/segment_sum_cpu.h"

#ifdef COMPILE_WITH_CUDA
#include "cuda/segment_sum_cuda.h"
#endif


std::vector<paddle::Tensor> SegmentSumForward(const paddle::Tensor& x,
                                                 const paddle::Tensor& index,
                                                 int64_t n) {
  // if (paddle::platform::is_cpu_place(x.place()) && paddle::platform::is_cpu_place(x.place())) { // is_xpu() may not support
  if (x.is_cpu() && index.is_cpu()) {  
    return segment_sum_cpu_forward(x, index, n);
#ifdef COMPILE_WITH_CUDA
  // } else if (paddle::platform::is_gpu_place(x.place()) && paddle::platform::is_gpu_place(x.place())) {
  } else if (x.is_gpu() && index.is_gpu()) {
    return segment_sum_cuda_forward(x, index, n);
#endif
  } else {
    PD_THROW("Unsupported device type.");
  }
}

std::vector<paddle::Tensor> SegmentSumBackward(const paddle::Tensor& index,
                                         const paddle::Tensor& grad_out) {
  auto grad_x = paddle::gather(grad_out, index, 0);

  return {grad_x};
}

// To support static graph mode
std::vector<std::vector<int64_t>> SegmentForwardShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& index_shape,
    const int64_t& n) {
  return {{n, x_shape[1]}};
}
std::vector<std::vector<int64_t>> SegmentBackwardShape(
    const std::vector<int64_t> x_shape,
    const std::vector<int64_t>& index_shape,
    const std::vector<int64_t>& grad_out_shape) {
  return {x_shape};
}
std::vector<paddle::DataType> SegmentForwardDtype(paddle::DataType x_dtype,
                                                paddle::DataType index_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(unsorted_segment_sum)
    .Inputs({"x", "Index"})
    .Outputs({"Out"})
    .Attrs({"N: int64_t"})
    .SetKernelFn(PD_KERNEL(SegmentSumForward))
    .SetInferShapeFn(PD_INFER_SHAPE(SegmentForwardShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SegmentForwardDtype));

PD_BUILD_GRAD_OP(unsorted_segment_sum)
    .Inputs({"Index", paddle::Grad("Out")})
    .Outputs({paddle::Grad("x")})
    .SetKernelFn(PD_KERNEL(SegmentSumBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(SegmentBackwardShape));
