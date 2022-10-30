#include <paddle/extension.h>
#include <vector>
#include "cpu/segment_sum_cpu.h"

#ifdef PADDLE_WITH_CUDA
#include "cuda/segment_sum_cuda.h"
#endif


std::vector<paddle::Tensor> SegmentSumForward(const paddle::Tensor& src,
                                                 const paddle::Tensor& index,
                                                 int64_t n) {
  // if (paddle::platform::is_cpu_place(src.place()) && paddle::platform::is_cpu_place(src.place())) { // is_xpu() may not support
  if (src.is_cpu() && index.is_cpu()) {  
    return segment_sum_cpu_forward(src, index, n);
#ifdef PADDLE_WITH_CUDA
  // } else if (paddle::platform::is_gpu_place(src.place()) && paddle::platform::is_gpu_place(src.place())) {
  } else if (src.is_gpu() && index.is_gpu()) {
    return segment_sum_cuda_forward(src, index, n);
#endif
  } else {
    PD_THROW("Unsupported device type.");
  }
}

std::vector<paddle::Tensor> SegmentSumBackward(const paddle::Tensor& index,
                                         const paddle::Tensor& grad_out) {
  auto grad_src = paddle::gather(grad_out, index, 0);

  return {grad_src};
}

// To support static graph mode
std::vector<std::vector<int64_t>> SegmentForwardShape(
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& index_shape,
    const int64_t& n) {
  return {{n, src_shape[1]}};
}
std::vector<std::vector<int64_t>> SegmentBackwardShape(
    const std::vector<int64_t> src_shape,
    const std::vector<int64_t>& index_shape,
    const std::vector<int64_t>& grad_out_shape) {
  return {src_shape};
}
std::vector<paddle::DataType> SegmentForwardDtype(paddle::DataType src_dtype,
                                                paddle::DataType index_dtype) {
  return {src_dtype};
}

PD_BUILD_OP(unsorted_segment_sum)
    .Inputs({"Src", "Index"})
    .Outputs({"Out"})
    .Attrs({"N: int64_t"})
    .SetKernelFn(PD_KERNEL(SegmentSumForward))
    .SetInferShapeFn(PD_INFER_SHAPE(SegmentForwardShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SegmentForwardDtype));

PD_BUILD_GRAD_OP(unsorted_segment_sum)
    .Inputs({"Index", paddle::Grad("Out")})
    .Outputs({paddle::Grad("Src")})
    .SetKernelFn(PD_KERNEL(SegmentSumBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(SegmentBackwardShape));
