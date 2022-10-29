#include "paddle/extension.h"

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

std::vector<paddle::Tensor> segment_sum_cpu_backward(
    // const paddle::Tensor& src,
    const paddle::Tensor& index,
    const paddle::Tensor& grad_out) {
  CHECK_INPUT(index);
  CHECK_INPUT(grad_out);

  auto grad_src = paddle::gather(grad_out, index, 0);

  return {grad_src};
}


#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> segment_sum_cuda_forward(const paddle::Tensor& src,
                                                 const paddle::Tensor& index,
                                                 int64_t n);
                                                 // `n` should not be ref type
std::vector<paddle::Tensor> segment_sum_cuda_backward(
    const paddle::Tensor& index,
    const paddle::Tensor& grad_out);
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
    PD_THROW("Unsupported device type for forward function of custom relu operator.");
  }
}

std::vector<paddle::Tensor> SegmentSumBackward(const paddle::Tensor& index,
                                         const paddle::Tensor& grad_out) {
  if (index.is_cpu()) {
    return segment_sum_cpu_backward(index, grad_out);
#ifdef PADDLE_WITH_CUDA
  } else if (index.is_gpu()) {
    return segment_sum_cuda_backward(index, grad_out);
#endif
  } else {
    PD_THROW("Unsupported device type for backward function of custom relu operator.");
  }
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
