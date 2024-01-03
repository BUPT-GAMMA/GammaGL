#include "../include/segment_mean.h"

#include <assert.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "../cpu/segment_mean_cpu.h"
#include "pybind11/pytypes.h"
#ifdef COMPILE_WITH_CUDA
#include "../cuda/segment_mean_cuda.h"
#endif
#include "../include/utils.h"

using torch::autograd::AutogradContext;

inline torch::Tensor mean_device_dispatch_forward(
    torch::Tensor& x, torch::Tensor& index, int64_t& N) {
  if (x.is_cuda() && index.is_cuda()) {
#ifdef COMPILE_WITH_CUDA
    return segment_mean_cuda_forward(x, index, N);
#else
    AT_ERROR("Compiled with CUDA support while tensor is on GPU!");
#endif
  } else if (x.is_cpu() && index.is_cpu()) {
    return segment_mean_cpu_forward(x, index, N);
  } else {
    AT_ERROR("Tensor device inconsistent error.");
  }
}

torch::Tensor SegmentMean::forward(
    AutogradContext* ctx, torch::Tensor x, torch::Tensor index, int64_t N) {
  ctx->saved_data["x_shape"] = x.sizes();
  auto result = mean_device_dispatch_forward(x, index, N);
  ctx->save_for_backward({index});
  return result;
}

std::vector<torch::Tensor> SegmentMean::backward(
    AutogradContext* ctx, std::vector<torch::Tensor> grad_outs) {
  auto grad_out = grad_outs[0];
  auto saved = ctx->get_saved_variables();
  auto index = saved[0];
  auto x_shape = list2vec(ctx->saved_data["x_shape"].toIntList());

  torch::Tensor grad_in = torch::zeros(x_shape, grad_out.options());
  torch::Tensor selected = grad_out.index_select(0, index);
  grad_in.copy_(selected);
  auto counts = torch::bincount(index);
  auto result = counts.index_select(0, index);

  std::vector<int64_t> result_shape(grad_in.dim(), 1);
  result_shape[0] = grad_in.size(0);
  auto index_expanded = result.view(result_shape);
  grad_in = grad_in / index_expanded;

  return {grad_in, torch::Tensor(), torch::Tensor()};
}
