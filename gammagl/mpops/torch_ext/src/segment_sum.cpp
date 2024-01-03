#include "../include/segment_sum.h"

#include <assert.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <vector>

#include "../cpu/segment_sum_cpu.h"
#ifdef COMPILE_WITH_CUDA
#include "../cuda/segment_sum_cuda.h"
#endif
#include "../include/utils.h"

using torch::autograd::AutogradContext;

inline torch::Tensor sum_device_dispatch_forward(
    torch::Tensor& x, torch::Tensor& index, int64_t& N) {
  if (x.is_cuda() && index.is_cuda()) {
#ifdef COMPILE_WITH_CUDA
    return segment_sum_cuda_forward(x, index, N);
#else
    AT_ERROR("Compiled with CUDA support while tensor is on GPU!");
#endif
  }
  if (x.is_cpu() && index.is_cpu()) {
    return segment_sum_cpu_forward(x, index, N);
  } else {
    AT_ERROR("Tensor device inconsistent error.");
  }
}

torch::Tensor SegmentSum::forward(
    AutogradContext* ctx, torch::Tensor x, torch::Tensor index, int64_t N) {
  ctx->saved_data["x_shape"] = x.sizes();
  auto result = sum_device_dispatch_forward(x, index, N);
  ctx->save_for_backward({index});
  return result;
}

std::vector<torch::Tensor> SegmentSum::backward(
    AutogradContext* ctx, std::vector<torch::Tensor> grad_outs) {
  auto grad_out = grad_outs[0];
  auto saved = ctx->get_saved_variables();
  auto index = saved[0];
  auto x_shape = list2vec(ctx->saved_data["x_shape"].toIntList());
  torch::Tensor grad_in = torch::zeros(x_shape, grad_out.options());
  torch::Tensor selected = grad_out.index_select(0, index);
  grad_in.copy_(selected);

  return {grad_in, torch::Tensor(), torch::Tensor()};
}
