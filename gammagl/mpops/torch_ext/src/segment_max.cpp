#include "../include/segment_max.h"
#include <assert.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "../cpu/segment_max_cpu.h"
#ifdef COMPILE_WITH_CUDA
#include "../cuda/segment_max_cuda.h"
#endif
#include <iostream>
#include <vector>
#include "../include/utils.h"

using torch::autograd::AutogradContext;

std::tuple<torch::Tensor, torch::Tensor>
inline max_device_dispatch_forward(torch::Tensor &x, torch::Tensor &index, int64_t &N) {
  if (x.is_cuda() && index.is_cuda()) {
#ifdef COMPILE_WITH_CUDA
    return segment_max_cuda_forward(x, index, N);
#else
    AT_ERROR("Compiled with CUDA support while tensor is on GPU!");
#endif
  } else if (x.is_cpu() && index.is_cpu()) {
    return segment_max_cpu_forward(x, index, N);
  } else {
    AT_ERROR("Tensor device inconsistent error.");
  }
}

torch::Tensor SegmentMax::forward(AutogradContext *ctx, torch::Tensor x,
                                  torch::Tensor index, int64_t N) {
  ctx->saved_data["x_shape"] = x.sizes();
  auto result = max_device_dispatch_forward(x, index, N);
  auto out = std::get<0>(result);
  auto arg_out = std::get<1>(result);
  ctx->save_for_backward({index, arg_out});
  ctx->mark_non_differentiable({arg_out});
  return out;
}

std::vector<torch::Tensor>
SegmentMax::backward(AutogradContext *ctx,
                     std::vector<torch::Tensor> grad_outs) {
  auto grad_out = grad_outs[0];
  auto saved = ctx->get_saved_variables();
  auto index = saved[0];
  auto arg_out = saved[1];
  auto x_shape = list2vec(ctx->saved_data["x_shape"].toIntList());

  auto grad_in = torch::zeros(x_shape, grad_out.options());
  auto grad_out_shape = grad_out.sizes().vec();
  grad_out_shape[0] = 1;
  auto zero_tensor = torch::zeros(grad_out_shape, grad_out.options());

  auto extended_grad_out =  torch::cat({grad_out, zero_tensor}, 0);

  for (int64_t i = 0; i < grad_in.size(0); ++i) {
      for (int64_t j = 0; j < grad_in.size(1); ++j) {
        auto index = arg_out[i][j].item<int64_t>();
        grad_in[i][j] = extended_grad_out[index][j].item<float>();
      }
  }

  return {grad_in, torch::Tensor(), torch::Tensor()};
}
