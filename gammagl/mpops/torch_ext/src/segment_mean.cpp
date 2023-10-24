#include "../include/segment_mean.h"
#include <assert.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <vector>
#include "../cpu/segment_mean_cpu.h"
#include "../cuda/segment_mean_cuda.h"
#include "../include/utils.h"


using torch::autograd::AutogradContext;

inline std::tuple<torch::Tensor, torch::Tensor> mean_device_dispatch_forward(torch::Tensor& x,
                                                          torch::Tensor& index,
                                                          int64_t& N) {
  if (x.is_cuda() && index.is_cuda()){
    return segment_mean_cuda_forward(x, index, N);
  } else if (x.is_cpu() && index.is_cpu()) {
    return segment_mean_cpu_forward(x, index, N);
  } else {
    AT_ERROR("Tensor device inconsistent error.");
  }
}

torch::Tensor SegmentMean::forward(AutogradContext* ctx,
                               torch::Tensor x,
                               torch::Tensor index,
                               int64_t N) {
    ctx->saved_data["x_shape"] = x.sizes();
    auto result = mean_device_dispatch_forward(x, index, N);
    auto out = std::get<0>(result);

  //   std::cout << "segment_mean.cpp**************" << std::endl;
  // std::cout << out << std::endl;
  // std::cout << "segment_mean.cpp**************" << std::endl;

    auto arg_out = std::get<1>(result);
    ctx->save_for_backward({index, arg_out});
    ctx->mark_non_differentiable({arg_out});
    return out;
}

std::vector<torch::Tensor> SegmentMean::backward(
      AutogradContext* ctx,
      std::vector<torch::Tensor> grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto index = saved[0];
    auto arg_out = saved[1];
    auto x_shape = list2vec(ctx->saved_data["x_shape"].toIntList());
    x_shape[0] += 1;
    auto grad_in = torch::zeros(x_shape, grad_out.options());
    grad_in.scatter_(0, arg_out, grad_out);
    grad_in = grad_in.narrow(0, 0, x_shape[0] - 1);
    return {grad_in, torch::Tensor(), torch::Tensor()};
}
