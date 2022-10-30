#include <assert.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <vector>
#include "cpu/segment_max_cpu.h"
#ifdef COMPILE_WITH_CUDA
#include "cuda/segment_max_cuda.h"
#endif

using torch::autograd::AutogradContext;

inline std::vector<int64_t> list2vec(const c10::List<int64_t> list) {
  std::vector<int64_t> result;
  result.reserve(list.size());
  for (size_t i = 0; i < list.size(); ++i)
    result.push_back(list[i]);
  return result;
}

std::tuple<torch::Tensor, torch::Tensor> device_dispatch_forward(torch::Tensor& src,
                                                          torch::Tensor& index,
                                                          int64_t& N) {
  if (src.is_cuda() && index.is_cuda()) {
#ifdef COMPILE_WITH_CUDA
    return segment_max_cuda_forward(src, index, N);
#else
    AT_ERROR("Compiled with CUDA support while tensor is on GPU!");
#endif
  } else if (src.is_cpu() && index.is_cpu()) {
    return segment_max_cpu_forward(src, index, N);
  } else {
    TORCH_CHECK(false, "Device type error.");
  }
}

class SegmentMax : public torch::autograd::Function<SegmentMax> {
 public:
  static torch::Tensor forward(AutogradContext* ctx,
                               torch::Tensor src,
                               torch::Tensor index,
                               int64_t N) {
    ctx->saved_data["src_shape"] = src.sizes();
    auto result = device_dispatch_forward(src, index, N);
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result);
    ctx->save_for_backward({index, arg_out});
    ctx->mark_non_differentiable({arg_out});
    return out;
  }

  static std::vector<torch::Tensor> backward(
      AutogradContext* ctx,
      std::vector<torch::Tensor> grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto index = saved[0];
    auto arg_out = saved[1];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    src_shape[0] += 1;
    auto grad_in = torch::zeros(src_shape, grad_out.options());
    grad_in.scatter_(0, arg_out, grad_out);
    grad_in = grad_in.narrow(0, 0, src_shape[0] - 1);
    return {grad_in, torch::Tensor(), torch::Tensor()};
  }
};

torch::Tensor segment_max(torch::Tensor src, torch::Tensor index, int64_t N) {
  auto result = SegmentMax::apply(src, index, N);
  return result;
}

TORCH_LIBRARY(torch_segment, m) {
  m.def("segment_max", segment_max);
}