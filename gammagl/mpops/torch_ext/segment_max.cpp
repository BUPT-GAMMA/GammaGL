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

std::tuple<torch::Tensor, torch::Tensor> device_dispatch_forward(torch::Tensor& x,
                                                          torch::Tensor& index,
                                                          int64_t& N) {
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

class SegmentMax : public torch::autograd::Function<SegmentMax> {
 public:
  static torch::Tensor forward(AutogradContext* ctx,
                               torch::Tensor x,
                               torch::Tensor index,
                               int64_t N) {
    ctx->saved_data["x_shape"] = x.sizes();
    auto result = device_dispatch_forward(x, index, N);
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
    auto x_shape = list2vec(ctx->saved_data["x_shape"].toIntList());
    x_shape[0] += 1;
    auto grad_in = torch::zeros(x_shape, grad_out.options());
    grad_in.scatter_(0, arg_out, grad_out);
    grad_in = grad_in.narrow(0, 0, x_shape[0] - 1);
    return {grad_in, torch::Tensor(), torch::Tensor()};
  }
};

torch::Tensor segment_max(torch::Tensor x, torch::Tensor index, int64_t N) {
  auto result = SegmentMax::apply(x, index, N);
  return result;
}

PYBIND11_MODULE(torch_segment, m) {
  m.def("segment_max", segment_max);
  // m.def("segment_sum", segment_sum);
}