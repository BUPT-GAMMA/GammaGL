#include <assert.h>
#include <iostream>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

#include "cpu/spmm_sum_cpu.h"
#ifdef COMPILE_WITH_CUDA
#include "cuda/spmm_sum_cuda.h"
#endif

using torch::autograd::AutogradContext;
using tenosr_list = std::vector<torch::Tensor>;

torch::Tensor device_dispatch_forward(torch::Tensor &index,
                                      torch::Tensor &weight, torch::Tensor &x) {
  if (x.is_cuda() && index.is_cuda() && weight.is_cuda()) {
#ifdef COMPILE_WITH_CUDA
    return spmm_sum_cuda_forward(index, weight, x);
#else
    AT_ERROR("Compiled with CUDA support while tensor is on GPU!");
#endif
  } else if (x.is_cpu() && index.is_cpu() && weight.is_cpu()) {
    return spmm_sum_cpu_forward(index, weight, x);
  } else {
    AT_ERROR("Tensor device inconsistent error.");
  }
}

torch::Tensor device_dispatch_backward(torch::Tensor &index,
                                       torch::Tensor &weight,
                                       torch::Tensor &grad) {
  if (grad.is_cuda() && index.is_cuda() && weight.is_cuda()) {
#ifdef COMPILE_WITH_CUDA
    return spmm_sum_cuda_backward(index, weight, grad);
#else
    AT_ERROR("Compiled with CUDA support while tensor is on GPU!");
#endif
  } else if (grad.is_cpu() && index.is_cpu() && weight.is_cpu()) {
    return spmm_sum_cpu_backward(index, weight, grad);
  } else {
    AT_ERROR("Tensor device inconsistent error.");
  }
}

// Treat `index` and `weight` as an coo sparse matrix.
// Only coo sparse dense matrix multiplication.
// TODO: 1. support SpMMMax, SpMMMean, etc.
//       2. generalized operators to support more data
//          structures, such as csr, csc, etc.
class SpMMSum : public torch::autograd::Function<SpMMSum> {
public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor index,
                               torch::Tensor weight, torch::Tensor x) {
    ctx->save_for_backward({index, weight, x});
    ctx->mark_non_differentiable({index, weight});
    torch::Tensor out = device_dispatch_forward(index, weight, x);
    return out;
  }

  static std::vector<torch::Tensor>
  backward(AutogradContext *ctx, std::vector<torch::Tensor> grad_outs) {
    auto saved = ctx->get_saved_variables();
    auto index = saved[0], weight = saved[1], x = saved[2];
    torch::Tensor grad_x =
        device_dispatch_backward(index, weight, grad_outs[0]);
    return {torch::Tensor(), torch::Tensor(), grad_x};
  }
};

torch::Tensor spmm_sum(torch::Tensor index, torch::Tensor weight,
                       torch::Tensor x) {
  auto result = SpMMSum::apply(index, weight, x);
  return result;
}

// TORCH_LIBRARY BUG: dynamic module does not define module export function.
// Use PYBIND11_MODULE
PYBIND11_MODULE(torch_gspmm, m) {
  m.def("spmm_sum", spmm_sum);
  // m.def("spmm_max", spmm_max);
  // ...
}
