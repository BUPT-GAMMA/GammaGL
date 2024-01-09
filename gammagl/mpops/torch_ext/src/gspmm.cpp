#include "../include/gspmm.h"
#include <assert.h>
#include <iostream>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

#include "../cpu/spmm_sum_cpu.h"
#include "../cpu/spmm_mean_cpu.h"
#include "../cpu/spmm_max_cpu.h"
#ifdef COMPILE_WITH_CUDA
#include "../cuda/spmm_sum_cuda.h"
#endif

// using torch::autograd::AutogradContext;
// using tenosr_list = std::vector<torch::Tensor>;

enum SpMMOpType { SUM, MEAN, MAX };

torch::Tensor device_dispatch_forward(SpMMOpType op_type,
                                      torch::Tensor &index,
                                      torch::Tensor &weight, 
                                      torch::Tensor &x) {
    // CUDA
    if (x.is_cuda() && index.is_cuda() && weight.is_cuda()) {
    #ifdef COMPILE_WITH_CUDA
        switch (op_type) {
            case SUM:
                return spmm_sum_cuda_forward(index, weight, x);
            case MEAN:
                return spmm_sum_cuda_forward(index, weight, x); 
            case MAX:
                return spmm_sum_cuda_forward(index, weight, x); 
        }
    #else
        AT_ERROR("The program is not compiled with CUDA support, but tensors are located on GPU. Please recompile with CUDA support or move tensors to CPU.");
    #endif
    }
    // CPU
    else if (x.is_cpu() && index.is_cpu() && weight.is_cpu()) {
        switch (op_type) {
            case SUM:
                return spmm_sum_cpu_forward(index, weight, x);
            case MEAN:
                return spmm_mean_cpu_forward(index, weight, x);
            case MAX:
                return spmm_max_cpu_forward(index, weight, x); 
        }
    } else {
        AT_ERROR("Tensor device inconsistent error.");
    }
}


torch::Tensor device_dispatch_backward(SpMMOpType op_type,
                                       torch::Tensor &index,
                                       torch::Tensor &weight,
                                       torch::Tensor &grad) {
  // CUDA
  if (grad.is_cuda() && index.is_cuda() && weight.is_cuda()) {
#ifdef COMPILE_WITH_CUDA
    switch (op_type) {
      case SUM:
        return spmm_sum_cuda_backward(index, weight, grad);
      case MEAN:
        return spmm_sum_cuda_backward(index, weight, grad);
      case MAX:
        return spmm_sum_cuda_backward(index, weight, grad);
    }
#else
    AT_ERROR("The program is not compiled with CUDA support, but tensors are located on GPU. Please recompile with CUDA support or move tensors to CPU.");
#endif
  }
  // CPU
  else if (grad.is_cpu() && index.is_cpu() && weight.is_cpu()) {
    switch (op_type) {
      case SUM:
        return spmm_sum_cpu_backward(index, weight, grad);
      case MEAN:
        return spmm_mean_cpu_backward(index, weight, grad);
      case MAX:
        return spmm_max_cpu_backward(index, weight, grad); 
    }
  } else {
    AT_ERROR("Tensor device inconsistent error.");
  }
}

// Treat `index` and `weight` as an coo sparse matrix.
// Only coo sparse dense matrix multiplication.
// TODO: 1. support SpMMMax, SpMMMean, etc.
//       2. generalized operators to support more data
//          structures, such as csr, csc, etc.

torch::Tensor SpMMSum::forward(torch::autograd::AutogradContext *ctx, torch::Tensor index,
                               torch::Tensor weight, torch::Tensor x) {
    ctx->save_for_backward({index, weight, x});
    ctx->mark_non_differentiable({index, weight});
    torch::Tensor out = device_dispatch_forward(SUM, index, weight, x);
    return out;
  }

std::vector<torch::Tensor> SpMMSum::backward(torch::autograd::AutogradContext *ctx, std::vector<torch::Tensor> grad_outs) {
    auto saved = ctx->get_saved_variables();
    auto index = saved[0], weight = saved[1], x = saved[2];
    torch::Tensor grad_x = device_dispatch_backward(SUM, index, weight, grad_outs[0]);
    return {torch::Tensor(), torch::Tensor(), grad_x};
  }
  
torch::Tensor SpMMMean::forward(torch::autograd::AutogradContext *ctx, torch::Tensor index,
                               torch::Tensor weight, torch::Tensor x) {
    ctx->save_for_backward({index, weight, x});
    ctx->mark_non_differentiable({index, weight});
    torch::Tensor out = device_dispatch_forward(MEAN, index, weight, x);
    return out;
  }

std::vector<torch::Tensor> SpMMMean::backward(torch::autograd::AutogradContext *ctx, std::vector<torch::Tensor> grad_outs) {
    auto saved = ctx->get_saved_variables();
    auto index = saved[0], weight = saved[1], x = saved[2];
    torch::Tensor grad_x = device_dispatch_backward(MEAN, index, weight, grad_outs[0]);
    return {torch::Tensor(), torch::Tensor(), grad_x};
  }

torch::Tensor SpMMMax::forward(torch::autograd::AutogradContext *ctx, torch::Tensor index,
                               torch::Tensor weight, torch::Tensor x) {
    ctx->save_for_backward({index, weight, x});
    ctx->mark_non_differentiable({index, weight});
    torch::Tensor out = device_dispatch_forward(MAX, index, weight, x);
    return out;
  }

std::vector<torch::Tensor> SpMMMax::backward(torch::autograd::AutogradContext *ctx, std::vector<torch::Tensor> grad_outs) {
    auto saved = ctx->get_saved_variables();
    auto index = saved[0], weight = saved[1], x = saved[2];
    torch::Tensor grad_x = device_dispatch_backward(MAX, index, weight, grad_outs[0]);
    return {torch::Tensor(), torch::Tensor(), grad_x};
  }