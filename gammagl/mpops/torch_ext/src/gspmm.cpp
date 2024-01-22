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

// Treat `index` and `weight` as an coo sparse matrix.
// Only coo sparse dense matrix multiplication.
// TODO: 1. support SpMMMax, SpMMMean, etc.
//       2. generalized operators to support more data
//          structures, such as csr, csc, etc.

torch::Tensor SpMMSum::forward(torch::autograd::AutogradContext *ctx, torch::Tensor index,
                               torch::Tensor weight, torch::Tensor x) {
    ctx->save_for_backward({index, weight, x});
    ctx->mark_non_differentiable({index, weight});
    torch::Tensor out;
    // CUDA
    if (x.is_cuda() && index.is_cuda() && weight.is_cuda()) {
    #ifdef COMPILE_WITH_CUDA
        out = spmm_sum_cuda_forward(index, weight, x);
    #else
        AT_ERROR("The program is not compiled with CUDA support, but tensors are located on GPU. Please recompile with CUDA support or move tensors to CPU.");
    #endif
    }
    // CPU
    else if (x.is_cpu() && index.is_cpu() && weight.is_cpu()) {
        out = spmm_sum_cpu_forward(index, weight, x);
    } else {
        AT_ERROR("Tensor device inconsistent error.");
    }

    return out;
  }

std::vector<torch::Tensor> SpMMSum::backward(torch::autograd::AutogradContext *ctx, std::vector<torch::Tensor> grad_outs) {
    auto saved = ctx->get_saved_variables();
    auto index = saved[0], weight = saved[1], x = saved[2];
    auto grad = grad_outs[0];
    torch::Tensor grad_x;

    // CUDA
    if (grad.is_cuda() && index.is_cuda() && weight.is_cuda()) {
  #ifdef COMPILE_WITH_CUDA
      grad_x = spmm_sum_cuda_backward(index, weight, grad);
  #else
      AT_ERROR("The program is not compiled with CUDA support, but tensors are located on GPU. Please recompile with CUDA support or move tensors to CPU.");
  #endif
    }
    // CPU
    else if (grad.is_cpu() && index.is_cpu() && weight.is_cpu()) {
      grad_x = spmm_sum_cpu_backward(index, weight, grad);
    } else {
      AT_ERROR("Tensor device inconsistent error.");
    }

    return {torch::Tensor(), torch::Tensor(), grad_x};
}
  
torch::Tensor SpMMMean::forward(torch::autograd::AutogradContext *ctx, torch::Tensor index,
                               torch::Tensor weight, torch::Tensor x) {
    ctx->mark_non_differentiable({index, weight});
    std::tuple<torch::Tensor, torch::Tensor> result;

    // CUDA
    if (x.is_cuda() && index.is_cuda() && weight.is_cuda()) {
      AT_ERROR("The program is not support CUDA !");
  // #ifdef COMPILE_WITH_CUDA
  //     // grad_x = spmm_sum_cuda_backward(index, weight, grad, max_indices);
  //     grad_x = spmm_sum_cuda_backward(index, weight, grad);
  // #else
  //     AT_ERROR("The program is not compiled with CUDA support, but tensors are located on GPU. Please recompile with CUDA support or move tensors to CPU.");
  // #endif
    }
    // CPU
    else if (x.is_cpu() && index.is_cpu() && weight.is_cpu()) {
        result = spmm_mean_cpu_forward(index, weight, x);
    } else {
        AT_ERROR("Tensor device inconsistent error.");
    }
    
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result);
    ctx->save_for_backward({index, weight, x, arg_out});
    return out;
  }

std::vector<torch::Tensor> SpMMMean::backward(torch::autograd::AutogradContext *ctx, std::vector<torch::Tensor> grad_outs) {
    auto saved = ctx->get_saved_variables();
    auto index = saved[0], weight = saved[1], x = saved[2], messages_count = saved[3];
    auto grad = grad_outs[0];
    torch::Tensor grad_x;

    // CUDA
    if (grad.is_cuda() && index.is_cuda() && weight.is_cuda()) {
      AT_ERROR("The program is not support CUDA !");
    // #ifdef COMPILE_WITH_CUDA
    //     result = spmm_sum_cuda_forward(index, weight, x); 
    // #else
    //     AT_ERROR("The program is not compiled with CUDA support, but tensors are located on GPU. Please recompile with CUDA support or move tensors to CPU.");
    // #endif
    }
    // CPU
    else if (grad.is_cpu() && index.is_cpu() && weight.is_cpu()) {
      grad_x = spmm_mean_cpu_backward(index, weight, grad, messages_count);
    } else {
      AT_ERROR("Tensor device inconsistent error.");
    }

    return {torch::Tensor(), torch::Tensor(), grad_x};
}

torch::Tensor SpMMMax::forward(torch::autograd::AutogradContext *ctx, torch::Tensor index,
                               torch::Tensor weight, torch::Tensor x) {
    ctx->mark_non_differentiable({index, weight});
    std::tuple<torch::Tensor, torch::Tensor> result;

    // CUDA
    if (x.is_cuda() && index.is_cuda() && weight.is_cuda()) {
      AT_ERROR("The program is not support CUDA !");
    // #ifdef COMPILE_WITH_CUDA
    //     result = spmm_sum_cuda_forward(index, weight, x); 
    // #else
    //     AT_ERROR("The program is not compiled with CUDA support, but tensors are located on GPU. Please recompile with CUDA support or move tensors to CPU.");
    // #endif
    }
    // CPU
    else if (x.is_cpu() && index.is_cpu() && weight.is_cpu()) {
        result = spmm_max_cpu_forward(index, weight, x); 
    } else {
        AT_ERROR("Tensor device inconsistent error.");
    }

    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result);
    ctx->save_for_backward({index, weight, x, arg_out});
    return out;
}

std::vector<torch::Tensor> SpMMMax::backward(torch::autograd::AutogradContext *ctx, std::vector<torch::Tensor> grad_outs) {
    auto saved = ctx->get_saved_variables();
    auto index = saved[0], weight = saved[1], x = saved[2], max_indices = saved[3];
    auto grad = grad_outs[0];
    torch::Tensor grad_x;

      // CUDA
    if (grad.is_cuda() && index.is_cuda() && weight.is_cuda()) {
      AT_ERROR("The program is not support CUDA !");
  // #ifdef COMPILE_WITH_CUDA
  //     // grad_x = spmm_sum_cuda_backward(index, weight, grad, max_indices);
  //     grad_x = spmm_sum_cuda_backward(index, weight, grad);
  // #else
  //     AT_ERROR("The program is not compiled with CUDA support, but tensors are located on GPU. Please recompile with CUDA support or move tensors to CPU.");
  // #endif
    }
    // CPU
    else if (grad.is_cpu() && index.is_cpu() && weight.is_cpu()) {
      grad_x = spmm_max_cpu_backward(index, weight, grad, max_indices); 
    } else {
      AT_ERROR("Tensor device inconsistent error.");
    }

    return {torch::Tensor(), torch::Tensor(), grad_x};
}
