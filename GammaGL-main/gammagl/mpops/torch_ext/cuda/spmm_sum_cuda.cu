#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <cuda.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <vector>

#include "spmm_sum_cuda.h"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void spmm_sum_cuda_forward_kernel(
    const int64_t *index_data, const scalar_t *weight_data,
    const scalar_t *x_data, scalar_t *out_data, int64_t E, int64_t K,
    int64_t numel) {
  int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx < numel) {
    int64_t e = (thread_idx / K) % E;
    int64_t k = thread_idx % K;
    int64_t col = index_data[e];
    int64_t row = index_data[e + E];  // or e + 1;
    scalar_t val = weight_data[e] * x_data[col * K + k];
    atomicAdd(out_data + row * K + k, val);
  }
}

template <typename scalar_t>
__global__ void spmm_sum_cuda_backward_kernel(
    const int64_t *index_data, const scalar_t *weight_data,
    const scalar_t *grad_data, scalar_t *out_data, int64_t E, int64_t K,
    int64_t numel) {
  int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx < numel) {
    int64_t e = (thread_idx / K) % E;
    int64_t k = thread_idx % K;
    int64_t col = index_data[e];
    int64_t row = index_data[e + E];  // or e + 1;
    scalar_t val = weight_data[e] * grad_data[row * K + k];
    atomicAdd(out_data + col * K + k, val);
  }
}

torch::Tensor spmm_sum_cuda_forward(
    torch::Tensor &index, torch::Tensor &weight, torch::Tensor &x) {
  x = x.contiguous();
  torch::Tensor out = torch::zeros_like(x, x.options());
  auto E = index.size(1);
  auto K = x.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto index_data = index.data_ptr<int64_t>();
  // AT_DISPATCH_ALL_TYPES(x.scalar_type(), "__ops_name", [&] {
  using scalar_t = float;
  auto x_data = x.data_ptr<scalar_t>();
  auto out_data = out.data_ptr<scalar_t>();
  auto weight_data = weight.data_ptr<scalar_t>();
  spmm_sum_cuda_forward_kernel<scalar_t>
      <<<BLOCKS(x.numel()), THREADS, 0, stream>>>(
          index_data, weight_data, x_data, out_data, E, K, x.numel());
  // });
  return out;
}

torch::Tensor spmm_sum_cuda_backward(
    torch::Tensor &index, torch::Tensor &weight, torch::Tensor &grad) {
  grad = grad.contiguous();
  torch::Tensor out = torch::zeros_like(grad, grad.options());
  auto E = index.size(1);
  auto K = grad.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto index_data = index.data_ptr<int64_t>();
  // AT_DISPATCH_ALL_TYPES(x.scalar_type(), "__ops_name", [&] {
  using scalar_t = float;
  auto weight_data = weight.data_ptr<scalar_t>();
  auto grad_data = grad.data_ptr<scalar_t>();
  auto out_data = out.data_ptr<scalar_t>();
  spmm_sum_cuda_backward_kernel<scalar_t>
      <<<BLOCKS(grad.numel()), THREADS, 0, stream>>>(
          index_data, weight_data, grad_data, out_data, E, K, grad.numel());
  // });
  return out;
}
