#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <cuda.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <vector>

#include "segment_sum_cuda.h"

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void segment_sum_cuda_forward_kernel(
    const scalar_t *x_data, const int64_t *index_data, scalar_t *out_data,
    int64_t E, int64_t K, int64_t N, int64_t numel) {
  int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t e = (thread_idx / K) % E;
  int64_t k = thread_idx % K;
  if (thread_idx < numel) {
    // TODO: support more data type
    int64_t idx = index_data[e];
    atomicAdd(out_data + idx * K + k, x_data[thread_idx]);
  }
}

torch::Tensor segment_sum_cuda_forward(
    torch::Tensor x, torch::Tensor index, int64_t N) {
  // check inputs
  TORCH_CHECK(x.device().is_cuda(), "x must be CUDA tensor");
  TORCH_CHECK(index.device().is_cuda(), "index must be CUDA tensor");
  TORCH_CHECK_INDEX(
      index.dim() == 1, "index dimension should be 1, but got ", index.dim());
  TORCH_CHECK_INDEX(
      x.size(0) == index.size(0),
      "fisrt dimension of x and index should be same");
  // only support float Tensor
  // TORCH_CHECK_TYPE(
  //     x.scalar_type() == c10::ScalarType::Float, "x should be float Tensor")
  cudaSetDevice(x.get_device());
  x = x.contiguous();
  index = index.contiguous();

  auto sizes = x.sizes().vec();
  sizes[0] = N > *index.max().cpu().data_ptr<int64_t>()
                 ? N
                 : *index.max().cpu().data_ptr<int64_t>();
  torch::Tensor out = torch::empty(sizes, x.options());
  // TORCH_CHECK(out.device().is_cuda(), "out must be CUDA tensor");
  torch::Tensor arg_out = torch::full_like(out, 0, index.options());
  int64_t *arg_out_data = arg_out.data_ptr<int64_t>();
  if (x.numel() == 0) {
    out.fill_(0);
    return out;
  }

  // out.fill_(std::numeric_limits<int64_t>::lowest());
  out.fill_(0);
  auto E = x.size(0);
  auto K = x.numel() / x.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (x.dtype() == torch::kInt8 || x.dtype() == torch::kInt16 ||
      x.dtype() == torch::kInt32 || x.dtype() == torch::kInt64) {
    auto type = x.dtype();
    using scalar_t = int;
    if (x.dtype() == torch::kInt8 || x.dtype() == torch::kInt16 ||
        x.dtype() == torch::kInt64) {
      x = x.to(torch::kInt32);
      out = out.to(torch::kInt32);
    }
    auto x_data = x.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    auto index_data = index.data_ptr<int64_t>();

    segment_sum_cuda_forward_kernel<scalar_t>
        <<<BLOCKS(x.numel()), THREADS, 0, stream>>>(
            x_data, index_data, out_data, E, K, N, x.numel());

    out = out.to(type);
  } else if (x.dtype() == torch::kFloat16 || x.dtype() == torch::kFloat32) {
    auto type = x.dtype();
    using scalar_t = float;
    if (x.dtype() == torch::kFloat16) {
      x = x.to(torch::kFloat32);
      out = out.to(torch::kFloat32);
    }

    auto x_data = x.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    auto index_data = index.data_ptr<int64_t>();

    segment_sum_cuda_forward_kernel<scalar_t>
        <<<BLOCKS(x.numel()), THREADS, 0, stream>>>(
            x_data, index_data, out_data, E, K, N, x.numel());

    out = out.to(type);
  } else if (x.dtype() == torch::kFloat64) {
    using scalar_t = double;
    auto x_data = x.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    auto index_data = index.data_ptr<int64_t>();

    segment_sum_cuda_forward_kernel<scalar_t>
        <<<BLOCKS(x.numel()), THREADS, 0, stream>>>(
            x_data, index_data, out_data, E, K, N, x.numel());
  }

  return out;
}
