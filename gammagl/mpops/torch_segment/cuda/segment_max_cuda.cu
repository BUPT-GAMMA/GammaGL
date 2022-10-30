#include "segment_max_cuda.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

inline __device__ void atomic_max_float(float *addr, float value) {
  int *addr_as_i = (int *)addr;
  int old = *addr_as_i;
  int assumed;
  do{
    assumed = old;
    old = atomicCAS(addr_as_i, assumed,
                    __float_as_int(max(value, __int_as_float(assumed))));
  } while (assumed != old);
}

template <typename scalar_t>
__global__ void segment_max_cuda_forward_kernel(const scalar_t *src_data, const int64_t *index_data,
                               scalar_t *out_data, int E, int K, int N, int numel) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int e = (thread_idx / K) % E;
  int k = thread_idx % K;
  if (thread_idx < numel)  {
    // TODO: support more data type
    int idx = index_data[e];
    atomic_max_float(out_data + idx * K + k,
                     src_data[thread_idx]);
  }
}

// TODO: fuse segment & arg_segment to one kernel function.
template <typename scalar_t>
__global__ void
arg_segment_max_cuda_forward_kernel(const scalar_t *src_data, const int64_t *index_data,
                   scalar_t *out_data, int64_t *arg_out_data, int E,
                   int K, int N, int numel) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int e = (thread_idx / K) % E;
  int k = thread_idx % K;

  if (thread_idx < numel) {
    int idx = index_data[e];
    if (src_data[thread_idx] == out_data[idx * K + k]) {
      arg_out_data[idx * K + k] = e;
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor>
segment_max_cuda_forward(torch::Tensor src, torch::Tensor index, int64_t N) {
  // check inputs
  TORCH_CHECK(src.device().is_cuda(), "src must be CUDA tensor");
  TORCH_CHECK(index.device().is_cuda(), "index must be CUDA tensor");
  TORCH_CHECK_INDEX(src.dim() == 2, "src dimension should be 2, but got ", src.dim());
  TORCH_CHECK_INDEX(index.dim() == 1, "index dimension should be 1, but got ", index.dim());
  TORCH_CHECK_INDEX(src.size(0) == index.size(0), "fisrt dimension of src and index should be same");
  // only support float Tensor
  TORCH_CHECK_TYPE(src.scalar_type() == c10::ScalarType::Float, "src should be float Tensor")
  cudaSetDevice(src.get_device());
  src = src.contiguous();

  auto sizes = src.sizes().vec();
  sizes[0] = N > *index.max().cpu().data_ptr<int64_t>()
                 ? N
                 : *index.max().cpu().data_ptr<int64_t>();
  torch::Tensor out = torch::empty(sizes, src.options());
  // TORCH_CHECK(out.device().is_cuda(), "out must be CUDA tensor");
  torch::Tensor arg_out = torch::full_like(out, 0, index.options());
  int64_t *arg_out_data = arg_out.data_ptr<int64_t>();
  if (src.numel() == 0) {
    out.fill_(0);
    return std::make_tuple(out, arg_out);
  }

  out.fill_(std::numeric_limits<int64_t>::lowest());
  auto E = src.size(0);
  auto K = src.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();

  // AT_DISPATCH_ALL_TYPES(src.scalar_type(), "__ops_name",  [&] {
  using scalar_t = float; // temporary usage, delete later
  auto src_data = src.data_ptr<scalar_t>();
  auto out_data = out.data_ptr<scalar_t>();
  auto index_data = index.data_ptr<int64_t>();

  segment_max_cuda_forward_kernel<scalar_t>
      <<<BLOCKS(src.numel()), THREADS, 0, stream>>>(
          src_data, index_data, out_data, E, K, N, src.numel());

  out.masked_fill_(out == std::numeric_limits<int64_t>::lowest(), (scalar_t)0);

  arg_segment_max_cuda_forward_kernel<scalar_t>
      <<<BLOCKS(src.numel()), THREADS, 0, stream>>>(
          src_data, index_data, out_data, arg_out_data, E, K, N,
          src.numel());
  // });

  return std::make_tuple(out, arg_out);
}
