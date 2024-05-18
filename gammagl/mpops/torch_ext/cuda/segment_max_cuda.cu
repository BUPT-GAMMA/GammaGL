#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <cuda.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "segment_max_cuda.h"

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

// template <typename scalar_t>
// __device__ void atomic_max_float(scalar_t *addr, scalar_t value) {
//   int *addr_as_i = (int *)addr;
//   int old = *addr_as_i;
//   int assumed;
//   do {
//     assumed = old;
//     old = atomicCAS(
//         addr_as_i, assumed,
//         __float_as_int(max(value, __int_as_float(assumed))));
//   } while (assumed != old);
// }

template <typename scalar_t>
__device__ void atomic_max(scalar_t* const address, const scalar_t value);

template <>
__device__ void atomic_max<int32_t>(int32_t* const address, const int32_t value) {
    atomicMax(address, value);
}

template <>
__device__ void atomic_max<float>(float* const address, const float value) {
    int* const address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
}

template <>
__device__ void atomic_max<double>(double* const address, const double value) {
    unsigned long long int* const address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(value, __longlong_as_double(assumed))));
    } while (assumed != old);
}

template <typename scalar_t>
__global__ void segment_max_cuda_forward_kernel(
    const scalar_t *x_data, const int64_t *index_data, scalar_t *out_data,
    int64_t E, int64_t K, int64_t N, int64_t numel) {
  int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t e = (thread_idx / K) % E;
  int64_t k = thread_idx % K;
  if (thread_idx < numel) {
    // TODO: support more data type
    int64_t idx = index_data[e];
    // atomic_max_float(out_data + idx * K + k, x_data[thread_idx]);
    atomic_max(out_data + idx * K + k, x_data[thread_idx]);
  }
}

// TODO: fuse segment & arg_segment to one kernel function.
template <typename scalar_t>
__global__ void arg_segment_max_cuda_forward_kernel(
    const scalar_t *x_data, const int64_t *index_data, scalar_t *out_data,
    int64_t *arg_out_data, int64_t E, int64_t K, int64_t N, int64_t numel,
    int64_t out_size) {
  int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t e = (thread_idx / K) % E;
  int64_t k = thread_idx % K;

  if (thread_idx < numel) {
    int64_t idx = index_data[e];
    if (x_data[thread_idx] == out_data[idx * K + k]) {
      arg_out_data[idx * K + k] = e;
      // arg_out_data[e * K + k] = idx;
      // for (auto pos = 0; pos < e && index_data[e] == index_data[pos]; pos++)
      // {
      //   arg_out_data[pos * K + k] = out_size;
      // }
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor> segment_max_cuda_forward(
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
  torch::Tensor arg_out = torch::full_like(out, out.size(0), index.options());
  int64_t *arg_out_data = arg_out.data_ptr<int64_t>();
  if (x.numel() == 0) {
    out.fill_(0);
    return std::make_tuple(out, arg_out);
  }

  // out.fill_(std::numeric_limits<int64_t>::lowest());
  auto E = x.size(0);
  auto K = x.numel() / x.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (x.dtype() == torch::kInt8 || x.dtype() == torch::kInt16 || x.dtype() == torch::kInt32 || x.dtype() == torch::kInt64) {
    if (x.dtype() == torch::kInt8){
      out.fill_(std::numeric_limits<int8_t>::lowest());
    } else if (x.dtype() == torch::kInt16){
      out.fill_(std::numeric_limits<int16_t>::lowest());
    } else if (x.dtype() == torch::kInt32){
      out.fill_(std::numeric_limits<int32_t>::lowest());
    } else if (x.dtype() == torch::kInt64){
      out.fill_(std::numeric_limits<int64_t>::lowest());
    }
    auto type = x.dtype();
    using scalar_t = int;
    if (x.dtype() == torch::kInt8 || x.dtype() == torch::kInt16 || x.dtype() == torch::kInt64) {
      x = x.to(torch::kInt32);
      out = out.to(torch::kInt32);
    }
    // out.fill_(std::numeric_limits<scalar_t>::lowest());
    auto x_data = x.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    auto index_data = index.data_ptr<int64_t>();

    segment_max_cuda_forward_kernel<scalar_t>
        <<<BLOCKS(x.numel()), THREADS, 0, stream>>>(
            x_data, index_data, out_data, E, K, N, x.numel());

    arg_segment_max_cuda_forward_kernel<scalar_t>
        <<<BLOCKS(x.numel()), THREADS, 0, stream>>>(
            x_data, index_data, out_data, arg_out_data, E, K, N, x.numel(),
            out.size(0));
    
    out = out.to(type);
    
  } else if (x.dtype() == torch::kFloat16 || x.dtype() == torch::kFloat32) {
    auto type = x.dtype();
    using scalar_t = float;
    if (x.dtype() == torch::kFloat16) {
      x = x.to(torch::kFloat32);
      out = out.to(torch::kFloat32);
      out.fill_(-65503.9);
    } else if (x.dtype() == torch::kFloat32) {
      out.fill_(std::numeric_limits<scalar_t>::lowest());
    }
    auto x_data = x.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    auto index_data = index.data_ptr<int64_t>();

    segment_max_cuda_forward_kernel<scalar_t>
        <<<BLOCKS(x.numel()), THREADS, 0, stream>>>(
            x_data, index_data, out_data, E, K, N, x.numel());

    arg_segment_max_cuda_forward_kernel<scalar_t>
        <<<BLOCKS(x.numel()), THREADS, 0, stream>>>(
            x_data, index_data, out_data, arg_out_data, E, K, N, x.numel(),
            out.size(0));
    
    out = out.to(type);
  } else if (x.dtype() == torch::kFloat64) {
    using scalar_t = double;
    out.fill_(std::numeric_limits<scalar_t>::lowest());
    auto x_data = x.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    auto index_data = index.data_ptr<int64_t>();

    segment_max_cuda_forward_kernel<scalar_t>
        <<<BLOCKS(x.numel()), THREADS, 0, stream>>>(
            x_data, index_data, out_data, E, K, N, x.numel());

    arg_segment_max_cuda_forward_kernel<scalar_t>
        <<<BLOCKS(x.numel()), THREADS, 0, stream>>>(
            x_data, index_data, out_data, arg_out_data, E, K, N, x.numel(),
            out.size(0));
  }

  return std::make_tuple(out, arg_out);
}
