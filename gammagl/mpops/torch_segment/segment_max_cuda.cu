#include "segment_max_cuda.h"
#include <iostream>
#include <vector>
#include <cuda.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

static inline __device__ void atomic_max_float(float *addr, float value)
{
  float old = *addr, assumed;
  if (old >= value)
    return;
  do
  {
    assumed = old;
    if (assumed > value)
      break;
    old = atomicCAS((unsigned int *)addr, __float_as_int(assumed),
                    __float_as_int(value));

  } while (old != assumed);
}

// template <typename scalar_t>
// inline __device__ void atomic_max(scalar_t *addr, scalar_t val) {
//   int *address_as_i = (int *)address;
//   int old = *address_as_i;
//   int assumed;
//   do{
//     assumed = old;
//     old = atomicCAS(address_as_i, assumed,
//                     __float_as_int(max(val, __int_as_float(assumed))));
//   } while (assumed != old);
// }

template <typename scalar_t>
__global__ void scatter_kernel(const scalar_t *src_data,
                               at::TensorAccessor<int64_t, 1> &index_accessor,
                               scalar_t *out_data, int E, int K, int N, int numel)
{
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int e = (thread_idx / K) % E;
  int k = thread_idx % K;

  if (thread_idx < numel)
  {
    // TODO: support more data type
    atomic_max_float(out_data + index_accessor[e] * K + k,
                     src_data[thread_idx]);
  }
}

template <typename scalar_t>
__global__ void
scatter_arg_kernel(const scalar_t *src_data,
                   at::TensorAccessor<int64_t, 1> &index_accessor,
                   const scalar_t *out_data, int64_t *arg_out_data, int E,
                   int K, int N, int numel)
{

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int e = (thread_idx / K) % E;
  int k = thread_idx % K;

  if (thread_idx < numel)
  {
    int64_t idx = index_accessor[e];
    if (src_data[thread_idx] == out_data[idx * K + k])
    {
      arg_out_data[idx * K + k] = e;
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor>
segment_cuda(torch::Tensor src, torch::Tensor index, int64_t N)
{
  // check inputs
  cudaSetDevice(src.get_device());
  src = src.contiguous();

  auto sizes = src.sizes().vec();
  sizes[0] = N > *index.max().data_ptr<int64_t>() ? N : *index.max().data_ptr<int64_t>();

  torch::Tensor out = torch::empty(sizes, src.options());
  torch::Tensor arg_out = torch::full_like(out, 0, index.options());
  int64_t *arg_out_data = arg_out.data_ptr<int64_t>();

  if (src.numel() == 0)
  {
    out.fill_(0);
    return std::make_tuple(out, arg_out);
  }

  out.fill_(std::numeric_limits<int64_t>::lowest());
  auto E = src.size(0);
  auto K = src.size(1);
  auto index_accessor = index.accessor<int64_t, 1>();
  auto arg_out_accessor = arg_out.accessor<int64_t, 2>();
  auto stream = at::cuda::getCurrentCUDAStream();

  // AT_DISPATCH_ALL_TYPES(src.scalar_type(), "__ops_name",  [&] {
  using scalar_t = float;
  auto src_data = src.data_ptr<scalar_t>();
  auto out_data = out.data_ptr<scalar_t>();

  scatter_kernel<scalar_t>
      <<<BLOCKS(src.numel()), THREADS, 0, stream>>>(
          src_data, index_accessor, out_data, E, K, N, src.numel());

  out.masked_fill_(out == std::numeric_limits<int64_t>::lowest(), (scalar_t)0);

  scatter_arg_kernel<scalar_t>
      <<<BLOCKS(src.numel()), THREADS, 0, stream>>>(
          src_data, index_accessor, out_data, arg_out_data, E, K, N,
          src.numel());
  // });

  return std::make_tuple(out, arg_out);
}

inline std::vector<int64_t> list2vec(const c10::List<int64_t> list)
{
  std::vector<int64_t> result;
  result.reserve(list.size());
  for (size_t i = 0; i < list.size(); i++)
    result.push_back(list[i]);
  return result;
}

class SegmentMaxCuda : public torch::autograd::Function<SegmentMaxCuda>
{
public:
  static torch::Tensor forward(AutogradContext *ctx, Variable src,
                               Variable index, int64_t N)
  {
    ctx->saved_data["src_shape"] = src.sizes();
    auto result = segment_cuda(src, index, N);
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result);
    ctx->save_for_backward({index, arg_out});
    ctx->mark_non_differentiable({arg_out});
    return out;
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs)
  {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto index = saved[0];
    auto arg_out = saved[1];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    src_shape[0] += 1;
    auto grad_in = torch::zeros(src_shape, grad_out.options());
    grad_in.scatter_(0, arg_out, grad_out);
    grad_in = grad_in.narrow(0, 0, src_shape[0] - 1);
    return {grad_in, Variable(), Variable()};
  }
};

torch::Tensor segment_max_cuda(torch::Tensor src, torch::Tensor index, int64_t N)
{
  auto result = SegmentMaxCuda::apply(src, index, N);
  return result;
}

// TORCH_LIBRARY(mp, m)
// {
//   m.def("segment_max_cuda", segment_max_cuda);
// }