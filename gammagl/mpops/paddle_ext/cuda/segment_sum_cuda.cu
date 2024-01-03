#include <paddle/extension.h>
#include <vector>
#include "segment_sum_cuda.h"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS
#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

template <typename data_t>
__global__ void segment_sum_cuda_forward_kernel(const data_t *x_data, const int64_t *index_data,
                               data_t *out_data, int64_t E, int64_t K, int64_t N, int64_t numel) {
  int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t e = (thread_idx / K) % E;
  int64_t k = thread_idx % K;
  if (thread_idx < numel)  {
    int64_t idx = index_data[e];
    atomicAdd(out_data + idx * K + k,
              x_data[thread_idx]);
  }
}

std::vector<paddle::Tensor> segment_sum_cuda_forward(const paddle::Tensor& x,
                                                 const paddle::Tensor& index,
                                                 int64_t n) {
  CHECK_INPUT(x);
  CHECK_INPUT(index);
  std::vector<int64_t> sizes = {n, x.shape()[1]};
  auto out = paddle::full(sizes, 0., x.dtype(), x.place());
  if (x.numel() == 0) {
    return {out};
  }

  auto E = x.shape()[0];
  auto K = x.shape()[1];
  
  // PD_DISPATCH_FLOATING_TYPES(x.type(), "cpu_segment_kernel", ([&] {
  using data_t = float;

  segment_sum_cuda_forward_kernel<data_t> 
  <<<BLOCKS(x.numel()), THREADS, 0, x.stream()>>>(
            x.data<data_t>(),
            index.data<int64_t>(),
            out.data<data_t>(),
            E,
            K,
            n,
            x.numel());
  // }));

  // cudaError_t cudaStatus = cudaGetLastError();
  // if (cudaStatus != cudaSuccess) {
  //     fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
  // }

  return {out}; 
}
