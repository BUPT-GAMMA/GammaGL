#include <paddle/extension.h>
#include <vector>
#include "segment_sum_cuda.h"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS
#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

template <typename data_t>
__global__ void segment_sum_cuda_forward_kernel(const data_t *src_data, const int64_t *index_data,
                               data_t *out_data, int E, int K, int N, int numel) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int e = (thread_idx / K) % E;
  int k = thread_idx % K;
  printf(" thread_idx = %d \n", thread_idx);
  if (thread_idx < numel)  {
    int idx = index_data[e];
    printf("%f \n", *out_data);
    atomicAdd(out_data + idx * K + k,
              src_data[thread_idx]);
  }
}

std::vector<paddle::Tensor> segment_sum_cuda_forward(const paddle::Tensor& src,
                                                 const paddle::Tensor& index,
                                                 int64_t n) {
  CHECK_INPUT(src);
  CHECK_INPUT(index);
  std::vector<int64_t> sizes = {n, src.shape()[1]};
  auto out = paddle::full(sizes, 0., src.dtype(), src.place());
  if (src.numel() == 0) {
    return {out};
  }

  auto E = src.shape()[0];
  auto K = src.shape()[1];
  
  // PD_DISPATCH_FLOATING_TYPES(src.type(), "cpu_segment_kernel", ([&] {
  using data_t = float;

  segment_sum_cuda_forward_kernel<data_t> 
  <<<BLOCKS(src.numel()), THREADS, 0, src.stream()>>>(
            src.data<data_t>(),
            index.data<int64_t>(),
            out.data<data_t>(),
            E,
            K,
            n,
            src.numel());
  // }));

  // cudaError_t cudaStatus = cudaGetLastError();
  // if (cudaStatus != cudaSuccess) {
  //     fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
  // }

  return {out}; 
}
