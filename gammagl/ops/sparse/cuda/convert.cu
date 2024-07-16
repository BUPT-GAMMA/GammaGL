#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../ticktock.h"
#include "convert.h"
// #include <helper_cuda.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>
// #include <__clang_cuda_runtime_wrapper.h>
#include <cstdint>
#include <iostream>

constexpr int BLOCK_SIZE = 128;

__global__ void kernel_set_to_one(int64_t *arr, int n) {
  int64_t it = blockDim.x * blockIdx.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  while (i < n) {
    arr[i] = 1;
    i += stride;
  }
}

void cuda_set_to_one(int64_t *arr, int n) {
  dim3 block(BLOCK_SIZE);
  dim3 grid(n + BLOCK_SIZE - 1 / BLOCK_SIZE);
  kernel_set_to_one<<<grid, block>>>(arr, n);
}

at::Tensor torch_cuda_set_to_one(at::Tensor &ts) {
  ts = ts.contiguous();
  int n = ts.numel();
  int64_t *ts_arr = (int64_t *)ts.data_ptr();
  cuda_set_to_one(ts_arr, n);
  return ts;
}

__global__ void kernel_ind2ptr(
    int64_t *ptr, int64_t *ind, int64_t N, int64_t num) {
  int64_t it = blockDim.x * blockIdx.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  while (i < num) {
    int64_t l = 0, r = N, mid;
    while (l < r) {
      mid = l + r >> 1;
      if (ind[mid] >= i) {
        r = mid;
      }
      if (ind[mid] < i) {
        l = mid + 1;
      }
    }
    ptr[i] = l;
    i += stride;
  }
}

void cuda_ind2ptr(int64_t *ptr, int64_t *ind, int64_t N, int64_t num) {
  dim3 block(BLOCK_SIZE);
  dim3 grid((num + BLOCK_SIZE - 1) / BLOCK_SIZE);
  kernel_ind2ptr<<<grid, block>>>(ptr, ind, N, num);
}

at::Tensor torch_cuda_ind2ptr(at::Tensor &ind, int64_t M) {
  int device = ind.get_device();
  cudaSetDevice(device);
  ind.contiguous();

  at::Tensor ptr =
      at::zeros({M + 1}, at::dtype(torch::kInt64).device(at::kCUDA, device));

  int64_t ind_len = ind.sizes()[0];

  int64_t *ind_arr = (int64_t *)ind.data_ptr();
  int64_t *ptr_arr = (int64_t *)ptr.data_ptr();

  cuda_ind2ptr(ptr_arr, ind_arr, ind_len, M + 1);
  return ptr;
}

__global__ void kernel_ptr2ind(
    int64_t *ind, int64_t *ptr, int64_t N, int64_t num) {
  int64_t it = blockDim.x * blockIdx.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  while (i < num) {
    int64_t l = 0, r = N - 1, mid;
    while (l < r) {
      mid = l + r + 1 >> 1;
      if (ptr[mid] > i) {
        r = mid - 1;
      }
      if (ptr[mid] <= i) {
        l = mid;
      }
    }
    ind[i] = l;
    i += stride;
  }
}

void cuda_ptr2ind(int64_t *ind, int64_t *ptr, int64_t N, int64_t num) {
  dim3 block(BLOCK_SIZE);
  dim3 grid((num + BLOCK_SIZE - 1) / BLOCK_SIZE);
  kernel_ptr2ind<<<grid, block>>>(ind, ptr, N, num);
}

at::Tensor torch_cuda_ptr2ind(at::Tensor &ptr, int64_t E) {
  int device = ptr.get_device();
  cudaSetDevice(device);
  ptr.contiguous();

  at::Tensor ind =
      at::zeros({E}, at::dtype(torch::kInt64).device(at::kCUDA, device));

  int64_t ptr_len = ptr.sizes()[0];

  int64_t *ind_arr = (int64_t *)ind.data_ptr();
  int64_t *ptr_arr = (int64_t *)ptr.data_ptr();

  cuda_ptr2ind(ind_arr, ptr_arr, ptr_len, E);
  return ind;
}

PYBIND11_MODULE(_convert_cuda, m) {
  m.doc() = "gammagl sparse convert cuda ops";
  m.def("cuda_torch_set_to_one", &torch_cuda_set_to_one);
  m.def("cuda_torch_ind2ptr", &torch_cuda_ind2ptr);
  m.def("cuda_torch_ptr2ind", &torch_cuda_ptr2ind);
  // m.def("test_main", &test_main);
}
