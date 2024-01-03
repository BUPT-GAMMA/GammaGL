#include "convert.h"
#include "ticktock.h"
// #include <__clang_cuda_runtime_wrapper.h>
#include <cstdint>
#include <iostream>

#define THREADS 256

__global__ void set_to_one_kernel(int64_t *arr, int n) {
  int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n)
    return;
  arr[tid] = 1;
}

void set_to_one_cuda(Tensor arr) {
  int n = arr.size();
  int block_size = (n - 1 + THREADS) / THREADS;

  // auto arr_data = arr.mutable_data();
  int64_t *arr_data;

  checkCudaErrors(cudaMallocManaged(&arr_data, n * sizeof(int64_t)));

  set_to_one_kernel<<<block_size, THREADS>>>(arr_data, n);

  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(arr.mutable_data(), arr_data, n * sizeof(int64_t),
                             cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(arr_data));
}


typedef py::array_t<int> Tensor32;

__global__ void ind2ptr_kernel(int *ind_data, int *out_data, int M,
                               int numel) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  //   if (tid >= 0 && tid < numel) {
  //     printf("data=%ld,", ind_data[tid]);
  //   }

  if (tid == 0) {
    for (int64_t i = 0; i <= ind_data[0]; i++)
      out_data[i] = 0;
  } else if (tid < numel) {
    for (int64_t i = ind_data[tid - 1]; i < ind_data[tid]; i++)
      out_data[i + 1] = tid;
  } else if (tid == numel) {
    for (int64_t i = ind_data[numel - 1] + 1; i < M + 1; i++) {
      out_data[i] = numel;
    }
  }
}

Tensor32 ind2ptr_cuda(Tensor32 ind, int M) {

  cout << "这里是cuda算子" << endl;
  int numel = ind.size();

  auto ind_data = ind.mutable_data();
  Tensor32 out{M + 1};

  int *ind_data_ptr;
  //   auto out_data = out.mutable_data();
  int *out_data_ptr;

  TICK(first)
  checkCudaErrors(cudaMalloc(&ind_data_ptr, numel * sizeof(int)));
  TOCK(first)

  TICK(pre)
  checkCudaErrors(cudaMemcpy(ind_data_ptr, ind_data, numel * sizeof(int),
                             cudaMemcpyHostToDevice));

  TOCK(pre)

  checkCudaErrors(cudaDeviceSynchronize());

  //   checkCudaErrors(cudaMallocManaged(&out_data, (M + 1) * sizeof(int)));

  TICK(second)
  checkCudaErrors(cudaMalloc(&out_data_ptr, (M + 1) * sizeof(int)));
  checkCudaErrors(cudaDeviceSynchronize());
  TOCK(second)

  TICK(kernel)

  auto stream = cudaStreamPerThread;
  ind2ptr_kernel<<<(numel + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
      ind_data_ptr, out_data_ptr, M, numel);

  checkCudaErrors(cudaDeviceSynchronize());
  TOCK(kernel)

  TICK(post)

  checkCudaErrors(cudaMemcpy(out.mutable_data(), out_data_ptr,
                             (M + 1) * sizeof(int),
                             cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaDeviceSynchronize());
  TOCK(post)

  checkCudaErrors(cudaFree(ind_data_ptr));
  checkCudaErrors(cudaFree(out_data_ptr));

  //   if (numel == 0)
  //     return out;

  //   auto ind_data = ind.mutable_data();
  //   auto out_data = out.mutable_data();

  //   checkCudaErrors(cudaMallocManaged(&ind_data, numel *
  //   sizeof(int64_t))); checkCudaErrors(cudaMallocManaged(&out_data, (M +
  //   1) * sizeof(int64_t)));

  //   auto stream = cudaStreamPerThread;
  //   ind2ptr_kernel<<<(numel + THREADS - 1) / THREADS, THREADS>>>(
  //       ind_data, out_data, M, numel);

  //   checkCudaErrors(cudaMemcpy(out.mutable_data(), out_data,
  //                              (M + 1) * sizeof(int64_t),
  //                              cudaMemcpyDeviceToHost));

  return out;
}


// __global__ void ind2ptr_kernel(int64_t *ind_data, int64_t *out_data, int64_t M,
//                                int64_t numel) {

//   int tid = blockDim.x * blockIdx.x + threadIdx.x;

//   //   if (tid >= 0 && tid < numel) {
//   //     printf("data=%ld,", ind_data[tid]);
//   //   }

//   if (tid == 0) {
//     for (int64_t i = 0; i <= ind_data[0]; i++)
//       out_data[i] = 0;
//   } else if (tid < numel) {
//     for (int64_t i = ind_data[tid - 1]; i < ind_data[tid]; i++)
//       out_data[i + 1] = tid;
//   } else if (tid == numel) {
//     for (int64_t i = ind_data[numel - 1] + 1; i < M + 1; i++) {
//       out_data[i] = numel;
//     }
//   }
// }

// Tensor32 ind2ptr_cuda(Tensor ind, int64_t M) {

//   cout << "这里是cuda算子" << endl;
//   int numel = ind.size();

//   auto ind_data = ind.mutable_data();
//   Tensor out{M + 1};

//   int64_t *ind_data_ptr;
//   //   auto out_data = out.mutable_data();
//   int64_t *out_data_ptr;

//   TICK(first)
//   checkCudaErrors(cudaMalloc(&ind_data_ptr, numel * sizeof(int64_t)));
//   TOCK(first)

//   TICK(pre)
//   checkCudaErrors(cudaMemcpy(ind_data_ptr, ind_data, numel * sizeof(int64_t),
//                              cudaMemcpyHostToDevice));

//   TOCK(pre)

//   checkCudaErrors(cudaDeviceSynchronize());

//   //   checkCudaErrors(cudaMallocManaged(&out_data, (M + 1) * sizeof(int)));

//   TICK(second)
//   checkCudaErrors(cudaMalloc(&out_data_ptr, (M + 1) * sizeof(int64_t)));
//   checkCudaErrors(cudaDeviceSynchronize());
//   TOCK(second)

//   TICK(kernel)

//   auto stream = cudaStreamPerThread;
//   ind2ptr_kernel<<<(numel + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
//       ind_data_ptr, out_data_ptr, M, numel);

//   checkCudaErrors(cudaDeviceSynchronize());
//   TOCK(kernel)

//   TICK(post)

//   checkCudaErrors(cudaMemcpy(out.mutable_data(), out_data_ptr,
//                              (M + 1) * sizeof(int64_t),
//                              cudaMemcpyDeviceToHost));

//   checkCudaErrors(cudaDeviceSynchronize());
//   TOCK(post)

//   checkCudaErrors(cudaFree(ind_data_ptr));
//   checkCudaErrors(cudaFree(out_data_ptr));

//   //   if (numel == 0)
//   //     return out;

//   //   auto ind_data = ind.mutable_data();
//   //   auto out_data = out.mutable_data();

//   //   checkCudaErrors(cudaMallocManaged(&ind_data, numel *
//   //   sizeof(int64_t))); checkCudaErrors(cudaMallocManaged(&out_data, (M +
//   //   1) * sizeof(int64_t)));

//   //   auto stream = cudaStreamPerThread;
//   //   ind2ptr_kernel<<<(numel + THREADS - 1) / THREADS, THREADS>>>(
//   //       ind_data, out_data, M, numel);

//   //   checkCudaErrors(cudaMemcpy(out.mutable_data(), out_data,
//   //                              (M + 1) * sizeof(int64_t),
//   //                              cudaMemcpyDeviceToHost));

//   return out;
// }

__global__ void ptr2ind_kernel(const int64_t *ptr_data, int64_t *out_data,
                               int64_t E, int64_t numel) {

  int64_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (thread_idx < numel) {
    int64_t idx = ptr_data[thread_idx], next_idx = ptr_data[thread_idx + 1];
    printf("idx=%ld,next_idx=%ld\n", idx, next_idx);
    for (int64_t i = idx; i < next_idx; i++) {
      out_data[i] = thread_idx;
      printf("thread_idx=%ld\n", thread_idx);
    }
  }
}

Tensor ptr2ind_cuda(Tensor ptr, int64_t E) {

  int size = ptr.size();
  Tensor out{E};
  auto ptr_data = ptr.data();
  auto out_data = out.mutable_data();

  checkCudaErrors(cudaMallocManaged(&ptr_data, size * sizeof(int64_t)));
  checkCudaErrors(cudaMallocManaged(&out_data, E * sizeof(int64_t)));

  ptr2ind_kernel<<<(size - 1 + THREADS - 1) / THREADS, THREADS>>>(
      ptr_data, out_data, E, size - 1);

  checkCudaErrors(cudaMemcpy(out.mutable_data(), out_data, E * sizeof(int64_t),
                             cudaMemcpyDeviceToHost));

  return out;
}

__global__ void kernel(int64_t *arr, int n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n)
    return;
  arr[tid] = tid;
}

int test_main() {
  int n = 65535;
  int64_t *arr;
  int thread_num = 256;
  // 向上取整
  int block_num = (n + thread_num - 1) / thread_num;
  checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int64_t)));
  kernel<<<block_num, thread_num>>>(arr, n);
  checkCudaErrors(cudaDeviceSynchronize());
  for (int i = 0; i < n; i++) {
    printf("arr[%ld]:%ld\n", i, arr[i]);
  }
  cudaFree(arr);
  return 0;
}

PYBIND11_MODULE(_convert_cuda, m) {
  m.doc() = "gammagl sparse convert cuda ops";
  m.def("cuda_set_to_one", &set_to_one_cuda);
  m.def("cuda_ind2ptr", &ind2ptr_cuda);
  m.def("cuda_ptr2ind", &ptr2ind_cuda);
  m.def("test_main", &test_main);
}
