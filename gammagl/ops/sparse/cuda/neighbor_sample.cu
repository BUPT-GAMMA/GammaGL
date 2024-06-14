/*
 * @Description: TODO
 * @Author: XiaoYixin
 * @created: 2024-05-21
 */

/*
 * 根据degree和min_degree_fanout得到对应的pos的时候，可以尝试把第一个位置空出来。
 */
#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
// #include <helper_cuda.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>
// #include <thrust/sort.h>

// #include "../ticktock.h"
#include "neighbor_sample.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

constexpr int BLOCK_SIZE = 128;

inline __device__ int64_t AtomicMax(int64_t* const address, const int64_t val) {
  // To match the type of "::atomicCAS", ignore lint warning.
  using Type = unsigned long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMax(reinterpret_cast<Type*>(address), static_cast<Type>(val));
}

__global__ void get_min_degree_fanout(
    int64_t* min_degree_fanout, int64_t* degree, int64_t* colptr,
    int64_t fanout, int64_t* input_nodes, int64_t num_input_node) {
  /*
   * Get the minimum value of node degrees and sampled fanouts.
   */
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  while (i < num_input_node) {
    const int64_t node = input_nodes[i];
    degree[i] = colptr[node + 1] - colptr[node];
    if (fanout == -1) min_degree_fanout[i] = degree[i];
    if (fanout > 0) min_degree_fanout[i] = min(degree[i], fanout);
    i += stride;
  }
}

// __global__ void init_pos(int64_t* pos, int64_t* size, int64_t num_node) {
//   int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
//   const int64_t stride = gridDim.x * blockDim.x;
//   int64_t i = it;
//   while (i < num_node) {
//     pos[i + 1] = size[i];
//     i += stride;
//   }
// }

__global__ void init_pos(int64_t* pos, int64_t* arr, int64_t size) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  while (i < size) {
    if (i == 0) {
      pos[i] = 0;
    }
    if (i != 0) {
      pos[i] = arr[i - 1];
    }
    i += stride;
  }
}

__global__ void get_pos_one_step(
    int64_t* pos, int64_t num_input_node, int64_t step_size) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  while (i < num_input_node) {
    if (i % step_size < (step_size >> 1)) return;
    const int64_t up_step = i / step_size * step_size + (step_size >> 1) - 1;
    pos[i] += pos[up_step];
    i += stride;
  }
}

__global__ void init_edge_to_node(
    int64_t* e_to_n, int64_t* pos_full, int64_t num) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  while (i < num) {
    e_to_n[pos_full[i]] = i;
    i += stride;
  }
}

__global__ void get_edge_to_node_one_step(
    int64_t* e_to_n, int64_t num_edge, int64_t step_size) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  while (i < num_edge) {
    if (i % step_size < (step_size >> 1)) return;
    const int64_t up_step = i / step_size * step_size + (step_size >> 1) - 1;
    e_to_n[i] = max(e_to_n[i], e_to_n[up_step]);
    i += stride;
  }
}

__global__ void get_eids_neighbor_sampler(
    int64_t* eids, int64_t* pos_full, int64_t* pos_sampler,
    int64_t* min_degree_fanout, int64_t* colptr, int64_t* e_to_n,
    int64_t* input_nodes, int64_t num_edge, int64_t fanout,
    int64_t random_seed) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;

  curandStatePhilox4_32_10_t rng;
  curand_init(random_seed, i, 0, &rng);

  while (i < num_edge) {
    const int64_t node = e_to_n[i];
    const int64_t row_offset = i - pos_full[node];
    const int64_t output_offset = pos_sampler[node];
    const int64_t rnd = row_offset < fanout || fanout == -1
                            ? row_offset
                            : curand(&rng) % (row_offset + 1);
    if (rnd < fanout || fanout == -1) {
      AtomicMax(
          eids + output_offset + rnd, row_offset + colptr[input_nodes[node]]);
    }
    i += stride;
  }
}

void output_int64_t(int64_t* arr, int64_t len) {
  // printf("[%lld] ", len);
  // if (len > 400) len = 400;
  // int64_t* ar;
  // // printf("----");
  // ar = (int64_t*)malloc(len * sizeof(int64_t));

  // // printf(">>>>");
  // checkCudaErrors(
  //     cudaMemcpy(ar, arr, len * sizeof(int64_t), cudaMemcpyDeviceToHost));

  // // printf("<<<<");
  // for (int i = 0; i < len; i++) printf("%lld ", (long long)ar[i]);
  // free(ar);
  // printf("\n");
}

struct my_array_int64_t {
  int64_t* arr;
  int64_t len;
};

my_array_int64_t cu_neighbor_sample_one_hop(
    int64_t* colptr, int64_t* row, int64_t* input_nodes, int64_t fanout,
    int64_t num_node, int64_t num_input_node, bool replace, bool directed,
    int random_seed) {
  // printf("fanout = %lld\n", (long long)fanout);

  cudaPointerAttributes attributes;
  checkCudaErrors(cudaPointerGetAttributes(&attributes, colptr));
  int device = attributes.device;

  // check: All data must be in several device.
  checkCudaErrors(cudaPointerGetAttributes(&attributes, row));
  assert(device == (int)attributes.device);

  checkCudaErrors(cudaPointerGetAttributes(&attributes, input_nodes));
  assert(device == (int)attributes.device);

  // checkCudaErrors(cudaPointerGetAttributes(&attributes, fanouts));
  // assert(device == (int)attributes.device);

  // checkCudaErrors(cudaSetDevice(device));

  if (replace) {
  } else {
    int64_t *degree, *min_degree_fanout;
    cudaMalloc((int64_t**)&degree, num_input_node * sizeof(int64_t));
    cudaMalloc((int64_t**)&min_degree_fanout, num_input_node * sizeof(int64_t));

    dim3 block(BLOCK_SIZE);
    dim3 grid((num_input_node + BLOCK_SIZE - 1) / BLOCK_SIZE);
    get_min_degree_fanout<<<grid, block>>>(
        min_degree_fanout, degree, colptr, fanout, input_nodes, num_input_node);

    // output degree and min degree fanout
    output_int64_t(degree, num_input_node);
    output_int64_t(min_degree_fanout, num_input_node);

    int64_t *pos_full, *pos_sampler;
    cudaMalloc((int64_t**)&pos_full, (num_input_node + 1) * sizeof(int64_t));
    cudaMalloc((int64_t**)&pos_sampler, (num_input_node + 1) * sizeof(int64_t));
    // cudaMemcpy(
    //     pos_full + 1, degree, num_input_node * sizeof(int64_t),
    //     cudaMemcpyDeviceToDevice);
    // cudaMemcpy(
    //     pos_sampler + 1, degree, num_input_node * sizeof(int64_t),
    //     cudaMemcpyDeviceToDevice);
    // dim3 block(BLOCK_SIZE);
    grid = (num_input_node + BLOCK_SIZE) / BLOCK_SIZE;
    init_pos<<<block, grid>>>(pos_full, degree, num_input_node + 1);
    init_pos<<<block, grid>>>(
        pos_sampler, min_degree_fanout, num_input_node + 1);

    output_int64_t(pos_full, num_input_node + 1);
    output_int64_t(pos_sampler, num_input_node + 1);

    // dim3 block(BLOCK_SIZE);
    grid = (num_input_node + BLOCK_SIZE) / BLOCK_SIZE;
    for (int64_t step_size = 2; step_size <= 2 * num_input_node;
         step_size <<= 1) {
      // printf("step size = %lld\n", (long long)step_size);
      get_pos_one_step<<<grid, block>>>(
          pos_full, num_input_node + 1, step_size);
      get_pos_one_step<<<grid, block>>>(
          pos_sampler, num_input_node + 1, step_size);
      cudaDeviceSynchronize();
      output_int64_t(pos_full, num_input_node + 1);
      output_int64_t(pos_sampler, num_input_node + 1);
    }

    output_int64_t(pos_full, num_input_node + 1);
    output_int64_t(pos_sampler, num_input_node + 1);

    // return;

    int64_t *num_edge_, *num_edge_sample_;
    num_edge_ = (int64_t*)malloc(sizeof(int64_t));
    num_edge_sample_ = (int64_t*)malloc(sizeof(int64_t));

    cudaMemcpy(
        num_edge_, pos_full + num_input_node, sizeof(int64_t),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        num_edge_sample_, pos_sampler + num_input_node, sizeof(int64_t),
        cudaMemcpyDeviceToHost);

    int64_t num_edge = num_edge_[0], num_edge_sample = num_edge_sample_[0];
    free(num_edge_);
    free(num_edge_sample_);

    // return;

    // printf("num_edge=%lld\n", (long long)num_edge);

    int64_t* e_to_n;
    checkCudaErrors(cudaMalloc((int64_t**)&e_to_n, num_edge * sizeof(int64_t)));
    checkCudaErrors(cudaMemset(e_to_n, 0, num_edge * sizeof(int64_t)));
    // printf("uninit e_to_n:");
    output_int64_t(e_to_n, num_edge);
    // dim3 block(BLOCK_SIZE);
    grid = (num_edge + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_edge_to_node<<<grid, block>>>(e_to_n, pos_full, num_input_node);
    // output_int64_t(e_to_n, num_edge);
    // return;
    for (int64_t step_size = 2; step_size < num_edge * 2; step_size <<= 1) {
      get_edge_to_node_one_step<<<grid, block>>>(e_to_n, num_edge, step_size);
    }

    output_int64_t(e_to_n, num_edge);

    // int num_edge_sample = pos_sampler[num_input_node];
    int64_t* eids;
    checkCudaErrors(
        cudaMalloc((int64_t**)&eids, num_edge_sample * sizeof(int64_t)));
    cudaMemset(eids, 0, num_edge_sample * sizeof(int64_t));

    get_eids_neighbor_sampler<<<grid, block>>>(
        eids, pos_full, pos_sampler, min_degree_fanout, colptr, e_to_n,
        input_nodes, num_edge, fanout, random_seed);

    cudaFree(e_to_n);

    // output_int64_t(eids, num_edge_sample);

    cudaFree(degree);
    cudaFree(min_degree_fanout);
    cudaFree(pos_full);
    cudaFree(pos_sampler);

    my_array_int64_t Eids;
    Eids.arr = eids;
    Eids.len = num_edge_sample;
    return Eids;
  }
}

__global__ void kernal_get_new_nodes(
    int64_t* new_nodes, int64_t* row, int64_t* eids, int64_t num_new_node) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;

  while (i < num_new_node) {
    new_nodes[i] = row[eids[i]];
    i += stride;
  }
}

__global__ void kernal_merge_int64_t(
    int64_t* merge, int64_t* arr1, int64_t* arr2, int64_t num1, int64_t num2) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;

  while (i < num1 + num2) {
    if (i < num1) {
      merge[i] = arr1[i] << 1;
    }
    if (i >= num1) {
      merge[i] = arr2[i - num1] << 1 | 1;
    }
    i += stride;
  }
}

__global__ void kernal_merge_arr_int64_t(
    int64_t* merge, int64_t* arr1, int64_t* arr2, int64_t num1, int64_t num2) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;

  while (i < num1 + num2) {
    if (i < num1) {
      merge[i] = arr1[i];
    }
    if (i >= num1) {
      merge[i] = arr2[i - num1];
    }
    i += stride;
  }
}

__global__ void prefix_sum_one_step(
    int64_t* arr, int64_t num, int64_t step_size) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  while (i < num) {
    if (i % step_size < (step_size >> 1)) return;
    const int64_t up_step = i / step_size * step_size + (step_size >> 1) - 1;
    arr[i] += arr[up_step];
    i += stride;
  }
}

__global__ void kernal_sort(int64_t* arr, int64_t num, int64_t _) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  while (i < num) {
    if (i & 1) {
      if ((_ & 1) && i + 1 < num && arr[i] > arr[i + 1]) {
        int64_t __ = arr[i];
        arr[i] = arr[i + 1];
        arr[i + 1] = __;
      }
      if (((_ & 1) ^ 1) && arr[i] < arr[i - 1]) {
        int64_t __ = arr[i];
        arr[i] = arr[i - 1];
        arr[i - 1] = __;
      }
    }
    i += stride;
  }
}

__global__ void kernal_push_one(
    int64_t* has_arr, int64_t* help_arr, int64_t num) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  while (i < num) {
    if (i == 0) {
      bool bo = help_arr[i] & 1;
      if (bo) {
        has_arr[i] = 1;
      }
      if (!bo) {
        has_arr[i] = 0;
      }
    }
    if (i) {
      bool bo =
          ((help_arr[i] & 1 == 1) && (help_arr[i] != help_arr[i - 1]) &&
           ((help_arr[i] ^ help_arr[i - 1]) != 1));
      if (bo) {
        has_arr[i] = 1;
      }
      if (!bo) {
        has_arr[i] = 0;
      }
    }
    i += stride;
  }
}

__global__ void kernal_push_new_nodes(
    int64_t* only_new_nodes, int64_t* merge_nodes, int64_t* has_new_node,
    int num) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  while (i < num) {
    if (has_new_node[i] > 0 &&
        (i == 0 || has_new_node[i] != has_new_node[i - 1])) {
      only_new_nodes[has_new_node[i] - 1] = merge_nodes[i] >> 1;
    }
    i += stride;
  }
}

my_array_int64_t get_new_input_nodes(
    int64_t* input_nodes, int64_t* new_nodes, int64_t num_input_node,
    int64_t num_new_node) {
  // printf("get new input nodes:\n");
  // printf("  input nodes:");
  output_int64_t(input_nodes, num_input_node);
  // printf("  new nodes:");
  output_int64_t(new_nodes, num_new_node);

  int64_t* merge_nodes;
  cudaMalloc(
      (int64_t**)&merge_nodes,
      (num_input_node + num_new_node) * sizeof(int64_t));

  dim3 block(BLOCK_SIZE);
  dim3 grid((num_input_node + num_new_node + BLOCK_SIZE - 1) / BLOCK_SIZE);
  kernal_merge_int64_t<<<grid, block>>>(
      merge_nodes, input_nodes, new_nodes, num_input_node, num_new_node);
  for (int64_t _ = 0; _ <= num_input_node + num_new_node; _++) {
    kernal_sort<<<grid, block>>>(merge_nodes, num_input_node + num_new_node, _);
  }
  // printf("merge nodes:");
  // output_int64_t(merge_nodes, num_input_node + num_new_node);

  int64_t* has_new_node;
  cudaMalloc(
      (int64_t**)&has_new_node,
      (num_input_node + num_new_node) * sizeof(int64_t));
  kernal_push_one<<<grid, block>>>(
      has_new_node, merge_nodes, num_input_node + num_new_node);
  // printf("has new nodes:");
  // output_int64_t(has_new_node, num_input_node + num_new_node);

  for (int64_t step_size = 2; step_size < 2 * (num_input_node + num_new_node);
       step_size <<= 1) {
    prefix_sum_one_step<<<grid, block>>>(
        has_new_node, num_input_node + num_new_node, step_size);
  }

  int64_t* num_only_new_node_;
  num_only_new_node_ = (int64_t*)(malloc(sizeof(int64_t)));
  cudaMemcpy(
      num_only_new_node_, has_new_node + num_input_node + num_new_node - 1,
      sizeof(int64_t), cudaMemcpyDeviceToHost);

  int64_t num_only_new_node = num_only_new_node_[0];

  free(num_only_new_node_);

  int64_t* only_new_nodes;
  cudaMalloc((int64_t**)&only_new_nodes, num_only_new_node * sizeof(int64_t));
  kernal_push_new_nodes<<<grid, block>>>(
      only_new_nodes, merge_nodes, has_new_node, num_input_node + num_new_node);
  // printf("only new node:");
  // output_int64_t(only_new_nodes, num_only_new_node);

  my_array_int64_t new_input_nodes;
  cudaMalloc(
      (int64_t**)&new_input_nodes.arr,
      (num_input_node + num_only_new_node) * sizeof(int64_t));

  grid = (num_input_node + num_only_new_node + BLOCK_SIZE - 1) / BLOCK_SIZE;
  kernal_merge_arr_int64_t<<<grid, block>>>(
      new_input_nodes.arr, input_nodes, only_new_nodes, num_input_node,
      num_only_new_node);
  new_input_nodes.len = num_input_node + num_only_new_node;

  cudaFree(merge_nodes);
  cudaFree(has_new_node);
  cudaFree(only_new_nodes);

  return new_input_nodes;
}

__global__ void init_name_arr(int64_t* name_arr, int64_t num) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  while (i < num) {
    name_arr[i] = i;
    i += stride;
  }
}

__global__ void kernal_sort2(
    int64_t* arr, int64_t* name, int64_t num, int64_t _) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  while (i < num) {
    if (i & 1) {
      if ((_ & 1) && i + 1 < num && arr[i] > arr[i + 1]) {
        int64_t __ = arr[i];
        arr[i] = arr[i + 1];
        arr[i + 1] = __;
        __ = name[i];
        name[i] = name[i + 1];
        name[i + 1] = __;
      }
      if (((_ & 1) ^ 1) && arr[i] < arr[i - 1]) {
        int64_t __ = arr[i];
        arr[i] = arr[i - 1];
        arr[i - 1] = __;
        __ = name[i];
        name[i] = name[i - 1];
        name[i - 1] = __;
      }
    }
    i += stride;
  }
  // __syncthreads();
}

__global__ void kernal_get_col(
    int64_t* output, int64_t* ord, int64_t* name, int64_t* input,
    int64_t* edges, int64_t num_node, int64_t num_edge) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  int64_t l, r, mid;
  while (i < num_edge) {
    l = 0, r = num_node - 1;
    while (l != r) {
      mid = l + r + 1 >> 1;
      bool bo = input[ord[mid]] <= edges[i];
      if (bo) l = mid;
      if (!bo) r = mid - 1;
    }
    output[i] = name[l];
    i += stride;
  }
}

__global__ void kernal_get_row(
    int64_t* output, int64_t* ord, int64_t* name, int64_t* input,
    int64_t* edges, int64_t num_node, int64_t num_edge) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  int64_t l, r, mid;
  while (i < num_edge) {
    l = 0, r = num_node - 1;
    while (l != r) {
      mid = l + r + 1 >> 1;
      bool bo = ord[mid] <= input[edges[i]];
      if (bo) l = mid;
      if (!bo) r = mid - 1;
    }
    output[i] = name[l];
    i += stride;
  }
}

void cu_get_cols_and_rows(
    my_array_int64_t* output, my_array_int64_t nodes, my_array_int64_t edges,
    int64_t* colptr, int64_t* row) {
  int64_t* sort_nodes;
  cudaMalloc((int64_t**)&sort_nodes, nodes.len * sizeof(int64_t));
  cudaMemcpy(
      sort_nodes, nodes.arr, nodes.len * sizeof(int64_t),
      cudaMemcpyDeviceToDevice);

  int64_t* name_arr;
  cudaMalloc((int64_t**)&name_arr, nodes.len * sizeof(int64_t));
  dim3 block(BLOCK_SIZE);
  dim3 grid((nodes.len + BLOCK_SIZE - 1) / BLOCK_SIZE);
  init_name_arr<<<grid, block>>>(name_arr, nodes.len);
  // printf("init name arr:");
  // output_int64_t(name_arr, nodes.len);

  for (int64_t _ = 0; _ <= nodes.len; _++) {
    kernal_sort2<<<grid, block>>>(sort_nodes, name_arr, nodes.len, _);
  }
  // printf("sort nodes:");
  // output_int64_t(sort_nodes, nodes.len);
  // printf("sort name arr:");
  // output_int64_t(name_arr, nodes.len);

  // my_array_int64_t col_row[2];
  cudaMalloc((int64_t**)&output[0].arr, edges.len * sizeof(int64_t));
  cudaMalloc((int64_t**)&output[1].arr, edges.len * sizeof(int64_t));
  grid = ((edges.len + BLOCK_SIZE - 1) / BLOCK_SIZE);
  output[0].len = edges.len;
  output[1].len = edges.len;
  kernal_get_col<<<grid, block>>>(
      output[0].arr, sort_nodes, name_arr, colptr, edges.arr, nodes.len,
      edges.len);
  // printf("Sample cols:");
  // output_int64_t(output[0].arr, edges.len);
  kernal_get_row<<<grid, block>>>(
      output[1].arr, sort_nodes, name_arr, row, edges.arr, nodes.len,
      edges.len);

  cudaFree(sort_nodes);
  cudaFree(name_arr);
}

__global__ void kernal_ind2ptr(int64_t* ptr, int64_t* ind, int64_t num) {
  int64_t it = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = gridDim.x * blockDim.x;
  int64_t i = it;
  while (i < num) {
    if (i == num - 1) ptr[ind[i] + 1] = i + 1;
    if (i == 0) ptr[0] = 0;
    if (i > 0 && ind[i] != ind[i - 1]) ptr[ind[i]] = i;
    i += stride;
  }
}

void cu_neighbor_sample(
    my_array_int64_t* output, int64_t* colptr, int64_t* row,
    int64_t* input_nodes, int64_t* fanouts, int num_node, int num_input_node,
    int num_fanout, bool replace, bool directed, int random_seed) {
  my_array_int64_t eids_array[num_fanout];

  int64_t last_num_input_node = 0;
  int64_t sum_num_e = 0;
  for (int layer = 0; layer < num_fanout; layer++) {
    // printf("input nodes:");
    // output_int64_t(input_nodes, num_input_node);
    int fanout = fanouts[layer];
    eids_array[layer] = cu_neighbor_sample_one_hop(
        colptr, row, input_nodes + last_num_input_node, fanout, num_node,
        num_input_node - last_num_input_node, replace, directed, random_seed);
    sum_num_e = sum_num_e + eids_array[layer].len;

    int num_new_node = eids_array[layer].len;
    int64_t* new_nodes;
    cudaMalloc((int64_t**)&new_nodes, num_new_node * sizeof(int64_t));
    dim3 block(BLOCK_SIZE);
    dim3 grid((num_new_node + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernal_get_new_nodes<<<grid, block>>>(
        new_nodes, row, eids_array[layer].arr, num_new_node);

    my_array_int64_t new_input_nodes = get_new_input_nodes(
        input_nodes, new_nodes, num_input_node, num_new_node);

    // printf("new input nodes:");
    // output_int64_t(new_input_nodes.arr, new_input_nodes.len);
    input_nodes = new_input_nodes.arr;
    last_num_input_node = num_input_node;
    num_input_node = new_input_nodes.len;
    if (num_input_node == last_num_input_node) {
      num_fanout = layer + 1;
      break;
    }
    cudaFree(new_nodes);
  }

  my_array_int64_t sample_nodes;
  sample_nodes.arr = input_nodes, sample_nodes.len = num_input_node;

  my_array_int64_t sample_edges;
  cudaMalloc((int64_t**)&sample_edges.arr, sum_num_e * sizeof(int64_t));
  sample_edges.len = 0;
  for (int layer = 0; layer < num_fanout; layer++) {
    output_int64_t(eids_array[layer].arr, eids_array[layer].len);
    cudaMemcpy(
        sample_edges.arr + sample_edges.len, eids_array[layer].arr,
        eids_array[layer].len * sizeof(int64_t), cudaMemcpyDeviceToDevice);
    sample_edges.len += eids_array[layer].len;
    // printf("%lld\n", (long long)sample_edges.len);
    cudaFree(eids_array[layer].arr);
  }
  // printf("sample edges:");
  output_int64_t(sample_edges.arr, sample_edges.len);

  my_array_int64_t sample_cols, sample_rows;
  my_array_int64_t s[2];
  cu_get_cols_and_rows(s, sample_nodes, sample_edges, colptr, row);

  sample_cols = s[0];
  sample_rows = s[1];

  // my_array_int64_t rt[4];
  output[0] = sample_cols;
  output[1] = sample_rows;
  output[2] = sample_nodes;
  output[3] = sample_edges;
  // printf("sample cols:");
  // output_int64_t(sample_cols.arr, sample_cols.len);
  // printf("sample rows:");
  // output_int64_t(sample_rows.arr, sample_rows.len);

  // return rt;
}

py::list torch_cu_neighbor_sample(
    at::Tensor& colptr, at::Tensor& row, at::Tensor& input_nodes,
    at::Tensor& fanouts, bool replace, bool directed, int random_seed) {
  int device = colptr.get_device();
  cudaSetDevice(device);

  int64_t* colptr_arr = (int64_t*)colptr.data_ptr();
  int64_t* row_arr = (int64_t*)row.data_ptr();
  int64_t* input_nodes_arr = (int64_t*)input_nodes.data_ptr();
  int64_t* fanouts_arr = (int64_t*)fanouts.data_ptr();
  int num_node = colptr.sizes()[0] - 1, num_input_node = input_nodes.sizes()[0],
      num_fanout = fanouts.sizes()[0];
  // printf("IIInput arr:");
  // output_int64_t(input_nodes_arr, num_input_node);
  my_array_int64_t output[4];
  cu_neighbor_sample(
      output, colptr_arr, row_arr, input_nodes_arr, fanouts_arr, num_node,
      num_input_node, num_fanout, replace, directed, random_seed);

  py::list res;
  for (int i = 0; i < 4; i++) {
    cudaPointerAttributes attributes;
    checkCudaErrors(cudaPointerGetAttributes(&attributes, output[i].arr));
    int device = attributes.device;
    at::Tensor out = at::zeros(
        {output[i].len}, at::dtype(torch::kInt64).device(at::kCUDA, device));
    cudaMemcpy(
        out.data_ptr(), output[i].arr, output[i].len * sizeof(int64_t),
        cudaMemcpyDeviceToDevice);
    res.append(out);
    cudaFree(output[i].arr);
  }
  cudaDeviceSynchronize();
  return res;
}

void cu_sample_adj(
    my_array_int64_t* output, int64_t* colptr, int64_t* row,
    int64_t* input_nodes, int64_t* fanouts, int num_node, int num_input_node,
    int num_fanout, bool replace, bool directed, int random_seed) {
  my_array_int64_t eids_array[num_fanout];

  int64_t last_num_input_node = 0;
  int64_t sum_num_e = 0;
  for (int layer = 0; layer < num_fanout; layer++) {
    // printf("input nodes:");
    // output_int64_t(input_nodes, num_input_node);
    int64_t fanout = fanouts[layer];
    eids_array[layer] = cu_neighbor_sample_one_hop(
        colptr, row, input_nodes + last_num_input_node, fanout, num_node,
        num_input_node - last_num_input_node, replace, directed, random_seed);
    sum_num_e = sum_num_e + eids_array[layer].len;

    int64_t num_new_node = eids_array[layer].len;
    int64_t* new_nodes;
    cudaMalloc((int64_t**)&new_nodes, num_new_node * sizeof(int64_t));
    dim3 block(BLOCK_SIZE);
    dim3 grid((num_new_node + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernal_get_new_nodes<<<grid, block>>>(
        new_nodes, row, eids_array[layer].arr, num_new_node);

    my_array_int64_t new_input_nodes = get_new_input_nodes(
        input_nodes, new_nodes, num_input_node, num_new_node);

    // printf("new input nodes:");
    // output_int64_t(new_input_nodes.arr, new_input_nodes.len);
    input_nodes = new_input_nodes.arr;
    last_num_input_node = num_input_node;
    num_input_node = new_input_nodes.len;
    if (num_input_node == last_num_input_node) {
      num_fanout = layer + 1;
      break;
    }
    cudaFree(new_nodes);
  }

  my_array_int64_t sample_nodes;
  sample_nodes.arr = input_nodes, sample_nodes.len = num_input_node;

  my_array_int64_t sample_edges;
  cudaMalloc((int64_t**)&sample_edges.arr, sum_num_e * sizeof(int64_t));
  sample_edges.len = 0;
  for (int layer = 0; layer < num_fanout; layer++) {
    output_int64_t(eids_array[layer].arr, eids_array[layer].len);
    cudaMemcpy(
        sample_edges.arr + sample_edges.len, eids_array[layer].arr,
        eids_array[layer].len * sizeof(int64_t), cudaMemcpyDeviceToDevice);
    sample_edges.len += eids_array[layer].len;
    // printf("%lld\n", (long long)sample_edges.len);
    cudaFree(eids_array[layer].arr);
  }
  // printf("sample edges:");
  // output_int64_t(sample_edges.arr, sample_edges.len);

  my_array_int64_t sample_cols, sample_rows;

  my_array_int64_t s[2];
  cu_get_cols_and_rows(s, sample_nodes, sample_edges, colptr, row);

  sample_cols = s[0];
  sample_rows = s[1];

  my_array_int64_t sample_colptrs;
  cudaMalloc(
      (int64_t**)&sample_colptrs.arr,
      (last_num_input_node + 1) * sizeof(int64_t));
  sample_colptrs.len = last_num_input_node + 1;
  dim3 block(BLOCK_SIZE);
  dim3 grid((sample_cols.len + BLOCK_SIZE - 1) / BLOCK_SIZE);
  kernal_ind2ptr<<<grid, block>>>(
      sample_colptrs.arr, sample_cols.arr, sample_cols.len);
  // printf("last num input node:%lld\n", (long long)last_num_input_node);
  // printf("col:");
  // output_int64_t(sample_cols.arr, sample_cols.len);
  // printf("colptr:");
  // output_int64_t(sample_colptrs.arr, sample_colptrs.len);

  cudaFree(sample_cols.arr);

  // my_array_int64_t rt[4];
  output[0] = sample_colptrs;
  output[1] = sample_rows;
  output[2] = sample_nodes;
  output[3] = sample_edges;
  // printf("sample cols:");
  // output_int64_t(sample_cols.arr, sample_cols.len);
  // printf("sample rows:");
  // output_int64_t(sample_rows.arr, sample_rows.len);

  // return rt;
}

py::list torch_cu_sample_adj(
    at::Tensor& colptr, at::Tensor& row, at::Tensor& input_nodes,
    at::Tensor& fanouts, bool replace, bool directed, int random_seed) {
  int device = colptr.get_device();
  cudaSetDevice(device);

  colptr = colptr.contiguous();
  row = row.contiguous();
  input_nodes = input_nodes.contiguous();
  fanouts = fanouts.contiguous();

  int64_t* colptr_arr = (int64_t*)colptr.data_ptr();
  int64_t* row_arr = (int64_t*)row.data_ptr();
  int64_t* input_nodes_arr = (int64_t*)input_nodes.data_ptr();
  int64_t* fanouts_arr = (int64_t*)fanouts.data_ptr();

  int num_node = colptr.sizes()[0] - 1, num_input_node = input_nodes.sizes()[0],
      num_fanout = fanouts.sizes()[0];
  // printf("IIInput arr:");
  // output_int64_t(input_nodes_arr, num_input_node);
  my_array_int64_t output[4];
  cu_sample_adj(
      output, colptr_arr, row_arr, input_nodes_arr, fanouts_arr, num_node,
      num_input_node, num_fanout, replace, directed, random_seed);

  // printf("sample is OK\n");
  py::list res;
  for (int i = 0; i < 4; i++) {
    cudaPointerAttributes attributes;
    checkCudaErrors(cudaPointerGetAttributes(&attributes, output[i].arr));
    int device = attributes.device;
    at::Tensor out = at::zeros(
        {output[i].len}, at::dtype(torch::kInt64).device(at::kCUDA, device));
    cudaMemcpy(
        out.data_ptr(), output[i].arr, output[i].len * sizeof(int64_t),
        cudaMemcpyDeviceToDevice);
    res.append(out);
    cudaFree(output[i].arr);
    // printf("{%d}\n", i);
  }
  cudaDeviceSynchronize();
  return res;
}

PYBIND11_MODULE(_cu_neighbor_sample, m) {
  m.doc() = "gammagl sparse neighbor_sample in cuda";
  m.def("cuda_torch_neighbor_sample", &torch_cu_neighbor_sample);
  m.def("cuda_torch_sample_adj", &torch_cu_sample_adj);
}
// int main(int argc, char** argv) {
//   int nDeviceNumber = 0;

//   checkCudaErrors(cudaGetDeviceCount(&nDeviceNumber));

//   printf("%d\n", nDeviceNumber);

//   int dev = 0;

//   checkCudaErrors(cudaSetDevice(dev));

//   int64_t colptr_[] = {0, 3, 6, 9, 11, 12},
//           row_[] = {0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 4},
//           input_nodes_[] = {3, 4};
//   int64_t *colptr, *row, *input_nodes;
//   cudaMalloc((int64_t**)&colptr, 6 * sizeof(int64_t));
//   cudaMalloc((int64_t**)&row, 12 * sizeof(int64_t));
//   cudaMalloc((int64_t**)&input_nodes, 2 * sizeof(int64_t));
//   cudaMemcpy(colptr, colptr_, 6 * sizeof(int64_t), cudaMemcpyHostToDevice);
//   cudaMemcpy(row, row_, 12 * sizeof(int64_t), cudaMemcpyHostToDevice);
//   cudaMemcpy(
//       input_nodes, input_nodes_, 2 * sizeof(int64_t),
//       cudaMemcpyHostToDevice);
//   int num_fanout = 2, num_node = 5, num_input_node = 2;
//   int64_t fanouts[] = {2, 2};
//   my_array_int64_t output[4];
//   cu_neighbor_sample(
//       output, colptr, row, input_nodes, fanouts, num_node, num_input_node,
//       num_fanout, false, false, 0);
// }