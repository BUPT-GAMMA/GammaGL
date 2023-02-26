#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "pyg_lib/csrc/utils/convert.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/util/host_tensor.h>
#include <torch/library.h>
#include <torch/version.h>

void grouped_matmul_out_cuda_kernel(const at::TensorList input,
                               const at::TensorList other,
                               const at::TensorList out);
std::vector<at::Tensor> grouped_matmul_cuda_kernel(const at::TensorList input,
                                              const at::TensorList other);
at::Tensor segment_matmul_cuda_kernel(const at::Tensor &input, const at::Tensor &ptr,
                                 const at::Tensor &other);