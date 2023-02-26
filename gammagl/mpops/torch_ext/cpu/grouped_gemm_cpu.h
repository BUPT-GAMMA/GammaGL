#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/torch.h>

std::vector<torch::Tensor> grouped_matmul_kernel(const torch::TensorList input,
                                                 const torch::TensorList other);

at::Tensor segment_matmul_kernel_cpu(const at::Tensor &input, const at::Tensor &ptr,
                                 const at::Tensor &other);

at::Tensor size_from_ptr(const at::Tensor &ptr);