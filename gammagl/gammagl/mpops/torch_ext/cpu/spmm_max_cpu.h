#include <torch/torch.h>

std::tuple<torch::Tensor, torch::Tensor> spmm_max_cpu_forward(
    torch::Tensor &index, torch::Tensor &weight, torch::Tensor &x);
torch::Tensor spmm_max_cpu_backward(
    torch::Tensor &index, torch::Tensor &weight, torch::Tensor &grad,
    torch::Tensor &max_indices);
