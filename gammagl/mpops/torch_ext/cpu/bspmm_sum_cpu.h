#include <torch/torch.h>

torch::Tensor bspmm_sum_cpu_forward(
    torch::Tensor &index, torch::Tensor &weight, torch::Tensor &x);
std::tuple<torch::Tensor, torch::Tensor> bspmm_sum_cpu_backward(
    torch::Tensor &index, torch::Tensor &weight, torch::Tensor &x,
    torch::Tensor &grad);
