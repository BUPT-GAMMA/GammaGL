#include <torch/torch.h>

torch::Tensor spmm_sum_cpu_forward(torch::Tensor &index, torch::Tensor &weight,
                                   torch::Tensor &x);
torch::Tensor spmm_sum_cpu_backward(torch::Tensor &index, torch::Tensor &weight,
                                   torch::Tensor &grad);
