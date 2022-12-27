#include <torch/torch.h>

torch::Tensor spmm_sum_cuda_forward(torch::Tensor &index, torch::Tensor &weight,
                                   torch::Tensor &x);
torch::Tensor spmm_sum_cuda_backward(torch::Tensor &index, torch::Tensor &weight,
                                   torch::Tensor &grad);
