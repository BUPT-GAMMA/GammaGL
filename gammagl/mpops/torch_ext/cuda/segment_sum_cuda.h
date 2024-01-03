#include <torch/torch.h>
torch::Tensor segment_sum_cuda_forward(
    torch::Tensor x, torch::Tensor index, int64_t N);