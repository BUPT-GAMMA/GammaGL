#include <torch/torch.h>
torch::Tensor segment_mean_cuda_forward(torch::Tensor x, torch::Tensor index, int64_t N);