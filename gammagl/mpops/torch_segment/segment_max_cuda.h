#pragma once
#include <torch/torch.h>
torch::Tensor segment_max_cuda(torch::Tensor src, torch::Tensor index, int64_t N);