#include <torch/torch.h>
#include <tuple>

torch::Tensor segment_sum_cpu_forward(torch::Tensor& x,
                                                     torch::Tensor& index,
                                                     int64_t& N);
