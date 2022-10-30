#include <torch/torch.h>
std::tuple<torch::Tensor, torch::Tensor> segment_max_cpu_forward(torch::Tensor& src,
                                                     torch::Tensor& index,
                                                     int64_t& N);