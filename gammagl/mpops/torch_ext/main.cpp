#include "c10/core/DeviceType.h"
#include "include/segment_max.h"
#include "include/segment_sum.h"
#include "include/segment_mean.h"
// #include "include/gspmm.h"
#include <assert.h>
#include <cstdint>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <vector>

torch::Tensor segment_max(torch::Tensor x, torch::Tensor index, int64_t N) {
  auto result = SegmentMax::apply(x, index, N);
  return result;
}

torch::Tensor segment_sum(torch::Tensor x, torch::Tensor index, int64_t N) {
  auto result = SegmentSum::apply(x, index, N);
  return result;
}

torch::Tensor segment_mean(torch::Tensor x, torch::Tensor index, int64_t N) {
  auto result = SegmentMean::apply(x, index, N);
  return result;
}

// torch::Tensor spmm_sum(torch::Tensor index, torch::Tensor weight,
//                        torch::Tensor x) {
//   auto result = SpMMSum::apply(index, weight, x);
//   return result;
// }

// PYBIND11_MODULE(torch_operator, m) {
//   m.def("segment_max", segment_max);
//   m.def("segment_sum", segment_sum);
//   m.def("segment_mean", segment_mean);
//   m.def("spmm_sum", spmm_sum);
// }

int main(){
    torch::Tensor x_sum = torch::tensor({{2.,3.,4.},{4.,2.,8.},{4.,9.,7.}});
    torch::Tensor index_sum = torch::tensor({0,2,2});
    int64_t N_sum = 3;

    torch::Tensor x_max = torch::tensor({{2.,3.,4.},{4.,2.,8.},{4.,9.,7.}});
    torch::Tensor index_max = torch::tensor({0,2,2});
    int64_t N_max = 3;


    torch::Tensor x_mean = torch::tensor({{2.,3.,4.},{4.,2.,8.},{4.,9.,7.}});
    torch::Tensor index_mean = torch::tensor({0,2,2});
    int64_t N_mean = 3;

    x_max = x_max.to(torch::kCUDA);
    index_max = index_max.to(torch::kCUDA);

    x_sum = x_sum.to(torch::kCUDA);
    index_sum = index_sum.to(torch::kCUDA);

    x_mean = x_mean.to(torch::kCUDA);
    index_mean = index_mean.to(torch::kCUDA);

    // std::cout << x << std::endl;
    // std::cout << index << std::endl;
    // std::cout << N << std::endl;

    std::cout << segment_max(x_max, index_max, N_max) << std::endl;
    std::cout << segment_sum(x_sum, index_sum, N_sum) << std::endl;
    std::cout << segment_mean(x_mean, index_mean, N_mean) << std::endl;
}

