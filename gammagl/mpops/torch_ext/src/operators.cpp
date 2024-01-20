#include "../include/segment_max.h"
#include "../include/segment_sum.h"
#include "../include/segment_mean.h"
#include "../include/gspmm.h"
#include <assert.h>
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

torch::Tensor spmm_sum(torch::Tensor index, torch::Tensor weight,
                       torch::Tensor x) {
  auto result = SpMMSum::apply(index, weight, x);
  return result;
}

torch::Tensor spmm_mean(torch::Tensor index, torch::Tensor weight, torch::Tensor x) {
    return SpMMMean::apply(index, weight, x);
}

torch::Tensor spmm_max(torch::Tensor index, torch::Tensor weight, torch::Tensor x) {
    return SpMMMax::apply(index, weight, x);
}

PYBIND11_MODULE(_torch_ext, m) {
  m.def("c_segment_max", segment_max);
  m.def("c_segment_sum", segment_sum);
  m.def("c_segment_mean", segment_mean);
  m.def("c_spmm_sum", spmm_sum);
  m.def("c_spmm_mean", spmm_mean);
  m.def("c_spmm_max", spmm_max);
}
