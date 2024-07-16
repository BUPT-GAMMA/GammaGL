#pragma once

#include <torch/extension.h>
#include <torch/torch.h>

#include "../../extensions.h"
#include "../../extensions_cuda.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

#ifndef REMOTE_NEIGHBOR_SAMPLE_CUDA_H
#define REMOTE_NEIGHBOR_SAMPLE_CUDA_H

py::list torch_cu_neighbor_sample(
    at::Tensor& colptr, at::Tensor& row, at::Tensor& input_nodes,
    at::Tensor& fanouts, bool replace, bool directed, int random_seed);

py::list torch_cu_sample_adj(
    at::Tensor& colptr, at::Tensor& row, at::Tensor& input_nodes,
    at::Tensor& fanouts, bool replace, bool directed, int random_seed);

#endif