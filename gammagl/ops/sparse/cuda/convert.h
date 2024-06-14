#pragma once

#include <torch/extension.h>
#include <torch/torch.h>

#include "../../extensions.h"
#include "../../extensions_cuda.h"

#ifndef REMOTE_CONVERT_CUDA_H
#define REMOTE_CONVERT_CUDA_H

void set_to_one_cuda(Tensor arr);

at::Tensor torch_cuda_ind2ptr(at::Tensor &ind, int64_t M);

at::Tensor torch_cuda_ptr2ind(at::Tensor &ptr, int64_t E);

at::Tensor torch_cuda_set_to_one(at::Tensor &ts);

#endif