#include "spmm_sum_cpu.h"
#include <torch/torch.h>

torch::Tensor spmm_sum_cpu_forward(torch::Tensor &index, torch::Tensor &weight,
                                   torch::Tensor &x) {
  x = x.contiguous();
  torch::Tensor out = torch::zeros_like(x, x.options());
  auto E = index.size(1);
  auto K = x.size(1);

  auto index_data = index.data_ptr<int64_t>();
  int64_t col, row;
  // AT_DISPATCH_ALL_TYPES(x.scalar_type(), "__ops_name", [&] {
  using scalar_t = float;
  auto x_data = x.data_ptr<scalar_t>();
  auto out_data = out.data_ptr<scalar_t>();
  auto weight_data = weight.data_ptr<scalar_t>();
#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
  for (auto e = 0; e < E; ++e) {
    col = index_data[e];
    row = index_data[e + E]; // or e + 1;
    for (auto k = 0; k < K; ++k) {
#ifdef COMPILE_WITH_OMP
#pragma omp atomic
#endif
      out_data[row * K + k] = out_data[row * K + k] + weight_data[e] * x_data[col * K + k];
    }
  }

  return out;
}

// almost same with forward
torch::Tensor spmm_sum_cpu_backward(torch::Tensor &index, torch::Tensor &weight,
                                    torch::Tensor &grad) {
  grad = grad.contiguous();
  torch::Tensor out = torch::zeros_like(grad, grad.options());
  auto E = index.size(1);
  auto K = grad.size(1);

  auto index_data = index.data_ptr<int64_t>();
  int64_t col, row;
  // AT_DISPATCH_ALL_TYPES(x.scalar_type(), "__ops_name", [&] {
  using scalar_t = float;
  auto weight_data = weight.data_ptr<scalar_t>();
  auto grad_data = grad.data_ptr<scalar_t>();
  auto out_data = out.data_ptr<scalar_t>();
#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
  for (auto e = 0; e < E; ++e) {
    col = index_data[e];
    row = index_data[e + E]; // or e + 1;
    
    for (auto k = 0; k < K; ++k) {
#ifdef COMPILE_WITH_OMP
#pragma omp atomic
#endif
      out_data[col * K + k] = out_data[col * K + k] +  weight_data[e] * grad_data[row * K + k];
    }
  }

  return out;
}
