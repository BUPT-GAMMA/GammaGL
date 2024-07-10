#include "./bspmm_sum_cpu.h"
#include <torch/torch.h>
#include "ATen/core/TensorBody.h"

torch::Tensor bspmm_sum_cpu_forward(torch::Tensor &index, torch::Tensor &weight, torch::Tensor &x){
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    if (!weight.is_contiguous()) {
        weight = weight.contiguous();
    }
    if (!index.is_contiguous()) {
        index = index.contiguous();
    }

    int num_nodes = x.size(0);
    int heads = x.size(1);
    int out_channels = x.size(2);

    torch::Tensor out = torch::zeros_like(x, x.options());
    auto E = index.size(1);
    // auto K = x.numel() / x.size(0);

    auto index_data = index.data_ptr<int64_t>();
    using scalar_t = float;
    auto x_data = x.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    auto weight_data = weight.data_ptr<scalar_t>();

#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
    for (auto e = 0; e < E; ++e) {
        auto src = index_data[e];
        auto dst = index_data[e + E];

        for (auto h = 0; h < heads; ++h){
            for (auto k = 0; k < out_channels; ++k){
#ifdef COMPILE_WITH_OMP
#pragma omp atomic
#endif
                out_data[dst * out_channels * heads + h * out_channels + k] +=
                    weight_data[e * heads + h] * x_data[src * out_channels * heads + h * out_channels + k];
            }
        }
    }
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> bspmm_sum_cpu_backward(torch::Tensor &index, torch::Tensor &weight, torch::Tensor &x, torch::Tensor &grad) {
    if (!grad.is_contiguous()) {
        grad = grad.contiguous();
    }
    if (!weight.is_contiguous()) {
        weight = weight.contiguous();
    }
    if (!index.is_contiguous()) {
        index = index.contiguous();
    }

    int num_nodes = grad.size(0);
    int heads = grad.size(1);
    int out_channels = grad.size(2);

    torch::Tensor grad_x = torch::zeros_like(grad, grad.options());
    torch::Tensor grad_weight = torch::zeros_like(weight, weight.options());
    auto E = index.size(1);
    // auto K = grad.numel() / grad.size(0);

    auto index_data = index.data_ptr<int64_t>();
    using scalar_t = float;
    auto grad_data = grad.data_ptr<scalar_t>();
    auto grad_x_data = grad_x.data_ptr<scalar_t>();
    auto grad_weight_data = grad_weight.data_ptr<scalar_t>();
    auto x_data = x.data_ptr<scalar_t>();
    auto weight_data = weight.data_ptr<scalar_t>();

// 计算反向传播的梯度
#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
    for (auto e = 0; e < E; ++e) {
        auto src = index_data[e];
        auto dst = index_data[e + E];

        for (auto h = 0; h < heads; ++h){
            for (auto k = 0; k < out_channels; ++k){
#ifdef COMPILE_WITH_OMP
#pragma omp atomic
#endif
                grad_x_data[src * out_channels * heads + h * out_channels + k] +=
                    weight_data[e * heads + h] * grad_data[dst * out_channels * heads + h * out_channels + k];

                grad_weight_data[e * heads + h] += x_data[src * out_channels * heads + h * out_channels + k] * 
                grad_data[dst * out_channels * heads + h * out_channels + k];

            }
        }
    }
    // return {grad_x, grad_weight};
    return std::make_tuple(grad_x, grad_weight);
}