#include "spmm_max_cpu.h"
#include <torch/torch.h>

std::tuple<torch::Tensor, torch::Tensor> spmm_max_cpu_forward(torch::Tensor &index, torch::Tensor &weight, torch::Tensor &x) {
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    if (!weight.is_contiguous()) {
        weight = weight.contiguous();
    }
    if (!index.is_contiguous()) {
        index = index.contiguous();
    }
    using scalar_t = float;
    // 初始化输出张量为最小浮点数
    torch::Tensor out = torch::full_like(x, std::numeric_limits<scalar_t>::lowest(), x.options()); 
    torch::Tensor max_indices = torch::zeros_like(x, torch::kInt64); // 保存最大值索引

    auto E = index.size(1);
    auto K = x.numel() / x.size(0);

    auto index_data = index.data_ptr<int64_t>();
    auto x_data = x.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    auto weight_data = weight.data_ptr<scalar_t>();
    auto max_indices_data = max_indices.data_ptr<int64_t>();

#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
    for (auto e = 0; e < E; ++e) {
        auto src = index_data[e];
        auto dst = index_data[e + E];

        for (auto k = 0; k < K; ++k) {
            scalar_t weighted_value = weight_data[e] * x_data[src * K + k];
            #ifdef COMPILE_WITH_OMP
            #pragma omp critical
            #endif
            {
                if (out_data[dst * K + k] < weighted_value) {
                    out_data[dst * K + k] = weighted_value;
                    max_indices_data[dst * K + k] = src; // 保存产生最大值的索引
                }
            }
        }
    }

    // return out;
    return std::make_tuple(out, max_indices);
}

torch::Tensor spmm_max_cpu_backward(torch::Tensor &index, torch::Tensor &weight, torch::Tensor &grad, torch::Tensor &max_indices) {
    if (!grad.is_contiguous()) {
        grad = grad.contiguous();
    }
    if (!weight.is_contiguous()) {
        weight = weight.contiguous();
    }
    if (!index.is_contiguous()) {
        index = index.contiguous();
    }
    torch::Tensor out = torch::zeros_like(grad, grad.options());
    auto E = index.size(1);
    auto K = grad.size(1);

    auto index_data = index.data_ptr<int64_t>();
    using scalar_t = float;
    auto grad_data = grad.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    auto weight_data = weight.data_ptr<scalar_t>();
    auto max_indices_data = max_indices.data_ptr<int64_t>();

#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
    for (auto e = 0; e < E; ++e) {
        auto src = index_data[e];
        auto dst = index_data[e + E];

        for (auto k = 0; k < K; ++k) {
            if (max_indices_data[dst * K + k] == src) { // 检查是否是贡献最大的元素
                scalar_t weighted_value = weight_data[e] * grad_data[dst * K + k];
                #ifdef COMPILE_WITH_OMP
                #pragma omp critical
                #endif
                {
                    out_data[src * K + k] += weighted_value;
                }
            }
        }
    }

    return out;
}
