#include "spmm_mean_cpu.h"
#include <torch/torch.h>

std::tuple<torch::Tensor, torch::Tensor> spmm_mean_cpu_forward(torch::Tensor &index, torch::Tensor &weight, torch::Tensor &x) {
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    if (!weight.is_contiguous()) {
        weight = weight.contiguous();
    }
    if (!index.is_contiguous()) {
        index = index.contiguous();
    }
    torch::Tensor out = torch::zeros_like(x, x.options());
    auto E = index.size(1);
    auto K = x.numel() / x.size(0);

    auto index_data = index.data_ptr<int64_t>();
    using scalar_t = float;
    auto x_data = x.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    auto weight_data = weight.data_ptr<scalar_t>();

    // 创建一个张量来存储每个节点的收到的消息数量(入度)
    torch::Tensor messages_count = torch::zeros(x.size(0), torch::kInt64);
    auto messages_count_data = messages_count.data_ptr<int64_t>();

    // 加权求和
#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
    for (auto e = 0; e < E; ++e) {
        auto src = index_data[e];
        auto dst = index_data[e + E];
        messages_count_data[dst]++;

        for (auto k = 0; k < K; ++k) {
#ifdef COMPILE_WITH_OMP
#pragma omp atomic
#endif
            out_data[dst * K + k] += weight_data[e] * x_data[src * K + k];
        }
    }

    // 对每个节点的特征进行平均
#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
    for (auto n = 0; n < x.size(0); ++n) {
        auto msg_count_val = messages_count_data[n];
        if (msg_count_val > 0) {
            for (auto k = 0; k < K; ++k) {
                out_data[n * K + k] /= static_cast<scalar_t>(msg_count_val);
            }
        }
    }

    return std::make_tuple(out, messages_count);
}


torch::Tensor spmm_mean_cpu_backward(torch::Tensor &index, torch::Tensor &weight, torch::Tensor &grad, torch::Tensor &messages_count) {
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
    auto K = grad.numel() / grad.size(0);

    auto index_data = index.data_ptr<int64_t>();
    using scalar_t = float;
    auto grad_data = grad.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    auto weight_data = weight.data_ptr<scalar_t>();

    // 计算反向传播的梯度
#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
    for (auto e = 0; e < E; ++e) {
        auto src = index_data[e];
        auto dst = index_data[e + E];

        for (auto k = 0; k < K; ++k) {
            auto grad_contribution = grad_data[dst * K + k] / messages_count[dst].item<int64_t>() * weight_data[e];
#ifdef COMPILE_WITH_OMP
#pragma omp atomic
#endif
            out_data[src * K + k] += grad_contribution;
        }
    }
    return out;
}