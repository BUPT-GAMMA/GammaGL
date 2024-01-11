#include "spmm_mean_cpu.h"
#include <torch/torch.h>

torch::Tensor spmm_mean_cpu_forward(torch::Tensor &index, torch::Tensor &weight, torch::Tensor &x) {
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

    // 创建一个向量来存储每个节点的收到的消息数量(入度)
    std::vector<int> messages_count(x.size(0), 0);
    // 加权求和
#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
    for (auto e = 0; e < E; ++e) {
        auto src = index_data[e];
        auto dst = index_data[e + E];
        messages_count[dst]++;

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
        if (messages_count[n] > 0) {
            for (auto k = 0; k < K; ++k) {
                out_data[n * K + k] /= messages_count[n];
            }
        }
    }

    return out;
}


torch::Tensor spmm_mean_cpu_backward(torch::Tensor &index, torch::Tensor &weight, torch::Tensor &grad) {
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

    // 创建一个向量来存储每个节点的出度
    std::vector<int> outdegree_count(grad.size(0), 0);
    // 计算反向传播的梯度
#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
    for (auto e = 0; e < E; ++e) {
        auto src = index_data[e];
        auto dst = index_data[e + E];
        outdegree_count[src]++;

        for (auto k = 0; k < K; ++k) {
#ifdef COMPILE_WITH_OMP
#pragma omp atomic
#endif
            out_data[src * K + k] += weight_data[e] * grad_data[dst * K + k];
        }
    }

#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
    for (auto n = 0; n < grad.size(0); ++n) {
        if (outdegree_count[n] > 0) {
            for (auto k = 0; k < K; ++k) {
                out_data[n * K + k] /= outdegree_count[n];
            }
        }
    }

    return out;
}