#include <assert.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <vector>

#ifdef COMPILE_WITH_CUDA
#include "segment_max_cuda.h"
#endif

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

std::tuple<torch::Tensor, torch::Tensor> segment_cpu(torch::Tensor src,
                                                     torch::Tensor index,
                                                     int64_t N) {
  TORCH_CHECK(src.device().is_cpu(), "src must be CPU tensor");
  TORCH_CHECK(index.device().is_cpu(), "index must be CPU tensor");
  TORCH_CHECK_INDEX(src.dim() == 2, "src dimension should be 2, but got ", src.dim());
  TORCH_CHECK_INDEX(index.dim() == 1, "index dimension should be 1, but got ", index.dim());
  TORCH_CHECK_INDEX(src.size(0) == index.size(0), "fisrt dimension of src and index should be same");

  src = src.contiguous();

  auto sizes = src.sizes().vec();
  sizes[0] = N > *index.max().data_ptr<int64_t>()
                 ? N
                 : *index.max().data_ptr<int64_t>();
  torch::Tensor out = torch::empty(sizes, src.options());
  torch::Tensor arg_out = torch::full_like(out, 0, index.options());
  if (src.numel() == 0) {
    out.fill_(0);
    return std::make_tuple(out, arg_out);
  }

  out.fill_(std::numeric_limits<int64_t>::lowest());
  auto E = src.size(0);
  auto K = src.size(1);
  auto index_data = index.data_ptr<int64_t>();
  auto arg_out_data = arg_out.data_ptr<int64_t>();

  // AT_DISPATCH_ALL_TYPES(src.scalar_type(), "__ops_name", [&] {
    using scalar_t = float;
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    int64_t idx;
#ifdef COMPILE_WITH_OMP
#pragma omp parallel for
#endif
    for (auto e = 0; e < E; ++e) {
      idx = index_data[e];
      for (auto k = 0; k < K; ++k) {
        if (out_data[idx * K + k] < src_data[e * K + k]) {
#ifdef COMPILE_WITH_OMP
#pragma omp atomic write
#endif
          out_data[idx * K + k] = src_data[e * K + k];
          arg_out_data[idx * K + k] = e;
        }
      }
    }
    out.masked_fill_(out == std::numeric_limits<int64_t>::lowest(), (scalar_t)0);
  // });

  return std::make_tuple(out, arg_out);
}

inline std::vector<int64_t> list2vec(const c10::List<int64_t> list) {
  std::vector<int64_t> result;
  result.reserve(list.size());
  for (size_t i = 0; i < list.size(); ++i) result.push_back(list[i]);
  return result;
}

class SegmentMax : public torch::autograd::Function<SegmentMax> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, Variable src,
                               Variable index, int64_t N) {
    ctx->saved_data["src_shape"] = src.sizes();
    auto result = segment_cpu(src, index, N);
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result);
    ctx->save_for_backward({index, arg_out});
    ctx->mark_non_differentiable({arg_out});
    return out;
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto index = saved[0];
    auto arg_out = saved[1];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    src_shape[0] += 1;
    auto grad_in = torch::zeros(src_shape, grad_out.options());
    grad_in.scatter_(0, arg_out, grad_out);
    grad_in = grad_in.narrow(0, 0, src_shape[0] - 1);
    return {grad_in, Variable(), Variable()};
  }
};

torch::Tensor segment_max(torch::Tensor src, torch::Tensor index, int64_t N) {
  auto result = SegmentMax::apply(src, index, N);
  return result;
}

TORCH_LIBRARY(mp, m) {
  m.def("segment_max", segment_max);
#ifdef COMPILE_WITH_CUDA
  m.def("segment_max_cuda", segment_max_cuda);
#endif
}