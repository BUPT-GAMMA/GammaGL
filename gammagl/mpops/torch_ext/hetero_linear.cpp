#include "cpu/grouped_gemm.h"
#include <ATen/ATen.h>
#include <torch/autograd.h>
#include <torch/extension.h>
#include <torch/torch.h>

using torch::autograd::Variable;
using torch::autograd::variable_list;

std::vector<at::Tensor> concat(std::vector<at::Tensor> t1,
                               std::vector<at::Tensor> t2) {
  for (size_t i = 0; i < t2.size(); ++i) {
    t1.push_back(t2[i]);
  }
  return t1;
}

void fill_tensor_args(std::vector<torch::TensorArg> &args,
                      const torch::TensorList &tensors, const std::string &name,
                      int pos) {

  args.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto full_name = name + "[" + std::to_string(i) + "]";
    args.emplace_back(tensors[i], full_name.c_str(), pos);
  }
}

void fill_tensor_args(std::vector<torch::TensorArg> &args,
                      const c10::Dict<std::string, torch::Tensor> &tensors,
                      const std::string &name, int pos) {
  args.reserve(tensors.size());
  for (const auto &kv : tensors) {
    const auto full_name = name + "[" + kv.key() + "]";
    args.emplace_back(kv.value(), full_name.c_str(), pos);
  }
}

// Performs matrix multiplication across list of elements.
std::vector<at::Tensor> grouped_matmul(const at::TensorList input,
                                       const at::TensorList other) {
  TORCH_CHECK(input.size() == other.size(),
              "Number of 'input' tensors must match number of 'other' tensors");

  std::vector<at::TensorArg> input_args;
  std::vector<at::TensorArg> other_args;
  fill_tensor_args(input_args, input, "input", 0);
  fill_tensor_args(other_args, other, "other", 1);
  at::CheckedFrom c{"grouped_matmul"};

  at::checkAllDefined(c, input_args);
  at::checkAllDefined(c, other_args);
  at::checkAllSameType(c, input_args);
  at::checkAllSameType(c, other_args);
  at::checkSameType(c, input_args[0], other_args[0]);
  for (size_t i = 0; i < input.size(); ++i) {
    at::checkDim(c, input_args[i], 2);
    at::checkDim(c, other_args[i], 2);
    at::checkSize(c, other_args[i], 0, input_args[i]->size(-1));
  }

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::grouped_matmul", "")
                       .typed<decltype(grouped_matmul)>();
  return op.call(input, other);
}

// Performs matrix multiplication according to segments.
at::Tensor segment_matmul(const at::Tensor &input, const at::Tensor &ptr,
                          const at::Tensor &other) {
  at::TensorArg input_arg{input, "input", 0};
  at::TensorArg ptr_arg{ptr, "ptr", 1};
  at::TensorArg other_arg{other, "other", 2};
  at::CheckedFrom c{"segment_matmul"};

  at::checkAllDefined(c, {input_arg, ptr_arg, other_arg});
  at::checkSameType(c, input_arg, other_arg);
  at::checkDim(c, input_arg, 2);
  at::checkDim(c, ptr_arg, 1);
  at::checkDim(c, other_arg, 3);
  at::checkSize(c, other_arg, 1, input_arg->size(-1));
  at::checkNumel(c, ptr_arg, other_arg->size(0) + 1);

  return segment_matmul_kernel_cpu(input, ptr, other);
}

class GroupedMatmul : public torch::autograd::Function<GroupedMatmul> {
public:
  static variable_list forward(torch::autograd::AutogradContext *ctx,
                               const variable_list input,
                               const variable_list other) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto out = grouped_matmul(input, other);
    variable_list input_and_other = concat(input, other);
    ctx->save_for_backward(input_and_other);
    return out;
  }

  static variable_list backward(torch::autograd::AutogradContext *ctx,
                                variable_list grad_outs) {
    auto input_and_other = ctx->get_saved_variables();
    int input_len = input_and_other.size() / 2;
    variable_list input(input_and_other.begin(),
                        input_and_other.begin() + input_len);
    variable_list other(input_and_other.begin() + input_len,
                        input_and_other.end());

    // We assume entire input variable list either requires grad or does not:
    variable_list other_grad;
    if (torch::autograd::any_variable_requires_grad(other)) {
      for (size_t i = 0; i < input.size(); ++i) {
        other[i] = other[i].transpose(-2, -1);
        other_grad.push_back(torch::matmul(grad_outs[i], other[i]));
      }
    } else {
      for (size_t i = 0; i < other.size(); ++i)
        other_grad.push_back(Variable());
    }

    variable_list input_grad;
    if (torch::autograd::any_variable_requires_grad(input)) {
      for (size_t i = 0; i < input.size(); ++i)
        input[i] = input[i].transpose(-2, -1);
      input_grad = grouped_matmul(input, grad_outs);
    } else {
      for (size_t i = 0; i < input.size(); ++i)
        input_grad.push_back(Variable());
    }
    return concat(input_grad, other_grad);
  }
};

class SegmentMatmul : public torch::autograd::Function<SegmentMatmul> {
public:
  static variable_list forward(torch::autograd::AutogradContext *ctx,
                               const Variable &input, const at::Tensor &ptr,
                               const Variable &other) {
    at::AutoDispatchBelowADInplaceOrView g;
    Variable out = segment_matmul(input, ptr, other);
    ctx->save_for_backward({input, ptr, other});
    return {out};
  }

  static variable_list backward(torch::autograd::AutogradContext *ctx,
                                variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto input = saved[0], ptr = saved[1], other = saved[2];

    auto input_grad = Variable();
    if (torch::autograd::any_variable_requires_grad({input})) {
      auto other_t = other.transpose(-2, -1);
      input_grad = segment_matmul(grad_out, ptr, other_t);
    }

    auto other_grad = Variable();
    if (torch::autograd::any_variable_requires_grad({other})) {
      auto size = size_from_ptr(ptr).cpu();
      // TODO (matthias) Allow for other types than `int64_t`.
      auto sizes = at::IntArrayRef(size.data_ptr<int64_t>(), size.numel());
      auto input_t = input.transpose(-2, -1);
      variable_list split_input_t =
          input_t.split_with_sizes(/*split_size=*/sizes, /*dim=*/1);
      variable_list grad_out_split =
          grad_out.split_with_sizes(/*split_size=*/sizes, /*dim=*/0);
      variable_list others_grad;
      for (size_t i = 0; i < split_input_t.size(); ++i)
        others_grad.push_back(
            torch::matmul(split_input_t[i], grad_out_split[i]));
      other_grad = at::stack(others_grad);
    }

    return {input_grad, Variable(), other_grad};
  }
};

at::Tensor segment_matmul_autograd(const at::Tensor &input,
                                   const at::Tensor &ptr,
                                   const at::Tensor &other) {
  return SegmentMatmul::apply(input, ptr, other)[0];
}

PYBIND11_MODULE(torch_hetero_linear, m) {
  m.def("segment_matmul", segment_matmul_autograd);
  m.def("grouped_matmul", grouped_matmul);
  // m.def("spmm_max", spmm_max);
  // ...
}