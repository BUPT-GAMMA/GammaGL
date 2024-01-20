#include <torch/torch.h>


class SpMMSum : public torch::autograd::Function<SpMMSum> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor index,
                               torch::Tensor weight, torch::Tensor x);
  static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext *ctx, 
                                            std::vector<torch::Tensor> grad_outs);
};

class SpMMMean : public torch::autograd::Function<SpMMMean> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor index,
                               torch::Tensor weight, torch::Tensor x);
  static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext *ctx, 
                                             std::vector<torch::Tensor> grad_outs);
};

class SpMMMax : public torch::autograd::Function<SpMMMax> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor index,
                               torch::Tensor weight, torch::Tensor x);
  static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext *ctx, 
                                             std::vector<torch::Tensor> grad_outs);
};
