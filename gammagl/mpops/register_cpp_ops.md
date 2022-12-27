Following these steps:  
1. create three files:  
 `segment_op.cpp`,   
 `cpu/segment_op_cpu.cpp`,  
 `cpu/segment_op_cpu.h`.   
If need to support cuda, create two other files:  
 `cuda/segment_op_cuda.cu`,  
 `cuda/segment_op_cuda.h`.

2. `segment_op.cpp` is for API binding and device dispatching.  
    * Following is an API binding example:  
    ```c++
    class SegmentMax : public torch::autograd::Function<SegmentOp> {
    public:
        static torch::Tensor forward(AutogradContext* ctx, torch::Tensor x) {
            // forward
        }

        static torch::Tensor backward(AutogradContext* ctx, 
                                      std::vector<torch::Tensor> grad_outs) {
            // backward(jvp)
        }
    };

    torch::Tensor segment_op(torch::Tensor x) {
        return SegmentOp::apply(x);
    }

    TORCH_LIBRARY(torch_segment, m) {
        m.def("segment_op", segment_op);
    }
    ```

    * Device dispatching function (CPU, CUDA, ROCm ...):   
    ```c++
    torch::Tensor device_dispatch_forward(
        torch::Tensor& x) {
        if (x.is_cpu()) {
            return segment_max_cuda_forward(x);
    #ifdef COMPILE_WITH_CUDA
        } else if (x.is_cuda()) {
            return segment_sum_cuda_forward(x, index, n);
    #endif
        } else {
            AT_ERROR("Tensor device inconsistent error.");
        }
    }
    ```
2. Implemente forward & backward functions in `cpu/segment_op_cpu.cpp`:  
    * `segment_op_cpu_forward` function:
    ```c++
    torch::Tensor segment_max_cpu_forward(torch::Tensor& x) {
        TORCH_CHECK(x.device().is_cpu(), "x must be CPU tensor");
        x = x.contiguous(); // torch Tensor my not be contiguous.
        auto out = torch::zeros(x.sizes(), x.options());
        auto E = x.size(0); // edge num
        auto K = x.size(1); // feature dim
        for (auto e = 0; e < E; ++e) {
            for (auto k = 0; k < K; ++k) {
                out[e * K + k] = op_func(out[e * K + k], x[e * K + k]);
            }
        }
        return out;
    }
    ```

    * `segment_op_cpu_backward` function:
    ```c++
    torch::Tensor segment_max_cpu_forward(torch::Tensor& x,
                                          torch::Tensor& grad_out) {
        TORCH_CHECK(x.device().is_cpu(), "x must be CPU tensor");
        auto grad_x = torch::zeros(x.sizes(), x.options());
        auto E = x.size(0);
        auto K = x.size(1);
        for (auto e = 0; e < E; ++e) {
            for (auto k = 0; k < K; ++k) {
                grad_x[e * K + k] = jvp_func(grad_out[e * K + k]);
            }
        }
        return grad_x;
    }
    ```