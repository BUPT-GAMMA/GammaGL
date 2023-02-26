#include "grouped_gemm_cuda.h"

template <typename GemmKernel>
void run_grouped_gemm(const at::TensorList input,
                      const at::TensorList other,
                      const at::TensorList out) {
  const auto num_matrices = input.size();
  std::vector<at::Tensor> new_input, new_other, new_out;
  std::vector<float*> ptr_A_host(num_matrices);
  std::vector<float*> ptr_B_host(num_matrices);
  std::vector<float*> ptr_C_host(num_matrices);

  for (size_t i = 0; i < num_matrices; ++i) {
    new_input.push_back(input[i].contiguous());
    ptr_A_host[i] = new_input[i].data_ptr<float>();

    new_other.push_back(other[i].contiguous());
    ptr_B_host[i] = new_other[i].data_ptr<float>();

    new_out.push_back(out[i].contiguous());
    ptr_C_host[i] = new_out[i].data_ptr<float>();
  }

  cutlass::DeviceAllocation<float*> ptr_A;
  ptr_A.reset(num_matrices);
  ptr_A.copy_from_host(ptr_A_host.data());

  cutlass::DeviceAllocation<float*> ptr_B;
  ptr_B.reset(num_matrices);
  ptr_B.copy_from_host(ptr_B_host.data());

  cutlass::DeviceAllocation<float*> ptr_C;
  ptr_C.reset(num_matrices);
  ptr_C.copy_from_host(ptr_C_host.data());

  std::vector<cutlass::gemm::GemmCoord> all_problems(num_matrices);
  std::vector<int64_t> ld_A_host(num_matrices);
  std::vector<int64_t> ld_B_host(num_matrices);
  std::vector<int64_t> ld_C_host(num_matrices);
  for (size_t i = 0; i < num_matrices; ++i) {
    auto m = new_input[i].size(0), k = new_input[i].size(1),
         n = new_out[i].size(1);
    all_problems[i] = cutlass::gemm::GemmCoord(m, n, k);
    ld_A_host[i] = GemmKernel::LayoutA::packed({m, k}).stride(0);
    ld_B_host[i] = GemmKernel::LayoutB::packed({k, n}).stride(0);
    ld_C_host[i] = GemmKernel::LayoutC::packed({m, n}).stride(0);
  }

  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> all_problems_device;
  all_problems_device.reset(num_matrices);
  all_problems_device.copy_from_host(all_problems.data());

  cutlass::DeviceAllocation<int64_t> ld_A;
  ld_A.reset(num_matrices);
  ld_A.copy_from_host(ld_A_host.data());

  cutlass::DeviceAllocation<int64_t> ld_B;
  ld_B.reset(num_matrices);
  ld_B.copy_from_host(ld_B_host.data());

  cutlass::DeviceAllocation<int64_t> ld_C;
  ld_C.reset(num_matrices);
  ld_C.copy_from_host(ld_C_host.data());

  using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
  typename EpilogueOutputOp::Params epilogue_op(1.0, 0.0);

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
  typename GemmGrouped::Arguments args(
      all_problems_device.get(), num_matrices, /*threadblock_count=*/1024,
      epilogue_op, ptr_A.get(), ptr_B.get(), ptr_C.get(), ptr_C.get(),
      ld_A.get(), ld_B.get(), ld_C.get(), ld_C.get());

  GemmGrouped gemm;
  auto status = gemm.initialize(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GroupedGEMM init failed");
  status = gemm.run();
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GroupedGEMM run failed");
}

// Returns the amount of shared memory required per threadblock in
// `GroupedGemmKernel`
template <typename GroupedGemmKernel>
int shared_memory_for_kernel() {
  return int(sizeof(typename GroupedGemmKernel::SharedStorage));
}

// Returns the bytes of shared memory available per SM on the GPU, or -1 on
// error.
cudaDeviceProp get_dev_prop() {
  cudaDeviceProp properties;
  int device_idx;
  cudaError_t result = cudaGetDevice(&device_idx);
  if (result != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(result));
  }

  result = cudaGetDeviceProperties(&properties, device_idx);
  if (result != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(result));
  }
  return properties;
}
cudaDeviceProp props;
bool props_queried = false;

void grouped_matmul_out_cuda_kernel(const at::TensorList input,
                               const at::TensorList other,
                               const at::TensorList out) {
  if (!props_queried) {
    props = get_dev_prop();
    props_queried = true;
  }
  if (props.major < 8) {
    // Compute capability less than that of Ampere. No TF32 available.
    // note: we only support Volta and onwards
    using GemmKernel_Volta = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        float,                                         // Element A
        cutlass::layout::RowMajor,                     // Layout A
        cutlass::ComplexTransform::kNone,              //
        1,                                             // Granularity A
        float,                                         // Element B
        cutlass::layout::RowMajor,                     // Layout B
        cutlass::ComplexTransform::kNone,              //
        1,                                             // Granularity B
        float,                                         // Element C&D
        cutlass::layout::RowMajor,                     // Layout C&D
        float,                                         // Element Accumulator
        cutlass::arch::OpClassSimt,                    // Operator Class Tag
        cutlass::arch::Sm70,                           // Architecture
        cutlass::gemm::GemmShape<128, 64, 8>,          // Threadblock-level Tile
        cutlass::gemm::GemmShape<64, 64, 8>,           // Warp-level Tile
        cutlass::gemm::GemmShape<1, 1, 1>,             // Warp-level Tile
        cutlass::epilogue::thread::LinearCombination<  // Epilogue
            float, 1, float, float>,                   //
        cutlass::gemm::threadblock::                   // Swizzling Operator
        GemmIdentityThreadblockSwizzle<8>,             //
        2,                                             // Stages
        cutlass::arch::OpMultiplyAdd                   // Operation
        >::GemmKernel;
    run_grouped_gemm<GemmKernel_Volta>(input, other, out);
  } else {
    // Compute capability at or beyond that of Ampere. TF32 is available.
    bool use_tf32;
#if TORCH_VERSION_MINOR >= 12 or TORCH_VERSION_MAJOR > 1
    use_tf32 = at::globalContext().float32MatmulPrecision() !=
               at::Float32MatmulPrecision::HIGHEST;
#else
    use_tf32 = at::globalContext().allowTF32CuBLAS();
#endif
    if (use_tf32) {
      // TF32 is enabled
      using DefaultGemmKernel_TF32 =
          typename cutlass::gemm::kernel::DefaultGemmGrouped<
              float,                                   // Element A
              cutlass::layout::RowMajor,               // Layout A
              cutlass::ComplexTransform::kNone,        //
              1,                                       // Granularity A
              float,                                   // Element B
              cutlass::layout::RowMajor,               // Layout B
              cutlass::ComplexTransform::kNone,        //
              1,                                       // Granularity B
              float,                                   // Element C&D
              cutlass::layout::RowMajor,               // Layout C&D
              float,                                   // Element Accumulator
              cutlass::arch::OpClassTensorOp,          // Operator Class Tag
              cutlass::arch::Sm80,                     // Architecture
              cutlass::gemm::GemmShape<256, 128, 32>,  // Threadblock-level Tile
              cutlass::gemm::GemmShape<64, 64, 32>,    // Warp-level Tile
              cutlass::gemm::GemmShape<16, 8, 8>,      // Warp-level Tile
              cutlass::epilogue::thread::LinearCombination<  // Epilogue
                  float, 1, float, float>,                   //
              cutlass::gemm::threadblock::        // Swizzling Operator
              GemmIdentityThreadblockSwizzle<8>,  //
              3,                                  // Stages
              cutlass::arch::OpMultiplyAdd        // Operation
              >::GemmKernel;
      int grouped_shared_mem =
          shared_memory_for_kernel<DefaultGemmKernel_TF32>();
      if (grouped_shared_mem < props.sharedMemPerBlockOptin) {
        // full size GPU
        run_grouped_gemm<DefaultGemmKernel_TF32>(input, other, out);
      } else {
        // Smaller GPU
        using SmallGemmKernel_TF32 =
            typename cutlass::gemm::kernel::DefaultGemmGrouped<
                float,                                  // Element A
                cutlass::layout::RowMajor,              // Layout A
                cutlass::ComplexTransform::kNone,       //
                1,                                      // Granularity A
                float,                                  // Element B
                cutlass::layout::RowMajor,              // Layout B
                cutlass::ComplexTransform::kNone,       //
                1,                                      // Granularity B
                float,                                  // Element C&D
                cutlass::layout::RowMajor,              // Layout C&D
                float,                                  // Element Accumulator
                cutlass::arch::OpClassTensorOp,         // Operator Class Tag
                cutlass::arch::Sm80,                    // Architecture
                cutlass::gemm::GemmShape<128, 64, 32>,  // Threadblock-level
                                                        // Tile
                cutlass::gemm::GemmShape<64, 64, 32>,   // Warp-level Tile
                cutlass::gemm::GemmShape<16, 8, 8>,     // Warp-level Tile
                cutlass::epilogue::thread::LinearCombination<  // Epilogue
                    float, 1, float, float>,                   //
                cutlass::gemm::threadblock::        // Swizzling Operator
                GemmIdentityThreadblockSwizzle<8>,  //
                3,                                  // Stages
                cutlass::arch::OpMultiplyAdd        // Operation
                >::GemmKernel;
        run_grouped_gemm<SmallGemmKernel_TF32>(input, other, out);
      }
    } else {
      // TF32 is manually disabled
      using DefaultGemmKernel_FP32 =
          typename cutlass::gemm::kernel::DefaultGemmGrouped<
              float,                                 // Element A
              cutlass::layout::RowMajor,             // Layout A
              cutlass::ComplexTransform::kNone,      //
              1,                                     // Granularity A
              float,                                 // Element B
              cutlass::layout::RowMajor,             // Layout B
              cutlass::ComplexTransform::kNone,      //
              1,                                     // Granularity B
              float,                                 // Element C&D
              cutlass::layout::RowMajor,             // Layout C&D
              float,                                 // Element Accumulator
              cutlass::arch::OpClassSimt,            // Operator Class Tag
              cutlass::arch::Sm80,                   // Architecture
              cutlass::gemm::GemmShape<128, 64, 8>,  // Threadblock-level Tile
              cutlass::gemm::GemmShape<64, 64, 8>,   // Warp-level Tile
              cutlass::gemm::GemmShape<1, 1, 1>,     // Warp-level Tile
              cutlass::epilogue::thread::LinearCombination<  // Epilogue
                  float, 1, float, float>,                   //
              cutlass::gemm::threadblock::        // Swizzling Operator
              GemmIdentityThreadblockSwizzle<8>,  //
              3,                                  // Stages
              cutlass::arch::OpMultiplyAdd        // Operation
              >::GemmKernel;
      int grouped_shared_mem =
          shared_memory_for_kernel<DefaultGemmKernel_FP32>();
      if (grouped_shared_mem < props.sharedMemPerBlockOptin) {
        // full size GPU
        run_grouped_gemm<DefaultGemmKernel_FP32>(input, other, out);
      } else {
        // Smaller GPU
        using SmallGemmKernel_FP32 =
            typename cutlass::gemm::kernel::DefaultGemmGrouped<
                float,                                // Element A
                cutlass::layout::RowMajor,            // Layout A
                cutlass::ComplexTransform::kNone,     //
                1,                                    // Granularity A
                float,                                // Element B
                cutlass::layout::RowMajor,            // Layout B
                cutlass::ComplexTransform::kNone,     //
                1,                                    // Granularity B
                float,                                // Element C&D
                cutlass::layout::RowMajor,            // Layout C&D
                float,                                // Element Accumulator
                cutlass::arch::OpClassSimt,           // Operator Class Tag
                cutlass::arch::Sm80,                  // Architecture
                cutlass::gemm::GemmShape<64, 64, 8>,  // Threadblock-level
                                                      // Tile
                cutlass::gemm::GemmShape<64, 64, 8>,  // Warp-level Tile
                cutlass::gemm::GemmShape<1, 1, 1>,    // Warp-level Tile
                cutlass::epilogue::thread::LinearCombination<  // Epilogue
                    float, 1, float, float>,                   //
                cutlass::gemm::threadblock::        // Swizzling Operator
                GemmIdentityThreadblockSwizzle<8>,  //
                3,                                  // Stages
                cutlass::arch::OpMultiplyAdd        // Operation
                >::GemmKernel;
        run_grouped_gemm<SmallGemmKernel_FP32>(input, other, out);
      }
    }
  }
}

std::vector<at::Tensor> grouped_matmul_cuda_kernel(const at::TensorList input,
                                              const at::TensorList other) {
  std::vector<at::Tensor> out(input.size());
  for (size_t i = 0; i < input.size(); ++i)
    out[i] = input[i].new_empty({input[i].size(0), other[i].size(-1)});
  grouped_matmul_out_kernel(input, other, out);

  return out;
}

at::Tensor segment_matmul_cuda_kernel(const at::Tensor& input,
                                 const at::Tensor& ptr,
                                 const at::Tensor& other) {
  const auto size = pyg::utils::size_from_ptr(ptr).cpu();
  // TODO (matthias) Allow for other types than `int64_t`.
  const auto sizes = at::IntArrayRef(size.data_ptr<int64_t>(), size.numel());
  const auto out = input.new_empty({input.size(0), other.size(-1)});

  // TODO (matthias) Better handle non-contiguous memory layouts.
  grouped_matmul_out_kernel(
      input.contiguous().split_with_sizes(/*split_size=*/sizes, /*dim=*/0),
      other.contiguous().split(/*split_size=*/1, /*dim=*/0),
      out.split_with_sizes(/*split_size=*/sizes, /*dim=*/0));

  return out;
}