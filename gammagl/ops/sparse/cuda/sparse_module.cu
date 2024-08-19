#include "../../extensions.h"
#include "convert.h"
#include "neighbor_sample.h"
// #include "saint.h"
// #include "sample.h"
#include "sparse.h"

PYBIND11_MODULE(_sparse_cuda, m) {
  m.doc() = "gammagl sparse ops";

  m.def("cuda_torch_set_to_one", &torch_cuda_set_to_one);
  m.def("cuda_torch_ind2ptr", &torch_cuda_ind2ptr);
  m.def("cuda_torch_ptr2ind", &torch_cuda_ptr2ind);
  m.def("cuda_torch_neighbor_sample", &torch_cu_neighbor_sample);
  m.def("cuda_torch_sample_adj", &torch_cu_sample_adj);
  // m.def("c_hetero_neighbor_sample", &hetero_neighbor_sample);
  // m.def("c_random_walk", &random_walk);
  // m.def("c_saint_subgraph", &subgraph);
  // m.def("c_sample_adj", &sample_adj);
}
