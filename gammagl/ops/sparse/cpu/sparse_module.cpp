#include "../../extensions.h"
#include "convert.h"
#include "neighbor_sample.h"
#include "saint.h"
#include "sample.h"
#include "sparse.h"

PYBIND11_MODULE(_sparse, m) {
  m.doc() = "gammagl sparse ops";

  m.def("c_ind2ptr", &ind2ptr);
  m.def("c_ptr2ind", &ptr2ind);
  m.def("c_set_to_one", &set_to_one);
  m.def("c_neighbor_sample", &neighbor_sample);
  m.def("c_hetero_neighbor_sample", &hetero_neighbor_sample);
  m.def("c_random_walk", &random_walk);
  m.def("c_saint_subgraph", &subgraph);
  m.def("c_sample_adj", &sample_adj);
}
