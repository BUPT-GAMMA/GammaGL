#include "../../extensions.h"
#include "segment_csr.h"

PYBIND11_MODULE(_segment, m) {
  m.doc() = "gammagl segment ops";

  m.def("c_segment_csr_cpu", &segment_csr_cpu);
}