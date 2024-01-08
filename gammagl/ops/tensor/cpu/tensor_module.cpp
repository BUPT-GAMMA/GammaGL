#include "../../extensions.h"
#include "unique.h"

PYBIND11_MODULE(_tensor, m) {
  m.doc() = "gammagl tensor ops";
  m.def(
      "c_unique", &unique_impl, "input"_a, "sorted"_a = true,
      "return_inverse"_a = false, "return_counts"_a = false);
}
