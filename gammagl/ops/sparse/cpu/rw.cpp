/**
 * @Description TODO
 * @Author WuJing
 * @Created 2023/4/5
 */

#include "sparse.h"

Tensor random_walk(
    Tensor rowptr, Tensor col, Tensor start, int64_t walk_length) {
  assert(rowptr.ndim() == 1);
  assert(col.ndim() == 1);
  assert(start.ndim() == 1);

  //    vector<int64_t> size_list{(int64_t) start.size(), walk_length};
  //    auto rand = rand_tensor<float_t>({start.size(), walk_length});

  py::array_t<double_t> randarr{{start.size(), walk_length}};
  auto rand_data = randarr.mutable_unchecked();
  for (int i = 0; i < randarr.shape(0); i++) {
    for (int j = 0; j < randarr.shape(1); j++) {
      srand(time(0));
      rand_data(i, j) =
          static_cast<double_t>(rand()) / static_cast<double_t>(RAND_MAX);
    }
  }

  auto L = walk_length + 1;

  Tensor out({start.size(), L});

  auto rowptr_data = rowptr.unchecked();
  auto col_data = col.unchecked();
  auto start_data = start.unchecked();
  //    auto rand_data = rand.unchecked();
  auto out_data = out.mutable_unchecked();

  for (auto n = 0; n < start.shape(0); n++) {
    auto cur = start_data(n);
    out_data(n, 0) = cur;

    int64_t row_start, row_end;
    for (auto l = 0; l < walk_length; l++) {
      row_start = rowptr_data(cur);
      row_end = rowptr_data(cur + 1);

      cur = col_data(
          row_start + int64_t(rand_data(n, l) * (row_end - row_start)));
      out_data(n, l + 1) = cur;
    }
  }

  return out;
}

PYBIND11_MODULE(_rw, m) {
  m.doc() = "gammagl random_walk method";
  m.def("c_random_walk", &random_walk);
}