/*
 * @Description: TODO
 * @Author: WuJing
 * @created: 2023-04-11
 */
/**
 * @Description TODO
 * @Author WuJing
 * @Created 2023/4/4
 */

#include "saint.h"

#include "sparse_utils.h"

py::list subgraph(Tensor idx, Tensor rowptr, Tensor row, Tensor col) {
  assert(idx.ndim() == 1);
  assert(rowptr.ndim() == 1);
  assert(col.ndim() == 1);

  // TODO need to optimize
  //    py::gil_scoped_acquire acquire{};
  //    py::module_ pfc = py::module_::import("gammagl.sparse.python_for_c");
  //
  //    py::object new_func = pfc.attr("new");
  //    auto assoc = py::cast<Tensor>(new_func(row.shape(0) - 1, -1));
  //
  //    py::object arange = pfc.attr("arange");
  //    auto arange_arr = py::cast<Tensor>(arange(idx.shape(0)));
  //    py::object put = pfc.attr("put");
  //    assoc = py::cast<Tensor>(put(assoc, idx, arange_arr));

  // assoc = python_for_c<double_t>("put", assoc, idx,
  // np_func<int64_t>("arange", 10));
  auto assoc = py_helper<int64_t>("new", rowptr.shape(0) - 1, -1);
  assoc = py_helper<int64_t>(
      "put", assoc, idx, np_func<int64_t>("arange", idx.shape(0)));

  //    py::gil_scoped_release release{};

  auto idx_data = idx.unchecked();
  auto rowptr_data = rowptr.unchecked();
  auto col_data = col.unchecked();
  auto assoc_data = assoc.unchecked();

  std::vector<int64_t> rows, cols, indices;

  int64_t v, w, w_new, row_start, row_end;
  for (int64_t v_new = 0; v_new < idx.shape(0); v_new++) {
    v = idx_data(v_new);
    row_start = rowptr_data(v);
    row_end = rowptr_data(v + 1);

    for (int64_t j = row_start; j < row_end; j++) {
      w = col_data(j);
      w_new = assoc_data(w);
      if (w_new > -1) {
        rows.push_back(v_new);
        cols.push_back(w_new);
        indices.push_back(j);
      }
    }
  }

  int64_t length = rows.size();

  py::list res;
  res.append(Tensor(length, rows.data()));
  res.append(Tensor(length, cols.data()));
  res.append(Tensor(length, indices.data()));

  return res;

  // return py::make_tuple(
  //         Tensor(length, rows.data()),
  //         Tensor(length, cols.data()),
  //         Tensor(length, indices.data())
  // );

  //    row = torch::from_blob(rows.data(), {length}, row.options()).clone();
  //    col = torch::from_blob(cols.data(), {length}, row.options()).clone();
  //    idx = torch::from_blob(indices.data(), {length}, row.options()).clone();

  //    return std::make_tuple(row, col, idx);
}

PYBIND11_MODULE(_saint, m) {
  m.doc() = "gammagl sparse saint";
  m.def("c_saint_subgraph", &subgraph);
}