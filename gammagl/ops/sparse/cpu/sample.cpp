/**
 * @Description TODO
 * @Author WuJing
 * @Created 2023/4/10
 */

#include "sample.h"

// Returns `rowptr`, `col`, `n_id`, `e_id`
py::list sample_adj(
    Tensor rowptr, Tensor col, Tensor idx, int64_t num_neighbors,
    bool replace) {
  assert(idx.ndim() == 1);

  auto rowptr_data = rowptr.unchecked();
  auto col_data = col.unchecked();
  auto idx_data = idx.unchecked();

  Tensor out_rowptr(idx.size() + 1);
  auto out_rowptr_data = out_rowptr.mutable_unchecked();
  out_rowptr_data(0) = 0;

  std::vector<std::vector<std::tuple<int64_t, int64_t>>> cols;  // col, e_id
  std::vector<int64_t> n_ids;
  std::unordered_map<int64_t, int64_t> n_id_map;

  int64_t i;
  for (int64_t n = 0; n < idx.size(); n++) {
    i = idx_data(n);
    cols.push_back(std::vector<std::tuple<int64_t, int64_t>>());
    n_id_map[i] = n;
    n_ids.push_back(i);
  }

  int64_t n, c, e, row_start, row_end, row_count;

  if (num_neighbors < 0) {  // No sampling
    for (int64_t i = 0; i < idx.size(); i++) {
      n = idx_data(i);
      row_start = rowptr_data(n), row_end = rowptr_data(n + 1);
      row_count = row_end - row_start;

      for (int64_t j = 0; j < row_count; j++) {
        e = row_start + j;
        c = col_data(e);

        if (n_id_map.count(c) == 0) {
          n_id_map[c] = n_ids.size();
          n_ids.push_back(c);
        }
        cols[i].push_back(std::make_tuple(n_id_map[c], e));
      }
      out_rowptr_data(i + 1) = out_rowptr_data(i) + cols[i].size();
    }
  } else if (replace) {
    for (int64_t i = 0; i < idx.size(); i++) {
      n = idx_data(i);
      row_start = rowptr_data(n), row_end = rowptr_data(n + 1);
      row_count = row_end - row_start;

      if (row_count > 0) {
        for (int64_t j = 0; j < num_neighbors; j++) {
          e = row_start + uniform_randint(row_count);
          c = col_data(e);
          if (n_id_map.count(c) == 0) {
            n_id_map[c] = n_ids.size();
            n_ids.push_back(c);
          }
          cols[i].push_back(std::make_tuple(n_id_map[c], e));
        }
      }
      out_rowptr_data(i + 1) = out_rowptr_data(i) + cols[i].size();
    }
  } else {
    for (int64_t i = 0; i < idx.size(); i++) {
      n = idx_data(i);
      row_start = rowptr_data(n), row_end = rowptr_data(n + 1);
      row_count = row_end - row_start;

      std::unordered_set<int64_t> perm;
      if (row_count <= num_neighbors) {
        for (int64_t j = 0; j < row_count; j++) perm.insert(j);
      } else {
        for (int64_t j = row_count - num_neighbors; j < row_count; j++) {
          if (!perm.insert(uniform_randint(j)).second) perm.insert(j);
        }
      }
      for (const int64_t &p : perm) {
        e = row_start + p;
        c = col_data(e);

        if (n_id_map.count(c) == 0) {
          n_id_map[c] = n_ids.size();
          n_ids.push_back(c);
        }
        cols[i].push_back(std::make_tuple(n_id_map[c], e));
      }
      out_rowptr_data(i + 1) = out_rowptr_data(i) + cols[i].size();
    }
  }

  int64_t N = n_ids.size();
  Tensor out_n_id(N, n_ids.data());

  int64_t E = out_rowptr_data(idx.size());
  Tensor out_col(E);
  auto out_col_data = out_col.mutable_unchecked();
  Tensor out_e_id(E);
  auto out_e_id_data = out_e_id.mutable_unchecked();

  i = 0;
  for (std::vector<std::tuple<int64_t, int64_t>> &col_vec : cols) {
    std::sort(
        col_vec.begin(), col_vec.end(),
        [](const std::tuple<int64_t, int64_t> &a,
           const std::tuple<int64_t, int64_t> &b) -> bool {
          return std::get<0>(a) < std::get<0>(b);
        });
    for (const std::tuple<int64_t, int64_t> &value : col_vec) {
      out_col_data(i) = std::get<0>(value);
      out_e_id_data(i) = std::get<1>(value);
      i += 1;
    }
  }

  py::list res;
  // return py::make_tuple(out_rowptr, out_col, out_n_id, out_e_id);

  res.append(out_rowptr);
  res.append(out_col);
  res.append(out_n_id);
  res.append(out_e_id);

  return res;
}

PYBIND11_MODULE(_sample, m) {
  m.doc() = "gammagl sparse sample";
  m.def("c_sample_adj", &sample_adj);
}
