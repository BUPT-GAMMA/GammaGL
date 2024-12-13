/**
 * @Description TODO
 * @Author WuJing
 * @Created 2023/4/4
 */

#include "neighbor_sample.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

template <class T>
py::dict from_dict(phmap::flat_hash_map<string, vector<T>> &map) {
  py::dict dict;
  for (auto &kv : map) {
    dict[py::str(kv.first)] = kv.second;
  }
  return dict;
}

py::list neighbor_sample(
    Tensor colptr, Tensor row, Tensor input_node,
    const vector<int64_t> &num_neighbors, bool replace, bool directed) {
  // Initialize some data structures for the sampling process:
  vector<int64_t> samples;
  phmap::flat_hash_map<int64_t, int64_t> to_local_node;

  auto colptr_data = colptr.unchecked();
  auto row_data = row.unchecked();
  auto input_node_data = input_node.unchecked();

  for (int64_t i = 0; i < input_node_data.size(); i++) {
    const int64_t &v = input_node_data(i);
    samples.push_back(v);
    to_local_node.insert({v, i});
  }

  vector<int64_t> rows, cols, edges;

  int64_t begin = 0, end = samples.size();
  for (int64_t ell = 0; ell < (int64_t)num_neighbors.size(); ell++) {
    const auto &num_samples = num_neighbors[ell];
    for (int64_t i = begin; i < end; i++) {
      const auto &w = samples[i];
      const auto &col_start = colptr_data(w);
      const auto &col_end = colptr_data(w + 1);
      const auto col_count = col_end - col_start;

      if (col_count == 0) continue;

      if ((num_samples < 0) || (!replace && (num_samples >= col_count))) {
        for (int64_t offset = col_start; offset < col_end; offset++) {
          const int64_t &v = row_data(offset);
          const auto res = to_local_node.insert({v, samples.size()});
          if (res.second) samples.push_back(v);
          if (directed) {
            cols.push_back(i);
            rows.push_back(res.first->second);
            edges.push_back(offset);
          }
        }
      } else if (replace) {
        for (int64_t j = 0; j < num_samples; j++) {
          const int64_t offset = col_start + uniform_randint(col_count);
          const int64_t &v = row_data(offset);
          const auto res = to_local_node.insert({v, samples.size()});
          if (res.second) samples.push_back(v);
          if (directed) {
            cols.push_back(i);
            rows.push_back(res.first->second);
            edges.push_back(offset);
          }
        }
      } else {
        unordered_set<int64_t> rnd_indices;
        for (int64_t j = col_count - num_samples; j < col_count; j++) {
          int64_t rnd = uniform_randint(j);
          if (!rnd_indices.insert(rnd).second) {
            rnd = j;
            rnd_indices.insert(j);
          }
          const int64_t offset = col_start + rnd;
          const int64_t &v = row_data(offset);
          const auto res = to_local_node.insert({v, samples.size()});
          if (res.second) samples.push_back(v);
          if (directed) {
            cols.push_back(i);
            rows.push_back(res.first->second);
            edges.push_back(offset);
          }
        }
      }
    }
    begin = end, end = samples.size();
  }

  if (!directed) {
    phmap::flat_hash_map<int64_t, int64_t>::iterator iter;
    for (int64_t i = 0; i < (int64_t)samples.size(); i++) {
      const auto &w = samples[i];
      const auto &col_start = colptr_data(w);
      const auto &col_end = colptr_data(w + 1);
      for (int64_t offset = col_start; offset < col_end; offset++) {
        const auto &v = row_data(offset);
        iter = to_local_node.find(v);
        if (iter != to_local_node.end()) {
          rows.push_back(iter->second);
          cols.push_back(i);
          edges.push_back(offset);
        }
      }
    }
  }

  // return py::make_tuple(samples, rows, cols, edges);
  py::list res;
  res.append(samples);
  res.append(rows);
  res.append(cols);
  res.append(edges);
  return res;
}

py::list hetero_neighbor_sample(
    const vector<node_t> &node_types, const vector<edge_t> &edge_types,
    const unordered_map<rel_t, tensor> &colptr_dict,
    const unordered_map<rel_t, tensor> &row_dict,
    const unordered_map<node_t, tensor> &input_node_dict,
    const unordered_map<rel_t, tensor> &num_neighbors_dict, int64_t num_hops,
    bool replace = false, bool directed = false) {
  // Create a mapping to convert single string relations to edge type triplets:
  //    phmap::flat_hash_map<py::str, edge_t> to_edge_type;

  phmap::flat_hash_map<rel_t, edge_t> to_edge_type;
  for (const auto &k : edge_types) {
    to_edge_type[k[0] + "__" + k[1] + "__" + k[2]] = k;
  }

  // Initialize some data structures for the sampling process:
  phmap::flat_hash_map<node_t, vector<int64_t>> samples_dict;
  phmap::flat_hash_map<node_t, phmap::flat_hash_map<int64_t, int64_t>>
      to_local_node_dict;
  for (const auto &node_type : node_types) {
    samples_dict[node_type];
    to_local_node_dict[node_type];
  }

  phmap::flat_hash_map<rel_t, vector<int64_t>> rows_dict, cols_dict, edges_dict;
  for (const auto &kv : colptr_dict) {
    const auto &rel_type = kv.first;
    rows_dict[rel_type];
    cols_dict[rel_type];
    edges_dict[rel_type];
  }

  // Add the input nodes to the output nodes:
  for (const auto &kv : input_node_dict) {
    const auto &node_type = kv.first;
    const tensor &input_node = kv.second;
    const auto input_node_data = input_node.unchecked();

    auto &samples = samples_dict.at(node_type);
    auto &to_local_node = to_local_node_dict.at(node_type);
    for (int64_t i = 0; i < input_node_data.size(); i++) {
      const auto &v = input_node_data(i);
      samples.push_back(v);
      to_local_node.insert({v, i});
    }
  }

  phmap::flat_hash_map<node_t, pair<int64_t, int64_t>> slice_dict;
  for (const auto &kv : samples_dict)
    slice_dict[kv.first] = {0, kv.second.size()};

  vector<rel_t> all_rel_types;
  for (const auto &kv : num_neighbors_dict) {
    all_rel_types.push_back(kv.first);
  }
  std::sort(all_rel_types.begin(), all_rel_types.end());

  for (int64_t ell = 0; ell < num_hops; ell++) {
    for (const auto &rel_type : all_rel_types) {
      const auto &edge_type = to_edge_type[rel_type];
      const auto &src_node_type = edge_type[0];
      const auto &dst_node_type = edge_type[2];
      const auto num_samples = num_neighbors_dict.at(rel_type).unchecked()(ell);
      const auto &dst_samples = samples_dict.at(dst_node_type);
      auto &src_samples = samples_dict.at(src_node_type);
      auto &to_local_src_node = to_local_node_dict.at(src_node_type);

      const tensor &colptr = colptr_dict.at(rel_type);
      const auto colptr_data = colptr.unchecked();
      const tensor &row = row_dict.at(rel_type);
      const auto row_data = row.unchecked();

      auto &rows = rows_dict.at(rel_type);
      auto &cols = cols_dict.at(rel_type);
      auto &edges = edges_dict.at(rel_type);

      // For temporal sampling, sampled nodes cannot have a timestamp greater
      // than the timestamp of the root nodes:

      const auto &begin = slice_dict.at(dst_node_type).first;
      const auto &end = slice_dict.at(dst_node_type).second;
      for (int64_t i = begin; i < end; i++) {
        const auto &w = dst_samples[i];
        const auto &col_start = colptr_data(w);
        const auto &col_end = colptr_data(w + 1);
        const auto col_count = col_end - col_start;

        if (col_count == 0) continue;

        if ((num_samples < 0) || (!replace && (num_samples >= col_count))) {
          // Select all neighbors:
          for (int64_t offset = col_start; offset < col_end; offset++) {
            const int64_t &v = row_data(offset);
            const auto res = to_local_src_node.insert({v, src_samples.size()});
            if (res.second) src_samples.push_back(v);
            if (directed) {
              cols.push_back(i);
              rows.push_back(res.first->second);
              edges.push_back(offset);
            }
          }
        } else if (replace) {
          // Sample with replacement:
          int64_t num_neighbors = 0;
          while (num_neighbors < num_samples) {
            const int64_t offset = col_start + uniform_randint(col_count);
            const int64_t &v = row_data(offset);
            const auto res = to_local_src_node.insert({v, src_samples.size()});
            if (res.second) src_samples.push_back(v);
            if (directed) {
              cols.push_back(i);
              rows.push_back(res.first->second);
              edges.push_back(offset);
            }
            num_neighbors += 1;
          }
        } else {
          unordered_set<int64_t> rnd_indices;
          for (int64_t j = col_count - num_samples; j < col_count; j++) {
            int64_t rnd = uniform_randint(j);
            if (!rnd_indices.insert(rnd).second) {
              rnd = j;
              rnd_indices.insert(j);
            }
            const int64_t offset = col_start + rnd;
            const int64_t &v = row_data(offset);
            const auto res = to_local_src_node.insert({v, src_samples.size()});
            if (res.second) src_samples.push_back(v);
            if (directed) {
              cols.push_back(i);
              rows.push_back(res.first->second);
              edges.push_back(offset);
            }
          }
        }
      }
    }

    for (const auto &kv : samples_dict)
      slice_dict[kv.first] = {slice_dict.at(kv.first).second, kv.second.size()};
  }

  // Temporal sample disable undirected
  if (!directed) {  // Construct the subgraph among the sampled nodes:
    phmap::flat_hash_map<int64_t, int64_t>::iterator iter;
    for (const auto &kv : colptr_dict) {
      const auto &rel_type = kv.first;
      const auto &edge_type = to_edge_type[rel_type];
      const auto &src_node_type = edge_type[0];
      const auto &dst_node_type = edge_type[2];
      const auto &dst_samples = samples_dict.at(dst_node_type);
      auto &to_local_src_node = to_local_node_dict.at(src_node_type);

      const auto colptr_data = ((tensor)kv.second).unchecked();
      const auto row_data = ((tensor)row_dict.at(rel_type)).unchecked();

      auto &rows = rows_dict.at(rel_type);
      auto &cols = cols_dict.at(rel_type);
      auto &edges = edges_dict.at(rel_type);

      for (int64_t i = 0; i < (int64_t)dst_samples.size(); i++) {
        const auto &w = dst_samples[i];
        const auto &col_start = colptr_data(w);
        const auto &col_end = colptr_data(w + 1);
        for (int64_t offset = col_start; offset < col_end; offset++) {
          const auto &v = row_data(offset);
          iter = to_local_src_node.find(v);
          if (iter != to_local_src_node.end()) {
            rows.push_back(iter->second);
            cols.push_back(i);
            edges.push_back(offset);
          }
        }
      }
    }
  }

  // vector<py::dict> res;
  py::dict samples_res_dict;
  for (auto &kv : samples_dict) {
    samples_res_dict[py::str(kv.first)] = kv.second;
  }

  // return py::make_tuple(
  //         from_dict<int64_t>(samples_dict),
  //         from_dict<int64_t>(rows_dict),
  //         from_dict<int64_t>(cols_dict),
  //         from_dict<int64_t>(edges_dict)
  // );
  py::list res;
  res.append(from_dict<int64_t>(samples_dict));
  res.append(from_dict<int64_t>(rows_dict));
  res.append(from_dict<int64_t>(cols_dict));
  res.append(from_dict<int64_t>(edges_dict));

  return res;
}

PYBIND11_MODULE(_neighbor_sample, m) {
  m.doc() = "gammagl sparse neighbor_sample";
  m.def("c_neighbor_sample", &neighbor_sample);
  m.def("c_hetero_neighbor_sample", &hetero_neighbor_sample);
}