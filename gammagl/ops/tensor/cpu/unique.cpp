/*
 * @Description: TODO
 * @Author: WuJing
 * @created: 2023-04-11
 */

#include "unique.h"

py::list unique_impl(
    const py::array_t<long long> &input, const bool sorted,
    const bool return_inverse, const bool return_counts) {
  const long long *input_data = input.data();

  int numel = *input.shape();

  py::array_t<long long> inverse_indices = py::array_t<long long>(0);
  py::array_t<long long> counts = py::array_t<long long>{0};

  std::unordered_set<long long> set;
  for (int i = 0; i < numel; i++) {
    set.insert(*(input.data() + i));
  }

  py::array_t<long long> output =
      py::array_t<long long>((long long)(int)set.size());

  long long *output_data = (long long *)output.data();

  if (sorted) {
    std::vector<long long> vec(set.begin(), set.end());
    std::sort(vec.begin(), vec.end());
    std::copy(vec.begin(), vec.end(), output_data);
  } else {
    std::copy(set.begin(), set.end(), output_data);
  }

  int output_numel = set.size();
  if (return_inverse || return_counts) {
    inverse_indices = py::array_t<long long>(input.size());
    auto *inverse_indices_data = (long long *)inverse_indices.data();
    std::unordered_map<long long, long long> inverse_map;
    inverse_map.reserve(output_numel);
    for (int64_t i = 0; i < output_numel; ++i) {
      inverse_map[output_data[i]] = i;
    }
    for (int64_t i = 0; i < numel; ++i) {
      inverse_indices_data[i] = inverse_map[input_data[i]];
    }
    if (return_counts) {
      std::unordered_map<long long, long long> counts_map;
      counts_map.reserve(output_numel);
      for (long long i = 0; i < output_numel; ++i) {
        counts_map[output_data[i]] = 0;
      }
      for (long long i = 0; i < numel; i++) {
        counts_map[input_data[i]] += 1;
      }
      counts = py::array_t<long long>(output_numel);
      long long *counts_data = (long long *)counts.data();
      for (long long i = 0; i < output_numel; i++) {
        counts_data[i] = counts_map[output_data[i]];
      }
    }
  }

  // return py::make_tuple(output, inverse_indices, counts);

  py::list res;
  res.append(output);
  res.append(inverse_indices);
  res.append(counts);

  return res;
}

PYBIND11_MODULE(_unique, m) {
  m.doc() = "gammagl tensor ops unique";
  m.def(
      "c_unique", &unique_impl, "input"_a, "sorted"_a = true,
      "return_inverse"_a = false, "return_counts"_a = false);
}
