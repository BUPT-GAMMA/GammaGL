#include <unordered_set>
#include "parallel_hashmap/phmap.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "utils.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;


template<class T>
py::dict from_dict(phmap::flat_hash_map<string, vector<T>> &map);

py::tuple neighbor_sample(py::array_t<long long> colptr, py::array_t<long long> row,
                          py::array_t<long long> input_node, const vector<long long> num_neighbors,
                          bool replace,
                          bool directed);

py::tuple hetero_neighbor_sample(
        const vector<node_t> &node_types,
        const vector<edge_t> &edge_types,
        const unordered_map<rel_t, tensor> &colptr_dict,
        const unordered_map<rel_t, tensor> &row_dict,
        const unordered_map<node_t, tensor> &input_node_dict,
        const unordered_map<rel_t, tensor> &num_neighbors_dict,
        const long long num_hops,
        bool replace,
        bool directed
);