/*
 * @Description: TODO
 * @Author: WuJing
 * @created: 2023-04-11
 */
#include "parallel_hashmap/phmap.h"
#include <unordered_set>


#include "../../extensions.h"
#include "sparse_utils.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

typedef string node_t;
typedef string rel_t; // "paper__to__paper"
typedef vector<string> edge_t;
typedef py::array_t<int64_t> tensor;

template <class T>
py::dict from_dict(phmap::flat_hash_map<string, vector<T>> &map);

py::list neighbor_sample(Tensor colptr, Tensor row, Tensor input_node,
                         const vector<int64_t> &num_neighbors, bool replace,
                         bool directed);

py::list
hetero_neighbor_sample(const vector<node_t> &node_types,
                       const vector<edge_t> &edge_types,
                       const unordered_map<rel_t, tensor> &colptr_dict,
                       const unordered_map<rel_t, tensor> &row_dict,
                       const unordered_map<node_t, tensor> &input_node_dict,
                       const unordered_map<rel_t, tensor> &num_neighbors_dict,
                       int64_t num_hops, bool replace, bool directed);