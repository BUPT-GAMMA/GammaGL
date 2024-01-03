/*
 * @Description: TODO
 * @Author: WuJing
 * @created: 2023-04-11
 */
#include "sparse_utils.h"

// template<class T, class ...Args>
// py::array_t<T> np_func(const char *func_name, Args ...args) {
//     static py::module_ np = py::module_::import("numpy");
//     py::object func = np.attr(func_name);
//     py::object res_obj = func(args...);
//     py::array_t<T> result_arr = py::cast<py::array_t<T>>(res_obj);
//     return result_arr;
// }

// template<class T, class ...Args>
// py::array_t<T> python_for_c(const char *func_name, Args &&...args) {
//     py::gil_scoped_acquire acquire{};
////    py::call_guard<py::gil_scoped_release>{};
//    py::module_ np = py::module_::import("python_for_c");
//    py::object func = np.attr(func_name);
//    py::object res_obj = func(std::forward<Args>(args)...);
//    py::array_t<T> result_arr = py::cast<py::array_t<T>>(res_obj);
//
//    return result_arr;
//}

int64_t uniform_randint(int64_t low, int64_t high) {
  srand(time(0));
  return rand() % (high - low) + low;
}

int64_t uniform_randint(int64_t high) {
  srand(time(0));
  return rand() % high;
}

// template<class T>
// py::array_t<T> rand_tensor(ssize_t size) {
//     py::array_t<T> arr{size};
//     auto arr_data = arr.mutable_unchecked();
//     for (int i = 0; i < arr.shape(0); i++) {
//         srand(time(0));
//         arr_data(i) = static_cast<T> (rand()) / static_cast<T> (RAND_MAX);
//     }
//     return arr;
// }

// template<class T>
// py::array_t<T> rand_tensor(vector<int64_t> &list) {
//
//     if (list.size() > 3) {
//         throw std::length_error("Do not support array shape over 3!");
//     } else if (list.size() <= 0) {
//         return py::array_t<T>{0};
//     }
//
//     if (list.size() == 1) {
//         return rand_tensor<T>(*list.begin());
//     } else if (list.size() == 2) {
//         py::array_t<T> arr{list[0], list[1]};
//         auto arr_data = arr.mutable_unchecked();
//         for (int i = 0; i < arr.shape(0); i++) {
//             for (int j = 0; j < arr.shape(1); j++) {
//                 srand(time(0));
//                 arr_data(i, j) = static_cast<T> (rand()) / static_cast<T>
//                 (RAND_MAX);
//             }
//         }
//         return arr;
//     } else if (list.size() == 3) {
//         py::array_t<T> arr{list[0], list[1], list[2]};
//         auto arr_data = arr.mutable_unchecked();
//         for (int i = 0; i < arr.shape(0); i++) {
//             for (int j = 0; j < arr.shape(1); j++) {
//                 for (int k = 0; k < arr.shape(2); k++) {
//                     srand(time(0));
//                     arr_data(i, j, k) = static_cast<T> (rand()) /
//                     static_cast<T> (RAND_MAX);
//                 }
//             }
//         }
//         return arr;
//     }
//
// }