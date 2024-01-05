/*
 * @Description: TODO
 * @Author: WuJing
 * @created: 2023-04-11
 * @LastEditTime: 2023-05-04
 */
#pragma once

#include <ctime>
#include <random>

#include "../../extensions.h"

int64_t uniform_randint(int64_t high);

int64_t uniform_randint(int64_t low, int64_t high);

// template<class T, class ...Args>
// py::array_t<T> np_func(const char *func_name, Args ...args);

#ifndef PYTHON_FOR_C
#define PYTHON_FOR_C

template <class T, class... Args>
py::array_t<T> python_for_c(
    const char *module_name, const char *func_name, Args &&...args) {
  py::call_guard<py::gil_scoped_acquire>{};
  py::module_ np = py::module_::import(module_name);
  py::object func = np.attr(func_name);
  py::object res_obj = func(std::forward<Args>(args)...);
  return py::cast<py::array_t<T>>(res_obj);
}

template <class T, class... Args>
py::array_t<T> np_func(const char *func_name, Args &&...args) {
  return python_for_c<T>("numpy", func_name, std::forward<Args>(args)...);
}

template <class T, class... Args>
py::array_t<T> py_helper(const char *func_name, Args &&...args) {
  return python_for_c<T>(
      "gammagl.ops.sparse.py_helper", func_name, std::forward<Args>(args)...);
}

#endif  // PYTHON_FOR_C

// template<class T>
// py::array_t<T> rand_tensor(ssize_t size);

// template<class T>
// py::array_t<T> rand_tensor(vector<int64_t> &list);
