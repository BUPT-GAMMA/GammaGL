#pragma once

#ifndef REMOTE_UTILS_H
#define REMOTE_UTILS_H

#include "../../extensions.h"

void fill(Tensor, int64_t);

template<class T, class ...Args>
py::array_t<T> python_for_c(const char *module_name, const char *func_name, Args &&...args) {
    py::call_guard<py::gil_scoped_acquire>{};
    py::module_ np = py::module_::import(module_name);
    py::object func = np.attr(func_name);
    py::object res_obj = func(std::forward<Args>(args)...);
    return py::cast<py::array_t<T>>(res_obj);
}

template<class T, class ...Args>
py::array_t<T> np_func(const char *func_name, Args &&...args) {
    return python_for_c<T>("numpy", func_name, std::forward<Args>(args)...);
}

template<class T, class ...Args>
py::array_t<T> py_helper(const char *func_name, Args &&...args) {
    return python_for_c<T>("gammagl.ops.segment.py_helper", func_name, std::forward<Args>(args)...);
}


#endif //REMOTE_UTILS_H
