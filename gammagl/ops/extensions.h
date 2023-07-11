/*
 * @Description: TODO 
 * @Author: WuJing
 * @created: 2023-04-11
 */
#pragma once

#ifndef GAMMALGL_OPS_EXTENSIONS_H
#define GAMMALGL_OPS_EXTENSIONS_H

#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;
using namespace pybind11::literals;

//using Tensor = py::array_t<int64_t>;
typedef py::array_t<int64_t> Tensor;

#endif //GAMMALGL_OPS_EXTENSIONS_H


