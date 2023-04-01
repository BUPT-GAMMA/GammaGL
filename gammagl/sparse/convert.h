#pragma once

#include <thread>
#include <iostream>
#include <vector>
#include <future>
#include <condition_variable>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <pybind11/stl.h>
#include "parallel_hashmap/phmap.h"
#include "utils.h"

#ifndef REMOTE_CONVERT_H
#define REMOTE_CONVERT_H


namespace py = pybind11;
using namespace std;
using namespace pybind11::literals;

typedef string node_t;
typedef string rel_t; // "paper__to__paper"类型
typedef vector<string> edge_t;
typedef py::array_t<long long> tensor;

void parallel_for(int begin, int end, std::function<void(int, int)> f, int num_worker);

py::array_t<long long> ind2ptr(py::array_t<long long> ind, long long M, int num_worker);


#endif //REMOTE_CONVERT_H
