#pragma once

#include <condition_variable>
#include <future>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>

#include "sparse_utils.h"

#ifndef REMOTE_CONVERT_H
#define REMOTE_CONVERT_H

void parallel_for(
    int64_t begin, int64_t end, std::function<void(int64_t, int64_t)> f,
    int num_worker);

int64_t long_min(int64_t a, int64_t b);

py::array_t<int64_t> ind2ptr(
    py::array_t<int64_t> ind, int64_t M, int num_worker);
py::array_t<int64_t> ptr2ind(
    py::array_t<int64_t> ind, int64_t M, int num_worker);

void set_to_one(Tensor arr, int num_worker = 0);

#endif  // REMOTE_CONVERT_H
