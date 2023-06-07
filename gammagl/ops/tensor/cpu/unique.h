/*
 * @Description: TODO 
 * @Author: WuJing
 * @created: 2023-05-04
 */
#pragma once

#include <algorithm>
#include "../../extensions.h"


py::list unique_impl(
        const py::array_t<long long> &input,
        const bool sorted,
        const bool return_inverse,
        const bool return_counts
);