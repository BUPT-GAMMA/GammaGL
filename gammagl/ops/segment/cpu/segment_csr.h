/*
 * @Description: TODO 
 * @Author: WuJing
 * @created: 2023-04-11
 */
#pragma once

#ifndef REMOTE_SEGMENT_CSR_H
#define REMOTE_SEGMENT_CSR_H

#include <pybind11/functional.h>
#include <optional>

#include "reducer.h"
#include "../../extensions.h"
#include "segment_utils.h"

py::list
segment_csr_cpu(Tensor src, Tensor indptr,
                std::optional<Tensor> optional_out_obj,
                std::string &reduce);


#endif //REMOTE_SEGMENT_CSR_H
