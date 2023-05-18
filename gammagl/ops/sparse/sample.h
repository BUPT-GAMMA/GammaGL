#pragma once

#ifndef REMOTE_SAMPLE_H
#define REMOTE_SAMPLE_H

#include <vector>
#include <unordered_map>
#include "../extensions.h"
#include "sparse_utils.h"

py::list
sample_adj(Tensor rowptr, Tensor col, Tensor idx,
           int64_t num_neighbors, bool replace);




#endif //REMOTE_SAMPLE_H
