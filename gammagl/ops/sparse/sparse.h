/*
 * @Description: TODO 
 * @Author: WuJing
 * @created: 2023-04-11
 */
#pragma once

#ifndef REMOTE_SPARSE_H
#define REMOTE_SPARSE_H

#include <cstdlib>
#include "../extensions.h"
#include <ctime>

Tensor random_walk(Tensor rowptr, Tensor col,
                   Tensor start, int64_t walk_length);


#endif //REMOTE_SPARSE_H
