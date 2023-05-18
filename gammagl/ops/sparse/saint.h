/*
 * @Description: TODO 
 * @Author: WuJing
 * @created: 2023-04-11
 */
#pragma once

#ifndef REMOTE_SAINT_H
#define REMOTE_SAINT_H

#include <iostream>
#include <vector>
#include "../extensions.h"


py::list subgraph(Tensor idx, Tensor rowptr, Tensor row,
                   Tensor col);

#endif //REMOTE_SAINT_H
