#pragma once

#ifndef REMOTE_SAINT_H
#define REMOTE_SAINT_H

#include <iostream>
#include <vector>
#include "extensions.h"


py::tuple subgraph(Tensor idx, Tensor rowptr, Tensor row,
                   Tensor col);

#endif //REMOTE_SAINT_H
