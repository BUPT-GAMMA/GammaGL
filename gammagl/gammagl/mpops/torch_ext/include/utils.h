#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <vector>

std::vector<int64_t> list2vec(const c10::List<int64_t> list);
