/**
 * @Description TODO
 * @Author WuJing
 * @Created 2023/4/11
 */

#include "segment_csr.h"

#include "optional"

int64_t index2offset(int idx, vector<ssize_t> sizes) {
  int offset = 0;
  for (int i = sizes.size() - 1; i >= 0; --i) {
    offset += idx % sizes[i];
    idx /= sizes[i];
  }
  return offset;
}

int64_t indptr2offset(int idx, vector<ssize_t> sizes) {
  int offset = idx % (sizes[sizes.size() - 1] - 1);
  idx /= sizes[sizes.size() - 1] - 1;
  for (int i = sizes.size() - 2; i >= 0; --i) {
    offset += idx % sizes[i];
    idx /= sizes[i];
  }
  return offset;
}

py::list segment_csr_cpu(
    Tensor src, Tensor indptr, std::optional<Tensor> optional_out_obj,
    std::string &reduce) {
  assert(src.ndim() >= indptr.ndim());

  // broadcast
  vector<ssize_t> sizes(indptr.ndim());
  sizes[sizes.size() - 1] = indptr.shape(indptr.ndim() - 1);

  for (auto i = 0; i < indptr.ndim() - 1; i++) sizes[i] = src.shape(i);
  // TODO

  //    indptr = indptr.expand(sizes);

  auto dim = indptr.ndim() - 1;

  //    src = src.contiguous();

  Tensor out;
  if (!optional_out_obj.has_value()) {
    auto out = py::cast<Tensor>(optional_out_obj.value());

    for (auto i = 0; i < out.ndim(); i++)
      if (i != dim) assert(src.shape(i) == out.shape(i));
    assert(src.size() == 0 || out.shape(dim) == indptr.shape(dim) - 1);
  } else {
    //        sizes = src.sizes().vec();
    vector<ssize_t> sizes(indptr.ndim());
    for (int i = 0; i < indptr.ndim(); i++) sizes[i] = indptr.shape(i);

    sizes[dim] = std::max<int64_t>(indptr.shape(dim) - 1, 0);
    //        out = torch::empty(sizes, src.options());
    out = Tensor(sizes);
  }

  //    torch::optional <torch::Tensor> arg_out = torch::nullopt;

  std::optional<Tensor> arg_out = std::nullopt;

  vector<ssize_t> indptr_sizes(indptr.ndim());
  for (int i = 0; i < indptr.ndim(); i++) indptr_sizes[i] = indptr.shape(i);

  //    auto arg_out_data = arg_out.value().mutable_unchecked();
  //    int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    //        arg_out = torch::full(out.sizes(), src.size(dim),
    //        indptr.options());

    arg_out = Tensor(indptr_sizes);
    // TODO
    //        fill(arg_out, src.shape(dim));
    py_helper<int64_t>("fill", arg_out, src.shape(dim));

    //        arg_out_data = arg_out.value().mutable_data();
  } else {
    arg_out = Tensor(0);
  }
  //    auto arg_out_data = arg_out.value_or(Tensor(0)).mutable_unchecked();

  //    auto arg_out_data = arg_out.value().mutable_unchecked();
  auto arg_out_data = arg_out.value().mutable_unchecked();

  if (src.size() == 0) {
    if (optional_out_obj.has_value()) {
      //            out.fill_(0);
      // TODO
      //            fill(out, 0);
      py_helper<int64_t>("fill", out, src.shape(dim));
    }
    return py::make_tuple(out, arg_out);
  }

  auto N = out.shape(dim) * (indptr.size() / indptr.shape(indptr.ndim() - 1));
  auto K = out.size() / N;
  auto E = src.shape(dim);

  //    auto indptr_info = getTensorInfo<int64_t>(indptr);
  //    auto stride = indptr_info.strides[indptr_info.dims - 1];
  auto stride = 1;
  std::vector<int64_t> args(K);

  auto src_data = src.unchecked();
  auto out_data = out.mutable_unchecked();

  std::vector<int64_t> vals(K);
  int64_t row_start, row_end;

  auto indptr_data = indptr.unchecked();

  //    DISPATCH_REDUCTION_TYPES(reduce,[&]{
  //
  //    })

  Reducer<int64_t> reducer{reduce2REDUCE.at(reduce)};

  for (auto n = 0; n < N; n++) {
    auto offset = indptr2offset(n, indptr_sizes);
    row_start = indptr_data(offset);
    row_end = indptr_data(offset + 1);

    offset = (n / indptr_sizes[indptr_sizes.size() - 1] - 1) * E * K;
    for (auto k = 0; k < K; k++) {
      vals[k] = reducer.init();
    }

    for (auto e = row_start; e < row_end; e++) {
      for (auto k = 0; k < K; k++) {
        reducer.update(&vals[k], src_data(offset + e * K + k), &args[k], e);
      }
    }
    for (auto k = 0; k < K; k++)
      reducer.write(
          &out_data(n * K + k), vals[k], &arg_out_data(n * K + k), args[k],
          row_end - row_start);
  }

  // return py::make_tuple(out, arg_out);
  py::list res;
  res.append(out);
  res.append(arg_out);

  return res;
}