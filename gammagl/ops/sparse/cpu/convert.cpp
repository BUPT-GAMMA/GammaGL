/**
 * @Description TODO
 * @Author WuJing
 * @Created 2023/4/4
 */

#include "convert.h"

int calc_threads_num(int diff) {
  int total_num = std::thread::hardware_concurrency();
  int norm = total_num / 4;
  int will_num = norm >= 4 ? norm : total_num;
  //    return will_num >= diff ? diff > 8 ? 8 : diff : will_num > 8 ? will_num
  //    : 8;
  return will_num >= diff ? diff : will_num;
}

void parallel_for(
    int64_t begin, int64_t end, std::function<void(int64_t, int64_t)> f,
    int num_worker = 0) {
  // over 8 threads won't accelerate the runtime speed.
  int valid_threads = std::thread::hardware_concurrency();
  int threads_num = num_worker == 0              ? calc_threads_num(end - begin)
                    : num_worker > valid_threads ? valid_threads
                                                 : num_worker;

  // std::cout << "fn[parallel_for] uses " << threads_num << " threads to
  // calculating..." << std::endl;

  int rem = 0;

  if (end - begin >= threads_num) {
    rem = (end - begin) % threads_num;
  }
  auto span = (int64_t)((end - begin) / threads_num);
  span = span > 1 ? span : 1;

  std::vector<std::thread> ts;

  for (int i = 0; i < threads_num; ++i, --rem) {
    int64_t c_end = begin + span + (rem > 0 ? 1 : 0);

    std::thread t(f, begin, c_end);

    ts.push_back(std::move(t));

    begin = c_end;
  }
  for (auto &t : ts) {
    t.join();
  }

  // std::cout << "parallel finished" << std::endl;
}

int64_t long_min(int64_t a, int64_t b) { return a <= b ? a : b; }

Tensor ind2ptr(Tensor ind, int64_t M, int num_worker = 0) {
  auto ind_data = ind.unchecked<1>();
  int64_t numel = (int64_t)ind_data.size();

  Tensor out(M + 1);
  auto out_data = out.mutable_unchecked<1>();

  if (numel == 0) {
    for (ssize_t i = 0; i < out_data.size(); i++) out_data(i) = 0;
    return out;
  }

  for (int i = 0; i <= ind_data(0); i++) {
    out_data(i) = 0;
  }
  parallel_for(
      0, numel,
      [&](int64_t begin, int64_t end) {
        if (end - begin < 1) {
          return;
        }
        int64_t idx = ind_data(begin);
        int64_t next_idx;
        for (int64_t i = begin; i < long_min(end, numel - 1); i++) {
          next_idx = ind_data(i + 1);
          for (; idx < next_idx; idx++) {
            out_data(idx + 1) = i + 1;
          }
        }

        //        for (ssize_t i = begin; i < std::min((int64_t end, numel - 1);
        //        i++) {
        //            next_idx = ind_data(i + 1);
        //            for (; idx < next_idx; idx++) {
        //                out_data(idx + 1) = i + 1;
        //            }
        //        }

        for (int64_t i = ind_data(numel - 1) + 1; i < M + 1; i++) {
          out_data(i) = numel;
        }
      },
      num_worker);

  return out;
}

Tensor ptr2ind(Tensor ptr, int64_t E, int num_worker = 0) {
  //    auto out = torch::empty(E);
  Tensor out(E);

  auto ptr_data = ptr.unchecked();
  auto out_data = out.mutable_unchecked();

  int64_t numel = ptr.size();

  //    int64_t grain_size = at::internal::GRAIN_SIZE;
  parallel_for(
      0, numel - 1,
      [&](int64_t begin, int64_t end) {
        int64_t idx = ptr_data(begin), next_idx;
        for (int64_t i = begin; i < end; i++) {
          next_idx = ptr_data(i + 1);
          for (int64_t e = idx; e < next_idx; e++) out_data(e) = i;
          idx = next_idx;
        }
      },
      num_worker);

  return out;
}

void set_to_one(Tensor arr, int num_worker) {
  auto arr_data = arr.mutable_unchecked();
  parallel_for(
      0, arr_data.size(),
      [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          arr_data(i) = 1;
        }
      },
      num_worker);
}

PYBIND11_MODULE(_convert, m) {
  m.doc() = "gammagl sparse convert";
  m.def("c_ind2ptr", &ind2ptr);
  m.def("c_ptr2ind", &ptr2ind);
  m.def("c_set_to_one", &set_to_one);
}
