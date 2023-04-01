#include "convert.h"
#include "neighbor_sample.h"


int calc_threads_num(int diff) {
    int total_num = std::thread::hardware_concurrency();
    int norm = total_num / 4;
    int will_num = norm >= 4 ? norm : total_num;
//    return will_num >= diff ? diff > 8 ? 8 : diff : will_num > 8 ? will_num : 8;
    return will_num >= diff ? diff : will_num;
}


void parallel_for(int begin, int end, std::function<void(int, int)> f, int num_worker = 0) {

    // over 8 threads won't accelerate the runtime speed.
    int valid_threads = std::thread::hardware_concurrency();
    int threads_num =
            num_worker == 0 ? calc_threads_num(end - begin) : num_worker > valid_threads ? valid_threads : num_worker;

    std::cout << "fn[parallel_for] uses " << threads_num << " threads to calculating..." << std::endl;

    int rem = 0;

    if (end - begin >= threads_num) {
        rem = (end - begin) % threads_num;
    }
    int span = (int) ((end - begin) / threads_num);
    span = span > 1 ? span : 1;

    std::vector<std::thread> ts;

    for (int i = 0; i < threads_num; ++i, --rem) {
        int c_end = begin + span + (rem > 0 ? 1 : 0);

        std::thread t(f, begin, c_end);

        ts.push_back(std::move(t));

        begin = c_end;
    }
    for (auto &t : ts) {
        t.join();
    }

    std::cout << "parallel finished" << std::endl;
}


py::array_t<long long> ind2ptr(py::array_t<long long> ind, long long M, int num_worker = 0) {

    auto ind_data = ind.mutable_unchecked<1>();
    long long numel = (long long) ind_data.size();

    py::array_t<long long> out(M + 1);
    auto out_data = out.mutable_unchecked<1>();

    if (numel == 0) {
        for (ssize_t i = 0; i < out_data.size(); i++)out_data(i) = 0;
        return out;
    }

    for (int i = 0; i <= ind_data(0); i++) {
        out_data(i) = 0;
    }
    parallel_for(0, numel, [&](int begin, int end) {
        if (end - begin < 1) {
            return;
        }
        long long idx = ind_data(begin);
        long long next_idx;
        for (int i = begin; i < std::min(end, (int) numel - 1); i++) {
            next_idx = ind_data(i + 1);
            for (; idx < next_idx; idx++) {
                out_data(idx + 1) = i + 1;
            }
        }

//        for (ssize_t i = begin; i < std::min((long long) end, numel - 1); i++) {
//            next_idx = ind_data(i + 1);
//            for (; idx < next_idx; idx++) {
//                out_data(idx + 1) = i + 1;
//            }
//        }

        for (long long i = ind_data(numel - 1) + 1; i < M + 1; i++) {
            out_data(i) = numel;
        }

    }, num_worker);

    return out;
}

void set_to_one(py::array_t<long long> arr, int num_worker) {

    auto arr_data = arr.mutable_unchecked();

    parallel_for(0, arr_data.size(), [&](int begin, int end) {
        for (int i = begin; i < end; i++) {
            arr_data(i) = 1;
        }
    }, num_worker);

}


PYBIND11_MODULE(_convert, m) {
    m.doc() = "gammagl sparse convert";
    m.def("c_ind2ptr", &ind2ptr);
    m.def("c_set_to_one", &set_to_one);
    m.def("c_neighbor_sample", &neighbor_sample);
    m.def("c_hetero_neighbor_sample", &hetero_neighbor_sample);
}

