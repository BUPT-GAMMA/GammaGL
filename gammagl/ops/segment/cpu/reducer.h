#pragma once

#ifndef REMOTE_REDUCER_H
#define REMOTE_REDUCER_H


#include <map>
#include "../../extensions.h"

enum ReductionType {
    SUM, MEAN, MUL, DIV, MIN, MAX
};

const std::map<std::string, ReductionType> reduce2REDUCE = {
        {"sum",  SUM},
        {"mean", MEAN},
        {"mul",  MUL},
        {"div",  DIV},
        {"min",  MIN},
        {"max",  MAX},
};

#define DISPATCH_REDUCTION_TYPES(reduce, ...)                                  \
  [&] {                                                                        \
    switch (reduce2REDUCE.at(reduce)) {                                        \
    case SUM: {                                                                \
      static constexpr ReductionType REDUCE = SUM;                             \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MEAN: {                                                               \
      static constexpr ReductionType REDUCE = MEAN;                            \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MUL: {                                                                \
      static constexpr ReductionType REDUCE = MUL;                             \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case DIV: {                                                                \
      static constexpr ReductionType REDUCE = DIV;                             \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MIN: {                                                                \
      static constexpr ReductionType REDUCE = MIN;                             \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MAX: {                                                                \
      static constexpr ReductionType REDUCE = MAX;                             \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    }                                                                          \
  }()

template<typename Tp>
struct Reducer {

    ReductionType REDUCE;

    Reducer(ReductionType REDUCE) {
        this->REDUCE = REDUCE;
    }

    Tp init() {
        if (REDUCE == MUL || REDUCE == DIV)
            return (Tp) 1;
        else if (REDUCE == MIN)
            return std::numeric_limits<Tp>::max();
        else if (REDUCE == MAX)
            return std::numeric_limits<Tp>::lowest();
        else
            return (Tp) 0;
    }

    void update(Tp *val, Tp new_val, Tp *arg,
                Tp new_arg) {
        if (REDUCE == SUM || REDUCE == MEAN)
            *val = *val + new_val;
        else if (REDUCE == MUL)
            *val = *val * new_val;
        else if (REDUCE == DIV)
            *val = *val / new_val;
        else if ((REDUCE == MIN && new_val < *val) ||
                 (REDUCE == MAX && new_val > *val)) {
            *val = new_val;
            *arg = new_arg;
        }
    }

    void write(Tp *address, Tp val,
               Tp *arg_address, Tp arg, int count) {
        if (REDUCE == SUM || REDUCE == MUL || REDUCE == DIV)
            *address = val;
        else if (REDUCE == MEAN)
            *address = val / (Tp) (count > 0 ? count : 1);
        else if (REDUCE == MIN || REDUCE == MAX) {
            if (count > 0) {
                *address = val;
                *arg_address = arg;
            } else
                *address = (Tp) 0;
        }
    }


};


#endif //REMOTE_REDUCER_H
