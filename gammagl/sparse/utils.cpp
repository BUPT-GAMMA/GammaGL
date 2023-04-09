#include "utils.h"



int64_t uniform_randint(int64_t low, int64_t high) {
    srand(time(0));
    return rand() % (high - low) + low;
}

int64_t uniform_randint(int64_t high) {
    srand(time(0));
    return rand() % high;
}
