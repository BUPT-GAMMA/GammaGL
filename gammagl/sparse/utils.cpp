#include <random>
#include <ctime>

long long uniform_randint(long long low, long long high) {
    srand(time(0));
    return rand() % (high - low) + low;
}

long long uniform_randint(long long high) {
    srand(time(0));
    return rand() % high;
}