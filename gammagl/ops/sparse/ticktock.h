#pragma once

#include <iostream>
#include <chrono>

#ifndef __ycm__
#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x) printf("%s: %lfs\n", #x, std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench_##x).count());

#define TICK_TIMES(x,times) auto bench_##x = std::chrono::steady_clock::now();\
for(int _i = 0;_i < times; _i++){\

#define TOCK_TIMES(x)}\
printf("%s: %lfs\n", #x, std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench_##x).count());

#else
#define TICK(x)
#define TOCK(x)
#define TICK(x,times)
#define TOCK(x)
#endif
