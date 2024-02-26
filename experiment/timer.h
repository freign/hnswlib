#pragma once
#include <bits/stdc++.h>

class StopW {
    std::chrono::high_resolution_clock::time_point time_begin;
 public:
    StopW() {
        time_begin = std::chrono::high_resolution_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::high_resolution_clock::time_point time_end = std::chrono::high_resolution_clock::now();
        return (std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::high_resolution_clock::now();
    }
};