#pragma once

#include <bits/stdc++.h>
#include "timer.h"

struct Config {

    bool test_dist_func_time = 0;

    std::mutex add_time_lock;
    float dist_func_time = 0;
    float search_time = 0;
    std::string log_file_path;
    int tick;

    Config() = default;
    
    void clear_dist_func_time() {
        dist_func_time = 0;
    }
    void clear_search_time() {
        search_time = 0;
    }
    void reset_timer(StopW &timer) {
        std::lock_guard<std::mutex> lock(add_time_lock);
        timer.reset();
    }
    void add_time(StopW &timer, float &sum) {
        std::lock_guard<std::mutex> lock(add_time_lock);
        sum += timer.getElapsedTimeMicro();
    }
};

