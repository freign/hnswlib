#pragma once

#include <bits/stdc++.h>
#include "timer.h"

struct Config {

    bool test_dist_func_time = 0;
    bool statis_wasted_cand = 0;

    uint64_t tot_cand_nodes;
    uint64_t wasted_cand_nodes;

    void clear_cand() {
        tot_cand_nodes = 0;
        wasted_cand_nodes = 0;
    }

    Config() = default;

};

