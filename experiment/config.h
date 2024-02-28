#pragma once

#include <bits/stdc++.h>
#include "timer.h"

struct Config {

    bool statis_wasted_cand = 0;
    uint64_t tot_cand_nodes;
    uint64_t wasted_cand_nodes;
    uint64_t tot_calculated_nodes;
    void clear_cand() {
        tot_cand_nodes = 0;
        wasted_cand_nodes = 0;
        tot_calculated_nodes = 0;
    }


    bool statis_used_neighbor_dist = 0;
    std::unordered_set<uint64_t> used_points_id;
    std::vector<uint64_t> used_points;
    std::unordered_set<uint64_t> all_points;
    void clear_used_neighbors() {
        used_points_id.clear();
        used_points.clear();
        all_points.clear();
    }

    Config() = default;

};

