#pragma once

#include <bits/stdc++.h>
#include "timer.h"

struct Config {

    int search_knn_times = 0;

    bool statis_wasted_cand = 0;
    uint64_t tot_cand_nodes;
    uint64_t wasted_cand_nodes;
    uint64_t tot_calculated_nodes;
    void clear_cand() {
        tot_cand_nodes = 0; // 加入candidates集合的点数
        wasted_cand_nodes = 0; // 最后留在candidates集合中的点数，
                               // 这些点加入了candidates集合但是没有任何作用
        tot_calculated_nodes = 0; // 计算了距离的点数
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

    
    bool test_dir_vector = 0;
    bool use_dir_vector = 0;
    Config() = default;

    bool test_enter_point_dis = 0;
    float ep_dis_tot = 0;
    
    uint64_t tot_dist_calc = 0;
    uint64_t disc_calc_avoided = 0;
};

