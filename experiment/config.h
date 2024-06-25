#pragma once

#include <bits/stdc++.h>
#include "timer.h"
#include "pq_dist.h"
using namespace std;

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

    int max_level = 0;

    bool use_degree_adjust = 0;
    
    bool use_reverse_edges = 0;

    // 测试最底层接入点距离
    bool statis_ep_dis;
    std::vector<float> ep_dist;

    bool use_multiple_ep;
    int ep_cnt = 10;
    std::priority_queue<std::pair<float, int> > eps;

    bool test_ep_with_calc = 0;
    float ep_dist_limit;
    int ep_in_limit_cnt;
    int ep_tot_dis_calc;
    bool ep_is_in_limit;
    void clear_test_ep() {
        ep_in_limit_cnt = 0;
        ep_tot_dis_calc = 0;
    }
    
    bool statis_ep_nn_pair = 0;
    vector<pair<float, float> > ep_nn_pair;

    bool test_bruteforce_ep = 0;

    int high_level_dist_calc = 0;

    bool statis_recursive_len = 0;
    int recursive_len = 0;

    float nn_dist;
    bool test_nn_path_len = 0;
    bool find_nn = 0;
    int nn_path_len = 0;
    int dist_calc_when_nn = 0;

    bool use_extent_neighbor = 0;

    bool use_PQ = 0;
    // PQDist tem;
};

