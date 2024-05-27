#pragma once

#include "visited_list_pool.h"
#include "../experiment/config.h"
#include "../experiment/dir_vector.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include "hnswalg.h"

using namespace std;

template<typename dist_t>
void hnswlib::HierarchicalNSW<dist_t>::reconstructGraphWithConstraint(int eo, int ei) {
    // 建立反向图
    vector<vector<pair<dist_t, int> > > reverse_graph(max_elements_, vector<pair<dist_t, int> >(0));
    for (int cur = 0; cur < max_elements_; cur++) {
        int *data = (int *) get_linklist0(cur);
        for (int i = 1; i <= *data; i++) {
            int n = *(data + i);
            auto d = this->fstdistfunc_(getDataByInternalId(cur), getDataByInternalId(n), this->dist_func_param_);
            reverse_graph[n].push_back(make_pair(d, cur));
        }
    }

    vector<pair<int ,int> > reverseEdgeSize(max_elements_);
    for (int cur = 0; cur < max_elements_; cur++) {
        reverseEdgeSize[cur] = make_pair(reverse_graph[cur].size(), cur);
    }
    sort(reverseEdgeSize.begin(), reverseEdgeSize.end());

    vector<int> indegree(max_elements_, 0);
    vector<vector<int> > outedge(max_elements_, vector<int>(0));

    for (auto pr: reverseEdgeSize) {
        int cur = pr.second;
        sort(reverse_graph[cur].begin(), reverse_graph[cur].end());
        for (auto graph_node: reverse_graph[cur]) {
            int nn = graph_node.second;
            if (indegree[nn] >= ei) continue;
            if (indegree[nn] > 0 && outedge[cur].size() >= eo) continue;
            outedge[cur].push_back(nn);
            indegree[nn] ++ ;
        }
    }

    for (int cur = 0; cur < max_elements_; cur++) {
        // 先翻转,再push_back,如果超出maxM0,则删掉最远的边
        reverse(outedge[cur].begin(), outedge[cur].end());

        // test
        // eo = 100;
        // int limit = 7;
        // if (outedge[cur].size() > limit)
        //     outedge[cur].resize(limit);


        unordered_set<int> exist;
        vector<pair<dist_t, int> > neighbors;
        for (auto o: outedge[cur]) exist.insert(o);
        int *data = (int *) get_linklist0(cur);
        for (int i = 1; i <= *data; i++) {
            int n = *(data + i);
            auto d = this->fstdistfunc_(getDataByInternalId(cur), getDataByInternalId(n), this->dist_func_param_);
            neighbors.push_back(make_pair(d, n));
        }
        sort(neighbors.begin(), neighbors.end());
        for (int i = 0; i < min(eo, (int)neighbors.size()); i++) {
            int n = neighbors[i].second;
            if (exist.find(n) != exist.end()) continue;
            outedge[cur].push_back(n);
        }
        if (outedge[cur].size() > maxM0_) {
            reverse(outedge[cur].begin(), outedge[cur].end());
            outedge[cur].resize(maxM0_);
        }
        if (outedge[cur].size() > maxM0_) {
            cout << outedge[cur].size() << ' ' << maxM0_ << "\n";
            perror("too many out edges\n");
            exit(1);
        }
        *data = outedge[cur].size();
        for (int i = 1; i <= *data; i++)
            *(data + i) = outedge[cur][i-1];
    }
}

template<typename dist_t>
void hnswlib::HierarchicalNSW<dist_t>::reconstructGraph(int eo, int ei) {
    vector<vector<pair<dist_t, int> > > reverse_graph(max_elements_, vector<pair<dist_t, int> >(0));
    reverse_edges.resize(max_elements_, vector<tableint>(0));
    this->neighbors_adjust.resize(max_elements_, vector<tableint>(0));
    for (int cur = 0; cur < max_elements_; cur++) {
        int *data = (int *) get_linklist0(cur);
        for (int i = 1; i <= *data; i++) {
            int n = *(data + i);
            auto d = this->fstdistfunc_(getDataByInternalId(cur), getDataByInternalId(n), this->dist_func_param_);
            reverse_graph[n].push_back(make_pair(d, cur));
            this->reverse_edges[n].push_back(cur);
            neighbors_adjust[cur].push_back(n);
        }
    }
    return ;
    for (int cur = 0; cur < max_elements_; cur++) {
        if (reverse_edges[cur].size() < 10) {
            int *data = (int *) get_linklist0(cur);
            for (int i = 1; i <= *data; i++) {
                int n = *(data + i);
                bool find_cur = 0;
                int *data_n = (int *) get_linklist0(n);

                if (neighbors_adjust[n].size() > 20) continue;

                for (int j = 1; j <= *data_n; j++) {
                    if (data_n[j] == cur) find_cur = 1;
                }
                if (!find_cur) {
                    cout << "add " << n << " " << cur << "\n";
                    neighbors_adjust[n].push_back(cur);
                    reverse_edges[cur].push_back(n);
                }
                if (reverse_edges[cur].size() >= 5) continue;
            }
        }
    }
}

template<typename dist_t>
void hnswlib::HierarchicalNSW<dist_t>::degree_adjust(int eo, int ei) {
    if (!config->use_degree_adjust) return ;
    cout << __FILE__ << " " << __LINE__ << " apply degree adjustment\n";

    double avg_indegree_before = 0;
    double avg_indegree_after = 0;


    // reconstructGraphWithConstraint(eo, ei);
    reconstructGraph(eo, ei);
}