#pragma once
#include <bits/stdc++.h>

#include "data_loader.h"

#include "../hnswlib/hnswlib.h"
#include "calc_group_truth.h"
#include "recall_test.h"
#include "ArgParser.h"
#include "config.h"
#include "dir_vector.h"
#include "k_means.h"

using namespace std;

template<typename dist_t>
class Tester {
public:
    Tester(
        CommandLineOptions *opt,
        DataLoader *_data_loader,
        DataLoader *_query_data_loader,
        GroundTruth::GT_Loader *_gt_loader,
        hnswlib::SpaceInterface<dist_t> *_space,
        string dist_t_type,
        int _M,
        Config *_config): M(_M), ef_construction(200) {

        data_dir = opt->dataDir;
        data_path = opt->point_data_path;
        query_data_path = opt->query_data_path;
        max_elements = opt->maxElements;

        data_loader = _data_loader;
        query_data_loader = _query_data_loader;
        gt_loader = _gt_loader;
        space = _space;
        config = _config;

        GroundTruth::calc_gt(data_dir, data_loader, query_data_loader, *space, 0);
        // gt_loader = new GroundTruth::GT_Loader(data_dir, data_loader, query_data_loader);

        alg_hnsw = new hnswlib::HierarchicalNSW<dist_t>(space, max_elements, M, ef_construction);
        alg_hnsw->config = config;

        // build_graph();

        // data_loader->free_data();
    }

    ~Tester() {
        if (data_loader != nullptr)
            delete data_loader;
        delete query_data_loader;
        delete gt_loader;
        delete alg_hnsw;
        delete space;
    }

    void test() {
        build_graph();
        alg_hnsw->get_extent_neighbors();
        config->statis_recursive_len = 1;
        config->use_dir_vector = 0;
        config->statis_ep_dis = 1;
        config->statis_ep_nn_pair = 1;
        config->high_level_dist_calc = 0;
        config->test_nn_path_len = 1;
        config->use_extent_neighbor = 0;
        
        if (config->use_extent_neighbor) {
            cout << "use extent neighbors\n";
        }
        if (config->use_dir_vector) {
            cout << "use dir vector\n";
            calc_dir_vector();
            alg_hnsw->dir_vectors_ptr = &dir_vectors;
        }

        config->test_ep_with_calc = 1;
        config->ep_dist_limit = 0.4;

        if (config->use_degree_adjust) {
            cout << "use degree adjustment\n";
            alg_hnsw->degree_adjust(16, 64);
        }

        config->test_bruteforce_ep = 0;
        if (config->test_bruteforce_ep) {
            cout << "test brute force ep\n";
        }

        alg_hnsw->get_neighbors();
        test_vs_recall(data_dir, data_loader, query_data_loader, gt_loader, alg_hnsw, 10, config);
        cout << "query elements: " << query_data_loader->get_elements() << "\n";
        cout << "max level = " << config->max_level << "\n";
        cout << "tot dist calc = " << config->tot_dist_calc << " dist calc avoid = " << config->disc_calc_avoided << "\n";
    }

    void build_graph() {
        for (int i = 0; i < data_loader->get_elements(); i++) {
            alg_hnsw->addPoint(data_loader->point_data(i), i);
        }
        alg_hnsw->calc_neighbor_dist();


        // vector<int> tem;
        // for (int i = 0; i < max_elements; i++) {
        //     if (alg_hnsw->element_levels_[i] == 1) tem.push_back(i);
        // }
        // for (int i = 0; i < tem.size(); i+=10) {
        //     for (int j = i+1; j < tem.size(); j++) {
        //         dist_t d = space->get_dist_func()(data_loader->point_data(tem[i]),
        //             data_loader->point_data(tem[j]), space->get_dist_func_param());
        //         cout << d << "\n";
        //     }
        // }
        // exit(0);

        cout << "build graph finished\n";
    }

    void test_waste_cands();

    void test_used_neighbor_dist();

    void calc_dir_vector();
    void test_dir_vector();

    void test_distribution();
    vector<dir_vector::Dir_Vector*> dir_vectors;

    void test_k_means();

    void test_degree_adjustment();
    void statis_indegree() {
        build_graph();
        alg_hnsw->statis_indegree();
    }

    void test_reverse();

    void test_multi_ep();

    void test_ep_dist_calc();
private:
    string data_dir;
    string data_path;
    string query_data_path;
    DATALOADER::DataLoader *data_loader;
    DATALOADER::DataLoader *query_data_loader;
    GroundTruth::GT_Loader *gt_loader;
    hnswlib::HierarchicalNSW<dist_t> *alg_hnsw;
    hnswlib::SpaceInterface<dist_t> *space;
    Config *config;
    int M;
    int ef_construction = 200;
    int max_elements;
};

template<typename dist_t>
void Tester<dist_t>::test_waste_cands() {

    config->clear_cand();
    config->statis_wasted_cand = 1;
    cout << "ef\t tot cands\t waste cands\t waste/cands\t calculated nodes\t cands/calculated\n";
    for (int ef = 10; ef <= 100; ef += 10) {
        config->clear_cand();
        alg_hnsw->setEf(ef);
        float recall = test_approx(query_data_loader, gt_loader, alg_hnsw, 10);
        cout << ef << "\t" << config->tot_cand_nodes << "\t" << config->wasted_cand_nodes << "\t" << 1.0 * config->wasted_cand_nodes / config->tot_cand_nodes 
            << config->tot_calculated_nodes << "\t" << 1.0 * config->tot_cand_nodes / config->tot_calculated_nodes << '\n';
    }
}

template<typename dist_t>
vector<int> get_hnsw_layer0_neighbors(hnswlib::HierarchicalNSW<dist_t> *hnsw, int id) {
    int *data = (int *) hnsw->get_linklist0(id);
    vector<int> neighbors;
    size_t size = hnsw->getListCount((unsigned int*)data);
    for (int i = 1; i <= size; i++) {
        neighbors.push_back(*(data + i));
    }
    return neighbors;
}
template<typename dist_t>
void Tester<dist_t>::test_used_neighbor_dist() {

    config->statis_used_neighbor_dist = 1;
    size_t qsize = query_data_loader->get_elements();
    alg_hnsw->setEf(30);

    for (int i = 0; i < qsize; i+=10) {
        config->clear_used_neighbors();
        auto ans = alg_hnsw->searchKnn(query_data_loader->point_data(i), 10);

        int nearest_neighbor = ans.top().second;

        int path_len = 0;
        for (auto id: config->used_points) {
            path_len ++ ;
            if (id == nearest_neighbor) break;
        }
        cout << path_len << '\n';
    }
}



template<typename dist_t>
void Tester<dist_t>::calc_dir_vector() {
    if (dir_vectors.size() > 0) {
        cout << "dir vectors calculated already\n";
        return ;
    }

    using dir_vector::Dir_Vector;
    Dir_Vector::init(data_loader->get_dim());
    dir_vectors.resize(data_loader->get_elements());

    for (int i = 0; i < data_loader->get_elements(); i++) {
        auto neighbors = get_hnsw_layer0_neighbors(alg_hnsw, i);
        dir_vectors[i] = new Dir_Vector(neighbors.size());

        if (is_same<dist_t, int>::value) {
            int tot = 0;
            for (auto n: neighbors) {
                dir_vectors[i]->calc_dir_vector_int8(data_loader->point_data(i),
                    data_loader->point_data(n), tot);
                tot++;
            }
        } else if (is_same<dist_t, float>::value) {
            int tot = 0;
            for (auto n: neighbors) {
                dir_vectors[i]->calc_dir_vector_float(data_loader->point_data(i),
                    data_loader->point_data(n), tot);
                tot++;
            }
        }
    }
    
}

template<typename dist_t>
void Tester<dist_t>::test_dir_vector() {
    using dir_vector::Dir_Vector;
    config->use_dir_vector = 1;
    cout << "use dir vector\n";
    calc_dir_vector();
    exit(0);
    size_t qsize = query_data_loader->get_elements();
    
    hnswlib::SpaceInterface<int> *space = new hnswlib::L2SpaceI(data_loader->get_dim());
    auto dist_func = space->get_dist_func();

    int dim = data_loader->get_dim();
    int vector_len = ceil(dim / 32);
    int v = 2;
    const uint8_t *v_data = reinterpret_cast<const uint8_t*>(data_loader->point_data(v));
    for (int i = 0; i < qsize; i+=100) {
        
        const uint8_t *q_data = reinterpret_cast<const uint8_t*>(query_data_loader->point_data(i));

        auto neighbors = get_hnsw_layer0_neighbors(alg_hnsw, v);
        Dir_Vector dvq(1);
        Dir_Vector *dv = dir_vectors[v];

        dvq.calc_dir_vector_int8(data_loader->point_data(v), query_data_loader->point_data(i), 0);

        vector<float> l1_dis(data_loader->get_dim());
        for (int j = 0; j < l1_dis.size(); j++) {
            l1_dis[j] = abs(v_data[j] - q_data[j]);
        }
        // for (auto l1: l1_dis) cout << l1 << ' '; cout << '\n';
        priority_queue<pair<int, int> > neighbor_dists;
        for (int j = 0; j < neighbors.size(); j++) {
            auto n = neighbors[j];

            int dist = dist_func(data_loader->point_data(n), query_data_loader->point_data(i), space->get_dist_func_param());
            neighbor_dists.push(make_pair(dist, j));

        }
    }

    delete space;
}

template<typename dist_t>
void Tester<dist_t>::test_distribution() {
    for (int d = 0; d < data_loader->get_dim(); d += 16) {
        for (int i = 0; i < data_loader->get_elements(); i++) {
            const uint8_t* dims = reinterpret_cast<const uint8_t*>(data_loader->point_data(i));
            cout << (uint32_t)dims[d] << '\n';
        }
    }
}

template<typename dist_t>
void Tester<dist_t>::test_k_means() {
    int max_level = 0;
    vector<int> levels;
    for (int i = 0; i < data_loader->get_elements(); i++) {
        int level = alg_hnsw->getRandomLevel(alg_hnsw->mult_);
        levels.push_back(level);
        max_level = max(max_level, level);
    }
    vector<int> num_layer(max_level+1);
    for (auto l: levels) {
        for (int j = 0; j <= l; j++)
            num_layer[j]++;
    }
    vector<int>().swap(levels);

    vector<int> ids_for_cluster;
    int N = data_loader->get_elements();

    ids_for_cluster.resize(N);
    for (int i = 0; i < N; i++)
        ids_for_cluster[i] = i;

    vector<vector<int> > high_layer_points(1);
    if (is_same<float, dist_t>::value) {
        
        hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(this->data_loader->get_dim());
        for (int l = 1; l < num_layer.size(); l++) {
            auto *kmeans = new KMeans<float, float>(num_layer[l], ids_for_cluster.size(), this->data_loader->get_dim(),
                space, this->data_loader, ids_for_cluster);

            kmeans->run();
            high_layer_points.push_back(kmeans->find_center_point_global_id());
            delete kmeans;
            ids_for_cluster = high_layer_points.back();
        }
    }
    {
        cout << "build graph begin\n";
        unordered_set<int> added;
        for (int l = max_level; l > 0; l--) {
            for (auto p: high_layer_points[l]) {
                if (added.find(p) != added.end()) continue;
                added.insert(p);
                // cout << "add " << p << " " << l << "\n";
                alg_hnsw->addPoint(data_loader->point_data(p), p, l);
            }
        }
        for (int i = 0; i < N; i++) {
            if (added.find(i) == added.end()) {
                alg_hnsw->addPoint(data_loader->point_data(i), i, 0);
            }
        }

        vector<vector<int> >().swap(high_layer_points);
        vector<int>().swap(ids_for_cluster);
        cout << "build graph finished\n";
        alg_hnsw->calc_neighbor_dist();
    }
    test_vs_recall(data_dir, data_loader, query_data_loader, gt_loader, alg_hnsw, 10, config);
}

template<typename dist_t>
void Tester<dist_t>::test_degree_adjustment() {
    M = 16;
    if (alg_hnsw != nullptr) delete alg_hnsw;
    alg_hnsw = new hnswlib::HierarchicalNSW<dist_t>(space, max_elements, M, ef_construction);
    config->use_degree_adjust = 1;
    alg_hnsw->config = config;
    test();
}

template<typename dist_t>
void Tester<dist_t>::test_reverse() {
    build_graph();
    config->use_reverse_edges = 1;

    alg_hnsw->get_reverse_edges();
    alg_hnsw->get_neighbors();

    test_vs_recall(data_dir, data_loader, query_data_loader, gt_loader, alg_hnsw, 10, config);
    cout << "tot dist calc = " << config->tot_dist_calc << " dist calc avoid = " << config->disc_calc_avoided << "\n";
}

template<typename dist_t>
void Tester<dist_t>::test_multi_ep() {
    build_graph();

    config->use_multiple_ep = 1;
    alg_hnsw->get_neighbors();


    test_vs_recall(data_dir, data_loader, query_data_loader, gt_loader, alg_hnsw, 10, config);

    cout << "tot dist calc = " << config->tot_dist_calc << " dist calc avoid = " << config->disc_calc_avoided << "\n";

}

template<typename dist_t>
void Tester<dist_t>::test_ep_dist_calc() {
    build_graph();


    config->test_ep_with_calc = 1;

    int ef = 70;
    alg_hnsw->setEf(ef);

    for (float i = 0.2; i < 4; i += 0.2) {
        config->tot_dist_calc = 0;
        config->ep_dist_limit = i * i;
        config->clear_test_ep();
        test_approx(query_data_loader, gt_loader, alg_hnsw, 10);
        cout << "limit = " << sqrt(config->ep_dist_limit) << "\tcnt = " << config->ep_in_limit_cnt << "\tavg dist = " << 1.0 * config->ep_tot_dis_calc / config->ep_in_limit_cnt << "\n";
    }


}