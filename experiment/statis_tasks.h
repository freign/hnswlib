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
        hnswlib::SpaceInterface<dist_t> *_space,
        string dist_t_type,
        Config *_config): M(16), ef_construction(200) {

        data_dir = opt->dataDir;
        data_path = opt->point_data_path;
        query_data_path = opt->query_data_path;
        max_elements = opt->maxElements;

        data_loader = _data_loader;
        query_data_loader = _query_data_loader;
        space = _space;
        config = _config;

        GroundTruth::calc_gt(data_dir, data_loader, query_data_loader, *space, 0);
        gt_loader = new GroundTruth::GT_Loader(data_dir, data_loader, query_data_loader);

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
        config->use_dir_vector = 0;
        if (config->use_dir_vector) {
            calc_dir_vector();
            alg_hnsw->dir_vectors_ptr = &dir_vectors;
        }
        test_vs_recall(data_dir, data_loader, query_data_loader, gt_loader, alg_hnsw, 10);
        cout << "tot dist calc = " << config->tot_dist_calc << " dist calc avoid = " << config->disc_calc_avoided << "\n";
    }

    void build_graph() {
        for (int i = 0; i < data_loader->get_elements(); i++) {
            alg_hnsw->addPoint(data_loader->point_data(i), i);
        }

        cout << "build graph finished\n";
    }

    void test_waste_cands();

    void test_used_neighbor_dist();

    void calc_dir_vector();
    void test_dir_vector();

    void test_distribution();
    vector<dir_vector::Dir_Vector*> dir_vectors;

    void test_k_means();
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
    config->test_dir_vector = 1;

    calc_dir_vector();

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
        for (auto l1: l1_dis) cout << l1 << ' '; cout << '\n';

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
    
    for (auto num: num_layer)
        cout << num << '\n';
    vector<int>().swap(levels);

    vector<int> ids_for_cluster;
    int N = data_loader->get_elements();

    ids_for_cluster.resize(N);
    for (int i = 0; i < N; i++)
        ids_for_cluster[i] = i;

    vector<vector<int> > high_layer_points(1);

    if (is_same<float, dist_t>::value) {
        for (int l = 1; l <= max_level; l++) {
            cout << "l = " << l << ' ' << ids_for_cluster.size() << ' ' << num_layer[l] << endl;

            if (num_layer[l] >= 1000) {
                random_shuffle(ids_for_cluster.begin(), ids_for_cluster.end());
                ids_for_cluster.resize(num_layer[l]);
                high_layer_points.push_back(ids_for_cluster);

            } else {

                KMeans<dist_t, float> *k_means = new KMeans<dist_t, float>(num_layer[l], ids_for_cluster.size(), data_loader->get_dim(),
                    space, data_loader, ids_for_cluster);

                k_means->run(5);

                // for (int i = 0; i < 3; i++) {
                //     KMeans<dist_t, float> *new_k_means = new KMeans<dist_t, float>(num_layer[l], ids_for_cluster.size(), data_loader->get_dim(),
                //         space, data_loader, ids_for_cluster);
                    
                //     new_k_means->run(5);
                //     if (new_k_means->tot_dist() < k_means->tot_dist())
                //         swap(new_k_means, k_means);

                //     delete new_k_means;
                // }

                auto centers = k_means->get_centers_global();
                high_layer_points.push_back(centers);
                
                ids_for_cluster = centers;
                delete k_means;
            }
            
        }
    } else if (is_same<int, dist_t>::value) {
        for (int l = 1; l <= max_level; l++) {
            KMeans<dist_t, int64_t> *k_means = new KMeans<dist_t, int64_t>(num_layer[l], ids_for_cluster.size(), data_loader->get_dim(),
                space, data_loader, ids_for_cluster);
        }
    }

    {
        cout << "build graph begin\n";
        unordered_set<int> added;
        for (int l = max_level; l > 0; l--) {
            for (auto p: high_layer_points[l]) {
                if (added.find(p) != added.end()) continue;
                added.insert(p);
                alg_hnsw->addPoint(data_loader->point_data(p), p, l);
            }
        }
        for (int i = 0; i < N; i++)
            if (added.find(i) == added.end()) {
                alg_hnsw->addPoint(data_loader->point_data(i), i, 0);
            }

        vector<vector<int> >().swap(high_layer_points);
        cout << "build graph finished\n";
    }
    test_vs_recall(data_dir, data_loader, query_data_loader, gt_loader, alg_hnsw, 10);
}