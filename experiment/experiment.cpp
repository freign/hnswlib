#include <bits/stdc++.h>
#include "data_loader.h"

#include "../hnswlib/hnswlib.h"
#include "calc_group_truth.h"
#include "recall_test.h"
#include "ArgParser.h"
#include "config.h"
#include "statis_tasks.h"
#include "dir_vector.h"
#include "k_means.h"
#include "pq_dist.h"

using namespace std;
using DATALOADER::DataLoader;



template<typename dist_t>
void begin_tst(Tester<dist_t> *rt, Config *config) {
    config->test_enter_point_dis = 1;
    rt->test();
    // rt->test_faiss();
    // rt->test_waste_cands();
    // rt->test_used_neighbor_dist();
    // rt->test_dir_vector();
    // rt->test_distribution();
    // rt->test_k_means();
    // rt->test_degree_adjustment();
    // rt->test_reverse();
    // rt->test_multi_ep();
    // rt->test_ep_dist_calc();
    cout << "avg ep dist = " << config->ep_dis_tot / config->search_knn_times << '\n';
    delete rt;
}


int main(int argc, char *argv[]) {
    CommandLineOptions opt = ArgParser(argc, argv);
    int max_elements = opt.maxElements;

    int M = 16;
    int ef_construction = 200;

    DataLoader *data_loader;
    DataLoader *query_data_loader;

    GroundTruth::GT_Loader *gt_loader;

    Config *config = new Config();

    if (opt.dataName == "bigann") {
        data_loader = new DataLoader("u8", opt.maxElements, opt.point_data_path, "bigann");
        query_data_loader = new DataLoader("u8", 0, opt.query_data_path, "bigann");
        gt_loader = new GroundTruth::GT_Loader(opt.dataDir, data_loader, query_data_loader);
        hnswlib::SpaceInterface<int> *space = new hnswlib::L2SpaceI(data_loader->get_dim());

        auto *rt = new Tester<int>(&opt, data_loader, query_data_loader, gt_loader, space, "u8", M, config);
        begin_tst(rt, config);

    } else if (opt.dataName == "yandex") {
        data_loader = new DataLoader("f", opt.maxElements, opt.point_data_path, "yandex");
        query_data_loader = new DataLoader("f", 0, opt.query_data_path, "yandex");
        gt_loader = new GroundTruth::GT_Loader(opt.dataDir, data_loader, query_data_loader);
        hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(data_loader->get_dim());

        auto *rt = new Tester<float>(&opt, data_loader, query_data_loader, gt_loader, space, "f", M, config);
        begin_tst(rt, config);

    } else if (opt.dataName == "gist") {
        data_loader = new DataLoader("f", opt.maxElements, opt.point_data_path, "gist");
        query_data_loader = new DataLoader("f", 0, opt.query_data_path, "gist");
        gt_loader = new GroundTruth::GT_Loader(opt.dataDir, data_loader, query_data_loader);
        hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(data_loader->get_dim());

        auto *rt = new Tester<float>(&opt, data_loader, query_data_loader, gt_loader, space, "f", M, config);
        begin_tst(rt, config);

    }
    else if(opt.dataName == "sift") {
        data_loader = new DataLoader("f", opt.maxElements, opt.point_data_path, "sift");
        query_data_loader = new DataLoader("f", 0, opt.query_data_path, "sift");
        gt_loader = new GroundTruth::GT_Loader(opt.dataDir, data_loader, query_data_loader);
        assert(data_loader->get_dim() == 128);
        hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(data_loader->get_dim());

        auto *rt = new Tester<float>(&opt, data_loader, query_data_loader, gt_loader, space, "f", M, config);
        begin_tst(rt, config);
    } else {
        cout << "data name error\n";
    }

    return 0;
}