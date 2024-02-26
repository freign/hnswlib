#include <bits/stdc++.h>
#include "data_loader.h"

#include "hnswlib.h"
#include "calc_group_truth.h"
#include "recall_test.h"
#include "ArgParser.h"
#include "config.h"

using namespace std;
using DATALOADER::DataLoader;

template<typename dist_t>
class RecallTester {
public:
    RecallTester(
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
        config->dist_func_time = 0;
        for (int i = 0; i < data_loader->get_elements(); i++) {
            alg_hnsw->addPoint(data_loader->point_data(i), i);
        }

    }

    ~RecallTester() {
        delete data_loader;
        delete query_data_loader;
        delete gt_loader;
        delete alg_hnsw;
        delete space;
    }

    void test() {
        test_vs_recall(data_dir, data_loader, query_data_loader, gt_loader, alg_hnsw, 10);
    }
private:
    string data_dir;
    string data_path;
    string query_data_path;
    DataLoader *data_loader;
    DataLoader *query_data_loader;
    GroundTruth::GT_Loader *gt_loader;
    hnswlib::HierarchicalNSW<dist_t> *alg_hnsw;
    hnswlib::SpaceInterface<dist_t> *space;
    Config *config;
    int M;
    int ef_construction = 200;
    int max_elements;    
};

template<typename dist_t>
void begin_tst(RecallTester<dist_t> *rt, Config *config) {

    config->clear_search_time();
    config->clear_dist_func_time();
    config->tick = 0;

    rt->test();

    cout << "search time: " << config->search_time << '\n';
    cout << "dist time: " << config->dist_func_time << '\n';
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
    config->test_dist_func_time = 1;

    if (opt.dataName == "bigann") {
        data_loader = new DataLoader("u8", opt.maxElements, opt.point_data_path);
        query_data_loader = new DataLoader("u8", 0, opt.query_data_path);
        hnswlib::SpaceInterface<int> *space = new hnswlib::L2SpaceI(data_loader->get_dim());
        auto *rt = new RecallTester<int>(&opt, data_loader, query_data_loader, space, "u8", config);

        begin_tst(rt, config);
    } else if (opt.dataName == "yandex") {
        data_loader = new DataLoader("f", opt.maxElements, opt.point_data_path);
        query_data_loader = new DataLoader("f", 0, opt.query_data_path);
        hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(data_loader->get_dim());
        auto *rt = new RecallTester<float>(&opt, data_loader, query_data_loader, space, "f", config);

        begin_tst(rt, config);
    }

    return 0;
}