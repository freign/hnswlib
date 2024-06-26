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
#include "ivf_hnsw.h"

using namespace std;


int main(int argc, char *argv[]) {

    CommandLineOptions opt = ArgParser(argc, argv);
    int max_elements = opt.maxElements;

    int M = 16;
    int ef_construction = 200;

    DataLoader *data_loader;
    DataLoader *query_data_loader;

    GroundTruth::GT_Loader *gt_loader;

    Config *config = new Config();

    if (opt.dataName == "yandex") {
        data_loader = new DataLoader("f", opt.maxElements, opt.point_data_path, "yandex");
        query_data_loader = new DataLoader("f", 0, opt.query_data_path, "yandex");
        hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(data_loader->get_dim());

        vector<int> ids(data_loader->get_elements());
        std::iota(ids.begin(), ids.end(), 0);
        int K = 320;
        GroundTruth::calc_gt<float>(opt.dataDir, data_loader, query_data_loader, *space, 0);
        gt_loader = new GroundTruth::GT_Loader(opt.dataDir, data_loader, query_data_loader);

        IVFHNSW<float, float> *ivf_hnsw = new IVFHNSW<float, float>(K, data_loader->get_elements(), data_loader->get_dim(),
            space, data_loader, ids);
        ivf_hnsw->run();
        ivf_hnsw->output();
        cout << "begin test\n";
        double avg_recall = 0;
        for (int i = 0; i < query_data_loader->get_elements(); i++) {
            auto gt_result = gt_loader->get_knn_gt(i);
            auto ivf_flat_result = ivf_hnsw->ivfflat_search(query_data_loader->point_data(i), 10, 3);
            avg_recall += gt_loader->calc_recall(ivf_flat_result, i, 10);
        }
        cout << "recall " << avg_recall / query_data_loader->get_elements() << " avg search = " << 1.0 * searchCalc / query_data_loader->get_elements() << '\n';
    } else if (opt.dataName == "gist") {
        data_loader = new DataLoader("f", opt.maxElements, opt.point_data_path, "gist");
        query_data_loader = new DataLoader("f", 0, opt.query_data_path, "gist");
        hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(data_loader->get_dim());

        vector<int> ids(data_loader->get_elements());
        std::iota(ids.begin(), ids.end(), 0);
        int K = 320;
        GroundTruth::calc_gt<float>(opt.dataDir, data_loader, query_data_loader, *space, 0);
        gt_loader = new GroundTruth::GT_Loader(opt.dataDir, data_loader, query_data_loader);
                IVFHNSW<float, float> *ivf_hnsw = new IVFHNSW<float, float>(K, data_loader->get_elements(), data_loader->get_dim(),
            space, data_loader, ids);

        M = 32;

        auto alg_hnsw = new hnswlib::HierarchicalNSW<float>(space, max_elements, M, ef_construction);
        config->use_degree_adjust = 1;
        alg_hnsw->config = config;

        cout << "begin add point" << endl;
        for (int i = 0; i < data_loader->get_elements(); i++) {
            alg_hnsw->addPoint(data_loader->point_data(i), i);
        }
        
        alg_hnsw->degree_adjust(32, 64);

        float avg_recall = 0;
        for (int i = 0; i < query_data_loader->get_elements(); i++) {
            auto gt_result = gt_loader->get_knn_gt(i);
            auto result = alg_hnsw->searchKnn(query_data_loader->point_data(i), 10);
            auto recall = gt_loader->calc_recall(result, i, 5);
            avg_recall += recall;
        }
        cout << "avg recall = " << avg_recall / query_data_loader->get_elements() << "\n";
    }

    return 0;
}