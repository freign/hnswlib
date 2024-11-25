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


        ivf_hnsw->run(40);

        // {
        //     float avg_recall = 0;
        //     for (int i = 0; i < query_data_loader->get_elements(); i++) {
        //         auto gt_result = gt_loader->get_knn_gt(i);
        //         auto ivf_flat_result = ivf_hnsw->ivfflat_search_prune(query_data_loader->point_data(i), 10, 20, 0.5);
        //         auto recall = gt_loader->calc_recall(ivf_flat_result, i, 5);
        //         avg_recall += recall;
        //     }
        //     cout << "avg recall = " << avg_recall / query_data_loader->get_elements() << '\n';
        //     cout << "prune_search_calc = " << 1.0 * prune_search_calc / query_data_loader->get_elements()
        //         << " prune_search_probes = " << 1.0 * prune_search_probes / query_data_loader->get_elements() << '\n';
        // }

        {
            ivf_hnsw->create_hnsws(M, ef_construction, config);
            cout << "begin test ivf-hnsw" << endl;

            float avg_recall = 0;
            for (int i = 0; i < query_data_loader->get_elements(); i++) {
                auto gt_result = gt_loader->get_knn_gt(i);
                auto ivf_hnsw_result = ivf_hnsw->ivf_hnsw_search(query_data_loader->point_data(i), 10, 20, 0.5);
                auto recall = gt_loader->calc_recall(ivf_hnsw_result, i, 5);
                avg_recall += recall;
            }
            cout << "ivf-hnsw avg recall = " << avg_recall / query_data_loader->get_elements() << '\n';
        }

        {
            // cout << "begin test prune\n";

            // for (int i = 0; i < query_data_loader->get_elements(); i++) {
            //     auto gt_result = gt_loader->get_knn_gt(i);

            //     auto ivf_flat_result = ivf_hnsw->ivfflat_search_prune(query_data_loader->point_data(i), 10, 20, 100);
            //     auto recall = gt_loader->calc_recall(ivf_flat_result, i, 1);
            //     if (recall == 1) {
            //         cout << "i = " << i << "\n";

            //         // for (auto p: ivf_flat_result) {
            //         //     cout << p << ' ' << space->get_dist_func()(data_loader->point_data(p), query_data_loader->point_data(i), space->get_dist_func_param()) << '\n';
            //         // }
            //         // cout << "------------------\n";
            //         auto gt_result = gt_loader->get_knn_gt(i);
            //         for (auto p: gt_result) {
            //             cout << p << ' ' << space->get_dist_func()(data_loader->point_data(p), query_data_loader->point_data(i), space->get_dist_func_param()) << '\n';
            //         }
            //         cout << "------------------\n";

            //         auto centroids = ivf_hnsw->find_nearest_centers(query_data_loader->point_data(i), 20);

            //         for (auto &ctr: centroids) {
            //             cout << "dis " << space->get_dist_func()(ctr.data(), query_data_loader->point_data(i), space->get_dist_func_param()) << '\n';
            //         }

            //         int nn_id = gt_result[0];
            //         float *nn_centroid = ivf_hnsw->get_assign(nn_id);

            //         cout << "triangle " << sqrt(space->get_dist_func()(data_loader->point_data(nn_id), query_data_loader->point_data(i), space->get_dist_func_param()))
            //             << " " << sqrt(space->get_dist_func()(nn_centroid, data_loader->point_data(nn_id), space->get_dist_func_param())) 
            //             << " " << sqrt(space->get_dist_func()(nn_centroid, query_data_loader->point_data(i), space->get_dist_func_param())) << '\n';

            //         cout << "nn id = " << nn_id << " nn centroid dis = " << space->get_dist_func()(nn_centroid, query_data_loader->point_data(i), space->get_dist_func_param()) << '\n';
            //         cout << "dis between nn and nn_centroid = " << space->get_dist_func()(nn_centroid, data_loader->point_data(nn_id), space->get_dist_func_param()) << '\n';
            //         for (auto &ctr: centroids) {
            //             cout << "dis between nn and centroid " << space->get_dist_func()(ctr.data(), data_loader->point_data(nn_id), space->get_dist_func_param()) << '\n';

            //         }

            //         cout << "\n\n\n";
            //         if (i > 100) exit(0);
            //     }
            // }
        }

    }

    return 0;
}