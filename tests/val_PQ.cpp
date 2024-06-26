#include <bits/stdc++.h>
#include "data_loader.h"

#include <faiss/IndexPQ.h>
#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include "../hnswlib/hnswlib.h"
#include "calc_group_truth.h"
#include "recall_test.h"
#include "ArgParser.h"
#include "config.h"
#include "statis_tasks.h"
#include "dir_vector.h"
#include "k_means.h"
#include "ivf_hnsw.h"
#include "pq_dist.h"
#include "timer.h"

using namespace std;
using DATALOADER::DataLoader;


int main(int argc, char *argv[]) {

    CommandLineOptions opt = ArgParser(argc, argv);
    int max_elements = opt.maxElements;

    int M = 16;
    int ef_construction = 200;

    DataLoader *data_loader;
    DataLoader *query_data_loader;

    GroundTruth::GT_Loader *gt_loader;

    Config *config = new Config();

    // Set random seed for reproducibility
    srand(static_cast<unsigned int>(time(0)));

    // Dimension of the vectors
    int d = 960;

    // Number of vectors to index
    int N = opt.maxElements;

    // Number of subquantizers
    int m = 60;

    // Number of bits per subquantizer
    int nbits = 4;

    

    data_loader = new DataLoader("f", opt.maxElements, opt.point_data_path, "gist");
    query_data_loader = new DataLoader("f", 0, opt.query_data_path, "gist");
    gt_loader = new GroundTruth::GT_Loader(opt.dataDir, data_loader, query_data_loader);
    hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(data_loader->get_dim());

    cout << "data prepared" << endl;
    vector<float> xb(N * d * sizeof(float));
    for (int i = 0; i < N; i++) {
        memcpy(xb.data() + (i * d * sizeof(float)), data_loader->point_data(i), d * sizeof(float));
    }
    StopW trainTimer;
    PQDist *pq_dist = new PQDist(d, m, nbits);


    pq_dist->load("../PQ/pq_100000.txt");
    for (int i = 99999; i < 100000; i++) {
        for (int j = 0; j < 120; j++)
            cout << (int)pq_dist->codes[i*pq_dist->m + j] << " ";
        cout << "\n";
    }
    // pq_dist->train(N, xb);
    // cout << "train time " << trainTimer.getElapsedTimeMicro() << endl;
    
    hnswlib::SpaceInterface<float> *space2 = new hnswlib::L2Space(data_loader->get_dim() / 2);

    // prefetch
    for (int j = 0; j < query_data_loader->get_elements(); j += 10) {
        pq_dist->load_query_data(reinterpret_cast<const float*>(query_data_loader->point_data(j)), 1);
        for (int i = 0; i < data_loader->get_elements(); i += 50) {
            // float distPQ = pq_dist->calc_dist_pq_loaded(i);
            // cout << distPQ << " " << real_dist << "\n";
            float real_dist = space->get_dist_func()(
                    query_data_loader->point_data(j), data_loader->point_data(i), space->get_dist_func_param());
        }
    }

    StopW PQTimer;
    PQTimer.reset();
    vector<float> pqs;
    for (int j = 0; j < query_data_loader->get_elements(); j += 10) {
        pq_dist->load_query_data(reinterpret_cast<const float*>(query_data_loader->point_data(j)), 1);
        for (int i = 0; i < data_loader->get_elements(); i += 100) {
            float distPQ = pq_dist->calc_dist_pq_loaded(i);
            // cout << distPQ << " " << real_dist << "\n";
            // for (int iter = 0; iter < 2; iter++)
            //     float real_dist = space2->get_dist_func()(
            //             query_data_loader->point_data(j), data_loader->point_data(i), space2->get_dist_func_param());
            pqs.push_back(distPQ);
        }
    }
    cout << "PQ time " << PQTimer.getElapsedTimeMicro() << endl;

    StopW RealTimer;
    RealTimer.reset();
    vector<float> reals;
    for (int j = 0; j < query_data_loader->get_elements(); j += 10) {
        pq_dist->load_query_data(reinterpret_cast<const float*>(query_data_loader->point_data(j)), 1);
        for (int i = 0; i < data_loader->get_elements(); i += 100) {
            float real_dist = space->get_dist_func()(
                    query_data_loader->point_data(j), data_loader->point_data(i), space->get_dist_func_param());
            reals.push_back(real_dist);
            // cout << distPQ << " " << real_dist << "\n";
        }
    }
    cout << "real time " << RealTimer.getElapsedTimeMicro() << endl;
    assert(pqs.size() == reals.size());

    float error = 0;

    // cout << "\n\n";

    for (int i = 0; i < pqs.size(); i++) {
        error += (sqrt(pqs[i]) - sqrt(reals[i])) * (sqrt(pqs[i]) - sqrt(reals[i]));
        // cout << pqs[i] << ' ' << reals[i] << "\n";
    }
    error /= pqs.size();
    cout << "error = " << error << "\n";
    
    return 0;
}