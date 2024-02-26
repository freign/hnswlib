#pragma once

#include "../hnswlib/hnswlib.h"
#include "data_loader.h"
#include "calc_group_truth.h"
#include "config.h"
#include "timer.h"
#include <fstream>

template<typename dist_t>
float test_approx(
    DATALOADER::DataLoader *query_data_loader,
    GroundTruth::GT_Loader *gt_loader,
    hnswlib::HierarchicalNSW<dist_t> *appr_alg,
    size_t K) {

    size_t qsize = query_data_loader->get_elements();
    
    double recall = 0;
    for (int i = 0; i < qsize; i++) {
        auto result = appr_alg->searchKnn(query_data_loader->point_data(i), K);
        recall += gt_loader->calc_recall(result, i, K);
    }
    return recall / qsize;
}

template<typename dist_t>
void test_vs_recall(
    std::string data_dir,
    DATALOADER::DataLoader *data_loader,
    DATALOADER::DataLoader *query_data_loader,
    GroundTruth::GT_Loader *gt_loader,
    hnswlib::HierarchicalNSW<dist_t> *appr_alg,
    size_t K) {

    std::string result_file_path = data_dir + "/result_" 
        + std::to_string(data_loader->get_elements()) + "_" + std::to_string(K) + ".res";
    
    std::ofstream file(result_file_path);

    if (!file) {
        std::cerr << "file " << result_file_path << " open fail!\n";
        return ;
    }

    std::vector<size_t> efs;

    for (int i = K; i < 30; i++) {
        efs.push_back(i);
    }
    for (int i = 30; i < 100; i += 10) {
        efs.push_back(i);
    }
    for (int i = 100; i < 500; i += 40) {
        efs.push_back(i);
    } 

    size_t qsize = query_data_loader->get_elements();

    for (size_t ef: efs) {
        appr_alg->setEf(ef);
        StopW stopw = StopW();

        float recall = test_approx(query_data_loader, gt_loader, appr_alg, K);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
        
        std::cout << "testing ef = " << ef << '\n';
        file << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
        if (recall > 1.0) {
            file << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
        file.flush();

    }

}
