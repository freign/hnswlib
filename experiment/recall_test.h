#pragma once

#include "../hnswlib/hnswlib.h"
#include "pq_dist.h"
#include "data_loader.h"
#include "calc_group_truth.h"
#include "config.h"
#include "timer.h"
#include <fstream>

using namespace std;

Config *global_config;
template<typename dist_t>
float test_approx(
    DATALOADER::DataLoader *data_loader,
    DATALOADER::DataLoader *query_data_loader,
    GroundTruth::GT_Loader *gt_loader,
    hnswlib::HierarchicalNSW<dist_t> *appr_alg,
    size_t K) {

    PQDist a;
    
    size_t qsize = query_data_loader->get_elements();
    
    double recall = 0;
    int lst_tot_calc = 0;

    for (int i = 0; i < qsize; i++) {
        if (global_config->use_PQ) {
            // appr_alg->pq_dist->load_query_data(reinterpret_cast<const float*>(query_data_loader->point_data(i)), 1);
            appr_alg->pq_dist->load_query_data_and_cache(reinterpret_cast<const float*>(query_data_loader->point_data(i)));
        }
        dist_t nn_dist;
        dist_t nn_node;

        global_config->nn_dist = nn_dist;
        auto result = appr_alg->searchKnn(query_data_loader->point_data(i), K);
        recall += gt_loader->calc_recall(result, i, K);
        // cout << "i = " << i << " recall = " << gt_loader->calc_recall(result, i, K) << "\n";
        // cout << "i = " << i << " recursive_len = " << global_config->recursive_len << " nn_path_len = " << global_config->nn_path_len << " dist_calc_now = " << global_config->dist_calc_when_nn << "\n";

        // global_config->use_extent_neighbor = 0;


        // while (result.size()) {
        //     nn_dist = result.top().first;
        //     nn_node = result.top().second;
        //     // cout << "not gt " << nn_node << " " << nn_dist << "\n";
        //     result.pop();
        // }
        // nn_dist = sqrt(nn_dist);

        // dist_t ep_dist = sqrt(global_config->ep_dist.back());
        // cout << "nn node = " << nn_node << " nn_dist = " << nn_dist << "\n";

        // if (appr_alg->reverse_edges[ep_dist].size() < 10) {
        //     cout << "nn node = " << nn_node << ' ' << appr_alg->reverse_edges[ep_dist].size() << "\n";
        // }
        // cout << ep_dist - nn_dist << " " << global_config->tot_dist_calc - lst_tot_calc << "\n";
        // cout << global_config->tot_dist_calc - lst_tot_calc << "\n";
        // cout << "\n";
        // lst_tot_calc = global_config->tot_dist_calc;
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
    size_t K,
    Config *config,
    int m,
    int nbits) {
    
    string result_file_path;
    if(config -> use_PQ) {
        result_file_path = data_dir + "/result_" 
            + std::to_string(data_loader->get_elements()) + "_" + std::to_string(K) + "_pq" + std::to_string(m) + "_" + std::to_string(nbits) + "_noresort.res";
    } else {
        result_file_path = data_dir + "/result_" 
            + std::to_string(data_loader->get_elements()) + "_" + std::to_string(K) + "oro.res";
    }

    
    std::ofstream file(result_file_path);

    if (!file) {
        std::cerr << "file " << result_file_path << " open fail!\n";
        return ;
    }

    std::vector<size_t> efs;

    for (int i = K; i < 80; i+=5) {
        efs.push_back(i);
    }
    for (int i = 80; i < 200; i += 10) {
        efs.push_back(i);
    }
    for (int i = 200; i < 500; i += 40) {
        efs.push_back(i);
    } 

    size_t qsize = query_data_loader->get_elements();

    global_config = config;

    for (size_t ef: efs) {
        // if (ef != 180) continue;
        // cout << "ef = " << ef << "\n";

        config->high_level_dist_calc = 0;
        appr_alg->setEf(ef);
        StopW stopw = StopW();

        config->tot_dist_calc = 0;
        config->clear_test_ep();

        float recall = test_approx(data_loader, query_data_loader, gt_loader, appr_alg, K);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
        
        std::cout << "testing ef = " << ef << '\n';
        file << ef << "\t" << recall << "\t" << time_us_per_query << " us\t" << config->tot_dist_calc << "\t" << config->high_level_dist_calc << "\n";
        if (recall > 1.0) {
            file << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
        file.flush();

        // for (auto pr: config->ep_nn_pair) {
        //     cout << pr.first << " " << pr.second << "\n";
        // }
    }
}
