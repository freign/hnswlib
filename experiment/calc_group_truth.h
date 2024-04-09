#pragma once

#include <bits/stdc++.h>
#include <fstream>
#include "data_loader.h"
#include "../hnswlib/hnswlib.h"

using DATALOADER::DataLoader;

namespace GroundTruth {

template<typename dist_t>
void calc_gt(std::string data_dir, 
    DataLoader *data_loader, 
    DataLoader *query_data_loader,
    hnswlib::SpaceInterface<dist_t> &space,
    bool overwrite) {
    
    std::string gnd_file = data_dir + "/gnd_" + std::to_string(data_loader->get_elements()) + ".gt";
    
    std::cout << "gndfile = " << gnd_file << '\n';

    std::ifstream file(gnd_file.c_str());
    if (file && !overwrite) {
        // gnd already exist.
        file.close();
        return ;
    }
    file.close();

    std::ofstream output_file(gnd_file.c_str(), std::ios::binary);

    uint32_t query_elements = query_data_loader->get_elements();
    uint32_t data_elements = data_loader->get_elements();
    uint32_t K = 10;
    output_file.write(reinterpret_cast<const char*>(&query_elements), 4);
    output_file.write(reinterpret_cast<const char*>(&K), 4);
    
    std::priority_queue<std::pair<dist_t, hnswlib::labeltype> > gnd;
    auto dist_func = space.get_dist_func();
    for (int i = 0; i < query_elements; i++) {
        while (gnd.size()) gnd.pop();
        for (int j = 0; j < data_elements; j++) {
            dist_t dis = dist_func(query_data_loader->point_data(i), data_loader->point_data(j), space.get_dist_func_param());
            gnd.push(std::make_pair(dis, j));
            if (gnd.size() > K) gnd.pop();
        }
        std::vector<uint32_t> nearestK;
        
        while (gnd.size()) {
            nearestK.push_back(gnd.top().second);
            gnd.pop();
        }
        std::reverse(nearestK.begin(), nearestK.end());
        // if (i < 10) {
        //     std::cout << "i = " << i << "\n";
        //     for (auto nn:nearestK) std::cout << nn << " "; std::cout << "\n";
        // }
        output_file.write(reinterpret_cast<const char*>(nearestK.data()), sizeof(uint32_t) * nearestK.size());
    }
}

class GT_Loader {
public:

    // 加载groud truth文件，但是不会生成该文件
    GT_Loader(std::string data_dir, 
    DataLoader *data_loader, 
    DataLoader *query_data_loader) {

        gt_path = data_dir + "/gnd_" + std::to_string(data_loader->get_elements()) + ".gt";
        file = std::ifstream(gt_path, std::ios::binary);
        if (!file) {
            std::cerr << "file " << gt_path << " not exist! calc ground truth first!\n";
            return ;
        }

        file.seekg(4);
        file.read(reinterpret_cast<char*>(&K), 4);
    }
    ~GT_Loader() {
        file.close();
    }
    std::vector<uint32_t> get_knn_gt(int id) {
        std::vector<uint32_t> ans(K);
        file.seekg(8 + id * 4 * K);
        file.read(reinterpret_cast<char*>(ans.data()), K * 4);
        return ans;
    }

    template<typename dist_t>
    double calc_recall(std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> knn, int id, int k) {
        std::vector<uint32_t> knn_id;
        while (knn.size()) {
            knn_id.push_back(knn.top().second);
            knn.pop();
        }
        return calc_recall(knn_id, id, k);
    }
    double calc_recall(std::vector<uint32_t> &knn, int id, int k) {
        auto gt = get_knn_gt(id);
        int correct = 0;
        for (int i = 0; i < k; i++) {
            for (auto nn: knn) {
                if (gt[i] == nn) {
                    correct++;
                    break;
                }
            }
        }
        
        return 1.0 * correct / k;
    }
private:
    std::string gt_path;
    std::ifstream file;
    uint32_t K;
};

}