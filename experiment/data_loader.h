#pragma once

#include <bits/stdc++.h>
#include "../hnswlib/hnswlib.h"
#include "pq_dist.h"
namespace DATALOADER {
class DataLoader {

public:
    DataLoader(std::string data_type_name, uint32_t _max_elements, std::string _data_path, std::string _data_name);
    ~DataLoader();

    const void *point_data(int id) {
        if (id >= elements) {
            std::cerr << "only have " << elements << " points\n";
            return nullptr;
        }
        return reinterpret_cast<const void*>(data + id * (dim * data_type_len + offset_per_elem));
    }
    void print_point_data_int8(int id) {
        const uint8_t *point = reinterpret_cast<const uint8_t*>(point_data(id));
        for (int i = 0; i < dim; i++)
            std::cout << (uint32_t)*(point + i) << ' '; 
            std::cout << '\n';
    }
    uint32_t get_elements();
    uint32_t get_dim();
    void free_data();
private:

    void *data;
    int dim;
    std::string data_name;
    uint32_t elements;
    uint64_t data_type_len;
    uint64_t tot_data_size;
    std::string data_path;
    uint64_t offset_per_elem = 0;
};

template<typename dist_t>
float dist_loaders(DataLoader *loader1, int i, DataLoader *loader2, int j, hnswlib::SpaceInterface<dist_t> *space) {
    return space->get_dist_func()(loader1->point_data(i), loader2->point_data(j), space->get_dist_func_param());
}


}