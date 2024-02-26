#pragma once

#include <bits/stdc++.h>

namespace DATALOADER {
class DataLoader {

public:
    DataLoader(std::string data_type_name, uint32_t _max_elements, std::string _data_path);
    ~DataLoader();

    const void *point_data(int id) {
        if (id >= elements) {
            std::cerr << "only have " << elements << " points\n";
            return nullptr;
        }
        return reinterpret_cast<const void*>(data + id * dim * data_type_len);
    }

    uint32_t get_elements();
    uint32_t get_dim();
private:

    int dim;
    uint32_t elements;
    uint64_t data_type_len;
    uint64_t tot_data_size;
    std::string data_path;
    void *data;
};
}