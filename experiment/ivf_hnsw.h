#pragma once

#include "k_means.h"
#include "../hnswlib/hnswlib.h"

template<typename dist_t, typename sum_type_t>
class IVFHNSW: public KMeans<dist_t, sum_type_t> {

public:
    IVFHNSW(int _k, int _N, int _dim, hnswlib::SpaceInterface<dist_t> *_space, DataLoader *_data_loader, vector<int> &ids)
        :KMeans<dist_t, sum_type_t>(_k, _N, _dim, _space, _data_loader, ids) {

        

    }

    vector<uint32_t> searchKnn(const void* data_point, int knn, int nprobe) {

    }
    
    vector<uint32_t> ivfflat_search(const void* data_point, int knn, int nprobe) {
        return KMeans<dist_t, sum_type_t>::searchKnn(data_point, knn, nprobe);
    }
    vector<uint32_t> ivfflat_search_prune(const void* data_point, int knn, int nprobe, float epsilon) {
        return KMeans<dist_t, sum_type_t>::searchKnn_prune(data_point, knn, nprobe, epsilon);
    }
};
