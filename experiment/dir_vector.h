#pragma once

#include <vector>
#include <iostream>
#include <emmintrin.h>
#include <immintrin.h>


#include <faiss/IndexPQ.h>
#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <iostream>
#include <vector>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()
#include <bits/stdc++.h>

// #include <faiss/IndexPQ.h>
// #include <faiss/Index.h>
// #include <faiss/MetricType.h>
// #include <iostream>
// #include <vector>
// #include <cstdlib>  // For rand() and srand()
// #include <ctime>    // For time()
// #include <bits/stdc++.h>
// #include "../hnswlib/hnswlib.h"

namespace dir_vector {

class Dir_Vector {
public:
    static int dim;
    static int vector_len; // vector_len = ceil(dim/32)
    int len;

    static void init(int _dim);
    Dir_Vector(size_t size);
    ~Dir_Vector();

    void calc_dir_vector_int8(const void *point_data, const void *neighbor_data, int id);
    void calc_dir_vector_float(const void *point_data, const void *neighbor_data, int id);
    void print_dir_vector(int id);
    uint32_t *dir_vector_data(int id);
    uint32_t calc_dis(uint32_t *vec1, uint32_t *vec2);
    uint32_t calc_dis_with_mask(uint32_t *vec1, uint32_t *vec2, uint32_t *mask);
    std::vector<uint32_t> get_mask_int8(const void *data1, const void *data2);
    std::vector<uint32_t> get_mask_float(const void *data1, const void *data2);

private:
    uint32_t *data;
};

}

// class PQDist {
// public:
//     PQDist() = default;
//     ~PQDist() = default;

//     PQDist(int _d, int _m, int _nbits);
//     int d, m, nbits;
//     int code_nums;
//     int d_pq;
//     std::vector<uint8_t> codes;
//     std::unique_ptr<faiss::IndexPQ> indexPQ;
//     // std::unique_ptr<hnswlib::SpaceInterface<float> > space;

//     void train(int N, std::vector<float> &xb);

//     std::vector<int> get_centroids_id(int id);
//     float* get_centroid_data(int quantizer, int code_id);

//     float calc_dist(int d, float *vec1, float *vec2);
//     float calc_dist_pq(int data_id, float *qdata, bool use_cache);


//     std::vector<float> pq_dist_cache;
//     void clear_pq_dist_cache();

//     std::vector<float> qdata;
//     bool use_cache;
//     void load_query_data(const float *_qdata, bool _use_cache);
//     float calc_dist_pq_loaded(int data_id);
// };

