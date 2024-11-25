#ifndef PQ_DIST_H
#define PQ_DIST_H

#include <faiss/IndexPQ.h>
#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <iostream>
#include <vector>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()
#include <bits/stdc++.h>
#include "../hnswlib/hnswlib.h"
#include "dir_vector.h"
using namespace std;
class PQDist {
public:
    PQDist() = default;
    ~PQDist();

    PQDist(int _d, int _m, int _nbits);
    int d, m, nbits;
    int code_nums;
    int d_pq;
    size_t table_size;
    std::vector<uint8_t> codes;
    std::vector<float> centroids;
    std::unique_ptr<faiss::IndexPQ> indexPQ;
    std::unique_ptr<hnswlib::SpaceInterface<float> > space;

    void train(int N, std::vector<float> &xb);

    std::vector<uint8_t> get_centroids_id(int id);
    float* get_centroid_data(int quantizer, int code_id);

    float calc_dist(int d, float *vec1, float *vec2);
    float calc_dist_pq(int data_id, float *qdata, bool use_cache);


    void clear_pq_dist_cache();

    std::vector<float> qdata;
    bool use_cache;
    void load_query_data(const float *_qdata, bool _use_cache);
    void load_query_data_and_cache(const float *_qdata) ;
    float calc_dist_pq_(int data_id, float *qdata, bool use_cache);
    float calc_dist_pq_loaded(int data_id);
    float calc_dist_pq_loaded_(int data_id);
    void load(std::string filename);
    vector<int> encode_query(float* query);
    vector<vector<vector<float>>> distance_table;
    void construct_distance_table();
    float calc_dist_pq_from_table(int data_id, vector<int>& qids);
    float calc_dist_pq_simd(int data_id, float *qdata, bool use_cache);
    float calc_dist_pq_loaded_simd(int data_id);
    inline float calc_dist_pq_loaded_simd(int data_id, const uint8_t* ids);
    inline float calc_dist_pq_loaded(int data_id, const uint8_t* ids);
    float calc_dist_pq_loaded_simd_scale(int data_id);

    float *pq_dist_cache_data = nullptr;


    vector<uint8_t> centroid_ids;
    void extract_centroid_ids(int n);
    void extract_neighbor_centroid_ids(vector<uint8_t> &result, int *neighbor, int size);
};

inline float PQDist::calc_dist_pq_loaded_simd(int data_id, const uint8_t* centroids) {
    float dist = 0;
    __m256 simd_dist = _mm256_setzero_ps();
    int q;
    for (q = 0; q <= m - 8; q += 8) {
        __builtin_prefetch(pq_dist_cache_data + q * code_nums, 0, 1);
        // 加载8个uint8_t值到128位寄存器
        __m128i id_vec_128 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(centroids + q));
        // __m128i id_vec_128 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(code + q));

        // 扩展为32位整数
        __m256i id_vec = _mm256_cvtepu8_epi32(id_vec_128);

        // 创建偏移向量
        __m256i offset_vec = _mm256_setr_epi32(
            0 * code_nums, 1 * code_nums, 2 * code_nums, 3 * code_nums,
            4 * code_nums, 5 * code_nums, 6 * code_nums, 7 * code_nums
        );
        
        // 将偏移向量添加到id_vec中
        id_vec = _mm256_add_epi32(id_vec, offset_vec);

        // 使用gather指令从pq_dist_cache_data中获取距离值
        __m256 dist_vec = _mm256_i32gather_ps(pq_dist_cache_data + q * code_nums, id_vec, 4);

        // 累加距离值
        simd_dist = _mm256_add_ps(simd_dist, dist_vec);
    }

    // 将结果存储到数组中
    float dist_array[8];
    _mm256_storeu_ps(dist_array, simd_dist);
    for (int i = 0; i < 8; ++i) {
        dist += dist_array[i];
    }

    // 处理剩余的元素
    for (; q < m; q++) {
        dist += pq_dist_cache_data[q * code_nums + centroids[q]];
        // dist += pq_dist_cache[q * code_nums + code[q]];
    }

    return dist;
}

inline float PQDist::calc_dist_pq_loaded(int data_id, const uint8_t* centroids) {
    const uint8_t *centroid_end = centroids + m;
    const float *LookUpTable = pq_dist_cache_data;
    float dist = 0;
    int idx = 0;

    // __builtin_prefetch()
    while (centroids < centroid_end) {
        dist += LookUpTable[*centroids];
        // LookUpTable += code_nums;
        centroids++;
        // dist += LookUpTable[idx++];
    }
    return dist;
}


#endif // !PQ_DIST_H