#ifndef PQ_DIST_H
#define PQ_DIST_H

#include <faiss/IndexPQ.h>
#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <iostream>
#include <vector>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()
#include <bits/stdc++.h>
#include "../hnswlib/hnswlib.h"
#include "dir_vector.h"
using namespace std;
class PQDist
{
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
    //std::unique_ptr<faiss::IndexPQ> indexPQ;
    std::unique_ptr<hnswlib::SpaceInterface<float>> space;

    void train(int N, std::vector<float> &xb);

    std::vector<uint8_t> get_centroids_id(int id);
    float *get_centroid_data(int quantizer, int code_id);

    float calc_dist(int d, float *vec1, float *vec2);
    float calc_dist_pq(int data_id, float *qdata, bool use_cache);

    void clear_pq_dist_cache();

    std::vector<float> qdata;
    bool use_cache;
    void load_query_data(const float *_qdata, bool _use_cache);
    void load_query_data_and_cache(const float *_qdata);
    float calc_dist_pq_(int data_id, float *qdata, bool use_cache);
    float calc_dist_pq_loaded(int data_id);
    float calc_dist_pq_loaded_(int data_id);
    void load(std::string filename);
    vector<int> encode_query(float *query);
    vector<vector<vector<float>>> distance_table;
    void construct_distance_table();
    float calc_dist_pq_from_table(int data_id, vector<int> &qids);
    float calc_dist_pq_simd(int data_id, float *qdata, bool use_cache);
    float calc_dist_pq_loaded_simd(int data_id);
    inline float calc_dist_pq_loaded_simd(int data_id, const uint8_t *ids);
    inline float calc_dist_pq_loaded(int data_id, const uint8_t *ids);
    float calc_dist_pq_loaded_simd_scale(int data_id);

    float *pq_dist_cache_data = nullptr;
    uint8_t* pq_dist_cache_data_uint8 = nullptr;
    float scale;
    float minx, maxx;
    std::unique_ptr<__m512i[]> simd_registers = nullptr;

    vector<uint8_t> centroid_ids;
    void extract_centroid_ids(int n);
    void extract_neighbor_centroid_ids(uint8_t* &result, int *neighbor, int size);
    inline void transpose(uint8_t* encodes, int size) {
        // encodes的大小是batch_size的倍数，
        // 将encodes由N * (m * nbits / 8)转置为(N / batch_size * m * nbits / 8) * batch_size
        // 只支持nbits = 4
        //batch_size=16，注意不足的部分。size大小有可能不是16的倍数。

        int batch_size = 16;
        //注意不足的部分。。。。。
        int scale_size = (size + 15) & (~15);
        int batch_num = scale_size / batch_size;
        //int batch_remain = size % batch_size;
        uint8_t *tmp = (uint8_t *)aligned_alloc(64, m * batch_size * nbits / 8 * sizeof(uint8_t));
        for (int i = 0; i < batch_num; i++)
        {
            for (int j = 0; j < batch_size; j++)
                for (int k = 0; k < m * nbits / 8; k++)
                {
                    tmp[k * batch_size + j] = encodes[(i * batch_size + j) * m * nbits / 8 + k];
                }
            memcpy(encodes + i * batch_size * m * nbits / 8, tmp, m * batch_size * nbits / 8);
        }
        
        
        free(tmp);
    }
    void calc_dist_ultimate(uint8_t *encodes, int size, float *dists);
    
    inline void set_registers() {
        // 设置寄存器的值

        uint8_t *temp_buffers = (uint8_t *)aligned_alloc(64, sizeof(uint8_t) * 4 * (1 << nbits));
        memset(temp_buffers, 0, sizeof(uint8_t) * 4 * (1 << nbits));
        for (int i = 0; i < m; i += 8)
        {
    
            memcpy(temp_buffers, pq_dist_cache_data_uint8 + i * code_nums, sizeof(uint8_t) * code_nums);
            if(i + 2 < m)
                memcpy(temp_buffers + code_nums, pq_dist_cache_data_uint8 + (i + 2) * code_nums, sizeof(uint8_t) * code_nums);
            if((i + 4 < m))
                memcpy(temp_buffers + 2 * code_nums, pq_dist_cache_data_uint8 + (i + 4) * code_nums, sizeof(uint8_t) * code_nums);
            if(i + 6 < m)
                memcpy(temp_buffers + 3 * code_nums, pq_dist_cache_data_uint8 + (i + 6) * code_nums, sizeof(uint8_t) * code_nums);
            simd_registers[2 * i / 8] = _mm512_load_si512(temp_buffers);
            // print_m512i_uint8(simd_registers[i / 8]);
            if(i + 1 < m)
                memcpy(temp_buffers, pq_dist_cache_data_uint8 + (i + 1) * code_nums, sizeof(uint8_t) * code_nums);
            if(i + 3 < m)
                memcpy(temp_buffers + code_nums, pq_dist_cache_data_uint8 + (i + 3) * code_nums, sizeof(uint8_t) * code_nums);
            if( i + 5 < m)
                memcpy(temp_buffers + 2 * code_nums, pq_dist_cache_data_uint8 + (i + 5) * code_nums, sizeof(uint8_t) * code_nums);
            if(i + 7 < m)
                memcpy(temp_buffers + 3 * code_nums, pq_dist_cache_data_uint8 + (i + 7) * code_nums, sizeof(uint8_t) * code_nums);
            simd_registers[2 * i / 8 + 1] = _mm512_load_si512(temp_buffers);
        }
        free(temp_buffers);
        //cout << "END SET" << endl;
    }
    
    
    inline void extract_and_upcast_and_add(__m512i& acc, __m512i& a){
        acc = _mm512_add_epi32(acc, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(a, 0)));
        acc = _mm512_add_epi32(acc, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(a, 1)));
        acc = _mm512_add_epi32(acc, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(a, 2)));
        acc = _mm512_add_epi32(acc, _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(a, 3)));
    }

};

inline float PQDist::calc_dist_pq_loaded_simd(int data_id, const uint8_t *centroids)
{
    float dist = 0;
    __m256 simd_dist = _mm256_setzero_ps();
    int q;
    if (nbits == 8)
    {
        for (q = 0; q <= m - 8; q += 8)
        {
            __builtin_prefetch(pq_dist_cache_data + q * code_nums, 0, 1);
            // 加载8个uint8_t值到128位寄存器
            __m128i id_vec_128 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(centroids + q));
            // __m128i id_vec_128 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(code + q));

            // 扩展为32位整数
            __m256i id_vec = _mm256_cvtepu8_epi32(id_vec_128);

            // 创建偏移向量
            __m256i offset_vec = _mm256_setr_epi32(
                0 * code_nums, 1 * code_nums, 2 * code_nums, 3 * code_nums,
                4 * code_nums, 5 * code_nums, 6 * code_nums, 7 * code_nums);

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
        for (int i = 0; i < 8; ++i)
        {
            dist += dist_array[i];
        }

        // 处理剩余的元素
        for (; q < m; q++)
        {
            dist += pq_dist_cache_data[q * code_nums + centroids[q]];
            // dist += pq_dist_cache[q * code_nums + code[q]];
        }
    }
    else

    {
        for(q = 0; q < m; q +=2)
        {
            uint8_t id = centroids[q/2];
            uint8_t low_id = id & 0x0F;
            uint8_t high_id = (id >> 4) & 0x0F;
            dist += pq_dist_cache_data[q * code_nums + low_id];
            dist += pq_dist_cache_data[(q + 1) * code_nums + high_id];
        }
    }

    return dist;
}

inline float PQDist::calc_dist_pq_loaded(int data_id, const uint8_t *centroids)
{
    const uint8_t *centroid_end = centroids + m;
    const float *LookUpTable = pq_dist_cache_data;
    float dist = 0;
    int idx = 0;

    // __builtin_prefetch()
    while (centroids < centroid_end)
    {
        dist += LookUpTable[*centroids];
        // LookUpTable += code_nums;
        centroids++;
        // dist += LookUpTable[idx++];
    }
    return dist;
}

#endif // !PQ_DIST_H





