#include "pq_dist.h"
#include <cstring>
#include <memory>
#include <immintrin.h>
#include <emmintrin.h>

using namespace std;

PQDist::PQDist(int _d, int _m, int _nbits) :d(_d), m(_m), nbits(_nbits) {
    //indexPQ = std::move(std::make_unique<faiss::IndexPQ>(d, m, nbits));
    code_nums = 1 << nbits;
    d_pq = _d / _m;
    table_size = m * code_nums;
    if (nbits > 8) {
        cout << "Warning nbits exceeds 8: " << nbits << "\n";
    } else if (8 % nbits != 0) {
        perror("nbits must be divided by 8!");
    }

    // pq_dist_cache.resize(m * code_nums);
    pq_dist_cache_data = (float*)aligned_alloc(64, sizeof(float) * table_size);
    pq_dist_cache_data_uint8 = (uint8_t*)aligned_alloc(64, sizeof(uint8_t) * table_size);
    if (pq_dist_cache_data == nullptr || pq_dist_cache_data_uint8 == nullptr) {
        perror("Not enough memory for pq_dist_cache_data");
        exit(-1);
    }
    
    qdata.resize(d);

    space = std::move(unique_ptr<hnswlib::SpaceInterface<float>> (new hnswlib::L2Space(d_pq)));
    simd_registers = std::move(std::make_unique<__m512i[]>(((this->m + 7) / 8) * 2));

}

PQDist::~PQDist() {
    if (pq_dist_cache_data != nullptr)
        free(pq_dist_cache_data);
    if (pq_dist_cache_data_uint8 != nullptr)
        free(pq_dist_cache_data_uint8);
    
}

void PQDist::train(int N, std::vector<float> &xb) {
    /* indexPQ->train(N, xb.data());
    std::cout << "code size = " << indexPQ->code_size << "\n";
    codes.resize(N * indexPQ->code_size);
    indexPQ->sa_encode(N, xb.data(), codes.data());

    centroids.assign(indexPQ->pq.centroids.begin(), indexPQ->pq.centroids.end()); */
}

// 获取每个quantizer对应的质心id
vector<uint8_t> PQDist::get_centroids_id(int id) {
    const uint8_t *code = codes.data() + id * (this->m * this->nbits / 8);
    vector<uint8_t> centroids_id(m * nbits / 8, 0);
    memcpy(centroids_id.data(), code, m * sizeof(uint8_t) * nbits / 8);
   /*  if (nbits == 8) {
        size_t num_ids = m;  // 每8bit一个id
        size_t num_bytes = num_ids;
        centroids_id.resize(num_ids);

        size_t i = 0;
        size_t j = 0;

        for (; i + 32 <= num_bytes; i += 32) {
            __m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(code + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(centroids_id.data() + i), input);
        }
        for (; i < num_bytes; i++)
            centroids[i] = code[i];
    } else {
        size_t num_ids = m;  // 每4bit一个id
        size_t num_bytes = (num_ids + 1) / 2;  // 每个字节包含两个ID
        centroids_id.resize(num_ids);

        size_t i = 0;
        size_t j = 0;

        // 使用AVX2指令处理每32个字节（256位）
        for (; i + 32 <= num_bytes; i += 32, j += 64) {
            __m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(code + i));

            // 提取低4位
            __m256i low_mask = _mm256_set1_epi8(0x0F);
            __m256i low = _mm256_and_si256(input, low_mask);

            // 提取高4位
            __m256i high = _mm256_srli_epi16(input, 4);
            high = _mm256_and_si256(high, low_mask);

            // 交错存储低4位和高4位

            __m256i interleave_lo = _mm256_unpacklo_epi8(low, high);
            __m256i interleave_hi = _mm256_unpackhi_epi8(low, high);

            __m128i seg0 = _mm256_extracti128_si256(interleave_lo, 0);
            __m128i seg2 = _mm256_extracti128_si256(interleave_lo, 1);
            __m128i seg1 = _mm256_extracti128_si256(interleave_hi, 0);
            __m128i seg3 = _mm256_extracti128_si256(interleave_hi, 1);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(centroids_id.data() + j), seg0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(centroids_id.data() + j + 16), seg1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(centroids_id.data() + j + 32), seg2);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(centroids_id.data() + j + 48), seg3);


        }

        // 处理剩余的数据
        for (; i < num_bytes; ++i, j += 2) {
            centroids_id[j] = code[i] & 0x0F;           // 提取低4位
            centroids_id[j + 1] = (code[i] >> 4) & 0x0F; // 提取高4位
        }
    } */
    return centroids_id;
}

float* PQDist::get_centroid_data(int quantizer, int code_id) {
    // return indexPQ->pq.centroids.data() + (quantizer*code_nums + code_id) * d_pq;
    return centroids.data() + (quantizer*code_nums + code_id) * d_pq;
}

inline float PQDist::calc_dist(int d, float *vec1, float *vec2) {
    assert(d == *reinterpret_cast<int*>(space->get_dist_func_param()));
    return space->get_dist_func()(vec1, vec2, space->get_dist_func_param());
}

float PQDist::calc_dist_pq(int data_id, float *qdata, bool use_cache=true) {
    static const float eps = 1e-7;
    float dist = 0;
    auto ids = get_centroids_id(data_id);
    for (int q = 0; q < m; q++) {
        float d;
        if (!use_cache || pq_dist_cache_data[q*code_nums + ids[q]] < eps) {
            // quantizers
            float *centroid_data = get_centroid_data(q, ids[q]);
            d = calc_dist(d_pq, centroid_data, qdata + (q * d_pq));

            if (use_cache) pq_dist_cache_data[q*code_nums + ids[q]] = d;
        } else {
            d = pq_dist_cache_data[q*code_nums + ids[q]];
        }
        dist += d;
    }
    return dist;
}

void PQDist::clear_pq_dist_cache() {
    // memset(pq_dist_cache.data(), 0, pq_dist_cache.size() * sizeof(float));
}

void PQDist::load_query_data(const float *_qdata, bool _use_cache) {
    memcpy(qdata.data(), _qdata, sizeof(float) * d);
    clear_pq_dist_cache();
    use_cache = _use_cache;
}

void PQDist::load_query_data_and_cache(const float *_qdata) {
    memcpy(qdata.data(), _qdata, sizeof(float) * d);
    clear_pq_dist_cache();
    use_cache = true;
    maxx = 0;
    minx = 1 << 10;

    for(int i = 0; i < m * code_nums; i++) {
        pq_dist_cache_data[i] = calc_dist(d_pq, get_centroid_data(i / code_nums, i % code_nums), qdata.data() + (i / code_nums) * d_pq);
        maxx = std::max(maxx, pq_dist_cache_data[i]);
        minx = std::min(minx, pq_dist_cache_data[i]);
    }

    // pq_dist_cache_data = pq_dist_cache.data();
    // this->offset = minn;
    // this->scale = (maxx - minn) / 255.0;
    scale = (maxx - minx) / 255.0;
    for(int i = 0; i < m * code_nums; i++) {
     pq_dist_cache_data_uint8[i] = std::round((pq_dist_cache_data[i] - minx) / scale);
    }
    
    set_registers();
   /*  _mm_prefetch(pq_dist_cache_data, _MM_HINT_NTA);

    size_t prefetch_size = 128;
    for (int i = 0; i < table_size * 4; i += prefetch_size / 4) {
        _mm_prefetch(pq_dist_cache_data + i, _MM_HINT_NTA);
    } */

}
float PQDist::calc_dist_pq_(int data_id, float *qdata, bool use_cache=true) {
    float dist = 0;
    auto ids = get_centroids_id(data_id);
    for (int q = 0; q < m; q++) {
        dist += pq_dist_cache_data[q*code_nums + ids[q]];
    }
    return dist;
}

float PQDist::calc_dist_pq_simd(int data_id, float *qdata, bool use_cache) {
    float dist = 0;
    std::vector<uint8_t> ids = get_centroids_id(data_id);
    __m256 simd_dist = _mm256_setzero_ps();
    int q;
    const int stride = 8;
    
    for (q = 0; q <= m - stride; q += stride) {
        // 加载8个uint8_t值到128位寄存器
        __m128i id_vec_128 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ids.data() + q));
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

    // 使用水平加法累加simd_dist中的值
    simd_dist = _mm256_hadd_ps(simd_dist, simd_dist);
    simd_dist = _mm256_hadd_ps(simd_dist, simd_dist);

    float dist_array[8];
    _mm256_storeu_ps(dist_array, simd_dist);
    dist += dist_array[0] + dist_array[4];

    // 处理剩余的元素
    for (; q < m; q++) {
        dist += pq_dist_cache_data[q * code_nums + ids[q]];
        // dist += pq_dist_cache[q * code_nums + code[q]];
    }

    return dist;
}



float PQDist::calc_dist_pq_loaded(int data_id) {
    return calc_dist_pq(data_id, qdata.data(), use_cache);
}
float PQDist::calc_dist_pq_loaded_(int data_id) {
    return calc_dist_pq_(data_id, qdata.data(), use_cache);
}
float PQDist::calc_dist_pq_loaded_simd(int data_id) {
    return calc_dist_pq_simd(data_id, qdata.data(), use_cache);
}

float PQDist::calc_dist_pq_loaded_simd_scale(int data_id) {

    // return dist * this->scale + this->offset;
}

void PQDist::load(string filename) {
    //
    ifstream fin(filename, std::ios::binary);
    if (fin.is_open() == false) {
        cout << "open " << filename << " fail\n";
        exit(-1);
    }
    // GIST
    // n d m nbit
    // int [n * m]
    // float [2^nbits * d]
    int N;
    fin.read(reinterpret_cast<char*>(&N), 4);
    fin.read(reinterpret_cast<char*>(&d), 4);
    fin.read(reinterpret_cast<char*>(&m), 4);
    fin.read(reinterpret_cast<char*>(&nbits), 4);
    cout << "load: " << N << ' ' << d << " " << m << " " << nbits << endl;
    assert(8 % nbits == 0);
    code_nums = 1 << nbits;

    d_pq = d / m;
    space = std::move(unique_ptr<hnswlib::SpaceInterface<float>> (new hnswlib::L2Space(d_pq)));
    table_size = m * code_nums;

    if (pq_dist_cache_data != nullptr)
        free(pq_dist_cache_data);
    pq_dist_cache_data = (float*)aligned_alloc(64, sizeof(float) * table_size);

    // pq_dist_cache.resize(m * code_nums);

    centroids.resize(code_nums * d);
    fin.read(reinterpret_cast<char*>(centroids.data()), 4 * centroids.size());

    codes.resize(N / 8 * m * nbits);
    
    fin.read(reinterpret_cast<char*>(codes.data()), codes.size());

    // cout << "codes " << codes.size() << "\n";
    // for (int i = 0; i < m/2; i++) {
    //     cout << (int)codes[i] << ' ';
    // }
    // cout << "\n";

    

    // auto code = get_centroids_id(0);
    // for (int i = 0; i < m; i++) {
    //     cout << (int)code[i] << ' ';
    // }
    // cout << "\n";
    // exit(0);

    fin.close();
}

vector<int> PQDist::encode_query(float *query) {
    // return indexPQ->pq.centroids.data() + (quantizer*code_nums + code_id) * d_pq;
    vector<int> res;
    for (int q = 0; q < m; q++) {
        int min_id = 0;
        float min_dist = 1e9;
        for (int i = 0; i < code_nums; i++) {
            float d = calc_dist(d_pq, get_centroid_data(q, i), query + (q * d_pq));
            if (d < min_dist) {
                min_dist = d;
                min_id = i;
            }
        }
        res.push_back(min_id);
    }
    return res;
}

void PQDist::construct_distance_table() {
    distance_table.resize(m);
    for (int i = 0; i < m; i++) {
        distance_table[i].resize(1 << nbits);
        for (int j = 0; j < 1 << nbits; j++) {
            distance_table[i][j].resize(1 << nbits);
            for (int k = 0; k < 1 << nbits; k++) {
                distance_table[i][j][k] = calc_dist(d_pq, get_centroid_data(i, j), get_centroid_data(i, k));
            }
        }
    }
}
float PQDist::calc_dist_pq_from_table(int data_id, vector<int>& qids) {
    float dist = 0;
    auto ids = get_centroids_id(data_id);
    for (int q = 0; q < m; q++) {
        dist += distance_table[q][ids[q]][qids[q]];
    }
    return dist;
}

void PQDist::extract_centroid_ids(int n) {
    centroid_ids.resize(n * m * nbits/8);
    for (int i = 0; i < n; i++) {
        auto ids = get_centroids_id(i);
        memcpy(centroid_ids.data() + i*m*nbits/8, ids.data(), m*nbits/8);
    }
}

void PQDist::extract_neighbor_centroid_ids(uint8_t* &result, int *neighbors, int size) {
    result = (uint8_t*)aligned_alloc(64, sizeof(uint8_t) * ((size + 15) & (~15)) * m * nbits / 8);
    memset(result, 0, sizeof(uint8_t) * ((size + 15) & (~15)) * m * nbits / 8);
    for (int i = 0; i < size; i++) {
        int neighbor = neighbors[i];
        memcpy(result + i*m*nbits/8, centroid_ids.data() + neighbor*m*nbits/8, m*nbits/8);
    }
}

void PQDist::calc_dist_ultimate(uint8_t *encodes, int size, float *dists) {

    __m512i mask = _mm512_set1_epi8(0x0F);
    __m512 scale_f = _mm512_set1_ps(scale);
    __m512 minx_f = _mm512_set1_ps(minx);
    __m512i index = _mm512_setzero_si512();
    __m512i dist = _mm512_setzero_si512();
    int batch_size = 16;
    int scale_size = (size + 15) & (~15);

    // print_m512i_uint8(simd_registers[20]);
    for (int b = 0; b < scale_size; b += batch_size)
    {
        uint8_t *b_encodes = encodes + b * m * nbits / 8;
        float *b_dists = dists + b;
        

        __m512i acc = _mm512_setzero_si512();
        for (int i = 0; i < m; i += 8)
        {

            index = _mm512_load_si512(b_encodes + i * batch_size * nbits / 8);
            // print_m512i_uint8(index);
            __m512i partial_id = _mm512_and_si512(index, mask);
            // print_m512i_uint8(partial_id);

            __m512i dist = _mm512_shuffle_epi8(simd_registers[2 * (i / 8)], partial_id);
            // print_m512i_uint8(dist);
            extract_and_upcast_and_add(acc, dist);
            // 饱和加法
            // dist = _mm512_adds_epu8(dist, partial_dist);
            // 将index右移4位。
            index = _mm512_srli_epi16(index, 4);
            partial_id = _mm512_and_si512(index, mask);
            // print_m512i_uint8(partial_id);
            dist = _mm512_shuffle_epi8(simd_registers[2 * (i / 8) + 1], partial_id);
            // print_m512i_uint8(dist);
            extract_and_upcast_and_add(acc, dist);
        }

        __m512 acc_f = _mm512_cvtepi32_ps(acc);

        acc_f = _mm512_mul_ps(acc_f, scale_f);
        acc_f = _mm512_add_ps(acc_f, minx_f);
        // 将acc存入dists
        _mm512_store_ps(b_dists, acc_f);

    }

    

}
