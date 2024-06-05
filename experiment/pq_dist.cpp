#include "pq_dist.h"
#include <cstring>
#include <memory>

using namespace std;

PQDist::PQDist(int _d, int _m, int _nbits) :d(_d), m(_m), nbits(_nbits) {
    indexPQ = std::move(std::make_unique<faiss::IndexPQ>(d, m, nbits));
    code_nums = 1 << nbits;
    d_pq = _d / _m;
    if (nbits > 8) {
        cout << "Warning nbits exceeds 8: " << nbits << "\n";
    } else if (8 % nbits != 0) {
        perror("nbits must be divided by 8!");
    }

    pq_dist_cache.resize(m * code_nums);
    qdata.resize(d);
}

void PQDist::train(int N, std::vector<float> &xb) {
    indexPQ->train(N, xb.data());
    std::cout << "code size = " << indexPQ->code_size << "\n";
    codes.resize(N * indexPQ->code_size);
    indexPQ->sa_encode(N, xb.data(), codes.data());
}

// 获取每个quantizer对应的质心id
vector<int> PQDist::get_centroids_id(int id) {

    const uint8_t *code = codes.data() + id * indexPQ->code_size;
    vector<int> centroids_id(m);
    int mask = (1<<nbits) - 1;
    int off = 0;
    for (int i = 0; i < m; i++) {
        centroids_id[i] = ((int)((*code>>off) & mask));
        off = (off + nbits) & 7; // mod 8
        if (!off) {
            code += 1; // 下一个code字节
        }
    }
    return centroids_id;
}

float* PQDist::get_centroid_data(int quantizer, int code_id) {
    return indexPQ->pq.centroids.data() + (quantizer*code_nums + code_id) * d_pq;
}

float PQDist::calc_dist(int d, float *vec1, float *vec2) {
    float ans = 0;
    for (int i = 0; i < d; i++)
        ans += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    return ans;
}

float PQDist::calc_dist_pq(int data_id, float *qdata, bool use_cache=true) {
    static const float eps = 1e-7;
    float dist = 0;
    auto ids = get_centroids_id(data_id);
    for (int q = 0; q < m; q++) {
        float d;
        if (!use_cache || pq_dist_cache[q*code_nums + ids[q]] < eps) {
            // quantizers
            float *centroid_data = get_centroid_data(q, ids[q]);
            d = calc_dist(d_pq, centroid_data, qdata + (q * d_pq));

            if (use_cache) pq_dist_cache[q*code_nums + ids[q]] = d;
        } else {
            d = pq_dist_cache[q*code_nums + ids[q]];
        }
        dist += d;
    }
    return dist;
}

void PQDist::clear_pq_dist_cache() {
    memset(pq_dist_cache.data(), 0, pq_dist_cache.size() * sizeof(float));
}

void PQDist::load_query_data(float *_qdata, bool _use_cache) {
    memcpy(qdata.data(), _qdata, sizeof(float) * d);
    clear_pq_dist_cache();
    use_cache = _use_cache;
}

float PQDist::calc_dist_pq_loaded(int data_id) {
    return calc_dist_pq(data_id, qdata.data(), use_cache);
}
