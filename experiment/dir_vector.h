#pragma once

#include <bits/stdc++.h>
#include <emmintrin.h>
#include <immintrin.h>

namespace dir_vector {

class Dir_Vector {
public:
    static int dim;
    static int vector_len; // vector_len = ceil(dim/32)
    int len;

    static void init(int _dim) {
        dim = _dim;
        vector_len = (dim + 31) / 32;
    }
    Dir_Vector(size_t size) {
        len = size;
        data = new uint32_t [size * vector_len];
        memset(data, 0, size * vector_len * sizeof(uint32_t));
    }
    ~Dir_Vector() {
        if (data != nullptr)
            delete []data;
    }
    void calc_dir_vector_int8(const void *point_data, const void *neighbor_data, int id);
    void calc_dir_vector_float(const void *point_data, const void *neighbor_data, int id);
    void print_dir_vector(int id) {
        for (int i = 0; i < vector_len; i++) {
            for (int j = 0; j < 32; j++)
                std::cout << (data[i + id*vector_len]>>j & 1);
        }
        std::cout << '\n';
    }
    uint32_t *dir_vector_data(int id) {
        return (data + id * vector_len);
    }
    uint32_t calc_dis(uint32_t *vec1, uint32_t *vec2) {
        int ans = 0;
        const int simd_width = 4;
        int i = 0;
        for (i = 0; i + simd_width <= vector_len; i += simd_width) {
            __m128i vec1_reg = _mm_loadu_si128((__m128i*)(vec1 + i));
            __m128i vec2_reg = _mm_loadu_si128((__m128i*)(vec2 + i));

            __m128i xor_result = _mm_xor_si128(vec1_reg, vec2_reg);

            for (int j = 0; j < simd_width; ++j) {
                ans += _mm_popcnt_u32(static_cast<unsigned int>(_mm_extract_epi32(xor_result, j)));
            }
        }

        for (; i < vector_len; i++)
            ans += __builtin_popcount(vec1[i] ^ vec2[i]);
        return ans;
    }
    uint32_t calc_dis_with_mask(uint32_t *vec1, uint32_t *vec2, uint32_t *mask) {
        if ((dim & 127)) {
            std::cerr << "dim can't divide 128";
            exit(0);
        }
        int ans = 0;
        const int simd_width = 4;
        int i = 0;
        for (i = 0; i + simd_width <= vector_len; i += simd_width) {
            __m128i vec1_reg = _mm_loadu_si128((__m128i*)(vec1 + i));
            __m128i vec2_reg = _mm_loadu_si128((__m128i*)(vec2 + i));
            __m128i mask_reg = _mm_loadu_si128((__m128i*)(mask + i));

            __m128i xor_result = _mm_xor_si128(vec1_reg, vec2_reg);
            __m128i mask_result = _mm_and_si128(xor_result, mask_reg);
            for (int j = 0; j < simd_width; ++j) {
                ans += _mm_popcnt_u32(static_cast<unsigned int>(_mm_extract_epi32(mask_result, j)));
            }
        }

        for (; i < vector_len; i++)
            ans += __builtin_popcount((vec1[i] ^ vec2[i]) & mask[i]);
        return ans;
    }
    std::vector<uint32_t> get_mask_int8(const void *data1, const void *data2) {
        const uint8_t *point_data1 = reinterpret_cast<const uint8_t*>(data1);
        const uint8_t *point_data2 = reinterpret_cast<const uint8_t*>(data2);

        std::vector<uint32_t> mask(vector_len);

        int threshold = 32;
        for (int j = 0; j < dim; j++) {
            if (std::abs(point_data1[j] - point_data2[j]) > threshold)
                mask[j/32] |= (1<<j%32);
        }
        return mask;
    }
    
private:
    uint32_t *data;
};

int Dir_Vector::dim = 0;
int Dir_Vector::vector_len = 0;

void Dir_Vector::calc_dir_vector_int8(const void *point, const void *neighbor, int id) {
    const uint8_t *point_data = reinterpret_cast<const uint8_t*>(point);
    const uint8_t *neighbor_data = reinterpret_cast<const uint8_t*>(neighbor);
    uint32_t *now_data = (data + id * vector_len);

    _mm_prefetch((const char *) (point), _MM_HINT_T0);
    _mm_prefetch((const char *) (neighbor), _MM_HINT_T0);

    __m256i offset = _mm256_set1_epi8(0x80); // 创建偏移量向量

    int blk = 0;
    for (int i = 0; i < dim; i += 32) {
        __m256i point_vec = _mm256_loadu_si256((__m256i*)(point_data + i));
        __m256i neighbor_vec = _mm256_loadu_si256((__m256i*)(neighbor_data + i));

        point_vec = _mm256_add_epi8(point_vec, offset);
        neighbor_vec = _mm256_add_epi8(neighbor_vec, offset);

        __m256i cmp_result = _mm256_cmpgt_epi8(neighbor_vec, point_vec); // 注意是greater than

        uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(cmp_result)); // 将比较结果移动到一个32位掩码中

        // if (now_data[blk] != mask) {
        //     std::cout << "mask " << mask << '\n';
        //     std::cout << "error\n";
        //     exit(-1);
        // }
            
        // 由于掩码是32位，我们可以直接将其存储到now_data中
        now_data[blk++] = mask;
    }

    return ;
}


void Dir_Vector::calc_dir_vector_float(const void *point, const void *neighbor, int id) {
    const float *point_data = reinterpret_cast<const float*>(point);
    const float *neighbor_data = reinterpret_cast<const float*>(neighbor);

    uint32_t *now_data = (data + id * vector_len);
    bool relative[dim];
    #pragma omp parallel for
    for (int i = 0; i < dim; i++) {
        relative[i] = (*(point_data+i) < *(neighbor_data+i));
    }
    #pragma omp parallel for
    for (int i = 0; i < 32; i++) {
        for (int j = i; j < dim; j+= 32) {
            #pragma omp atomic
            now_data[j/32] |= (relative[j] << i);
        }
    }
}

}
