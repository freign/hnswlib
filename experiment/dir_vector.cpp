#include "dir_vector.h"
#include <cstring> // For memset

namespace dir_vector {

int Dir_Vector::dim = 0;
int Dir_Vector::vector_len = 0;

void Dir_Vector::init(int _dim) {
    dim = _dim;
    vector_len = (dim + 31) / 32;
}

Dir_Vector::Dir_Vector(size_t size) {
    len = size;
    data = new uint32_t[size * vector_len];
    memset(data, 0, size * vector_len * sizeof(uint32_t));
}

Dir_Vector::~Dir_Vector() {
    if (data != nullptr) {
        delete[] data;
    }
}

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

        // 由于掩码是32位，我们可以直接将其存储到now_data中
        now_data[blk++] = mask;
    }

    return ;
}

void Dir_Vector::calc_dir_vector_float(const void *point, const void *neighbor, int id) {
    const float *point_data = reinterpret_cast<const float*>(point);
    const float *neighbor_data = reinterpret_cast<const float*>(neighbor);

    uint32_t *now_data = (data + id * vector_len);

    _mm_prefetch((const char *) (point), _MM_HINT_T0);
    _mm_prefetch((const char *) (neighbor), _MM_HINT_T0);

    __m256i offset = _mm256_set1_epi8(0x80); // 创建偏移量向量

    int blk = 0;
    for (int i = 0; i < dim; i += 8) { // 处理8个floats为一组的块
        // 加载两个向量的当前块
        __m256 point_vec = _mm256_loadu_ps(point_data + i);
        __m256 neighbor_vec = _mm256_loadu_ps(neighbor_data + i);

        // 比较point_vec和neighbor_vec
        __m256 cmp_result = _mm256_cmp_ps(point_vec, neighbor_vec, _CMP_GE_OQ);

        // 将比较结果从浮点数转换为整数位掩码
        uint32_t mask = _mm256_movemask_ps(cmp_result);

        // 存储结果。这里每个位都代表对应元素的比较结果，1为true，0为false
        // 因为现在是按位存储，所以可能需要按照实际情况调整数据存储方式
        now_data[blk] |= mask << (i&31);
        if ((i&24) == 24) blk++;
    }
}

void Dir_Vector::print_dir_vector(int id) {
    for (int i = 0; i < vector_len; i++) {
        for (int j = 0; j < 32; j++) {
            std::cout << ((data[i + id * vector_len] >> j) & 1);
        }
    }
    std::cout << '\n';
}

uint32_t* Dir_Vector::dir_vector_data(int id) {
    return (data + id * vector_len);
}

uint32_t Dir_Vector::calc_dis(uint32_t *vec1, uint32_t *vec2) {
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

uint32_t Dir_Vector::calc_dis_with_mask(uint32_t *vec1, uint32_t *vec2, uint32_t *mask) {
    // if ((dim & 127)) {
    //     std::cerr << "dim can't divide 128";
    //     exit(0);
    // }
    int ans = 0;
    const int simd_width = 4;
    int i = 0;
    for (i = 0; i + simd_width <= vector_len; i += simd_width) {
        __m128i vec1_reg = _mm_loadu_si128((__m128i*)(vec1 + i));
        __m128i vec2_reg = _mm_loadu_si128((__m128i*)(vec2 + i));
        __m128i mask_reg = _mm_loadu_si128((__m128i*)(mask + i));

        __m128i xor_result = _mm_xor_si128(vec1_reg, vec2_reg);
        __m128i mask_result = _mm_and_si128(xor_result, mask_reg);

        uint64_t* mask_result_ptr = reinterpret_cast<uint64_t*>(&mask_result);

        ans += _mm_popcnt_u64(mask_result_ptr[0]);
        ans += _mm_popcnt_u64(mask_result_ptr[1]);
    }

    for (; i < vector_len; i++)
        ans += __builtin_popcount((vec1[i] ^ vec2[i]) & mask[i]);
    return ans;
}

std::vector<uint32_t> Dir_Vector::get_mask_int8(const void *data1, const void *data2) {
    const uint8_t *point_data1 = reinterpret_cast<const uint8_t*>(data1);
    const uint8_t *point_data2 = reinterpret_cast<const uint8_t*>(data2);

    std::vector<uint32_t> mask(vector_len, 0);


    __m256i threshold_vec = _mm256_set1_epi8(8); // 正阈值向量，已调整偏移

    int blk = 0;
    for (int j = 0; j < dim; j += 32) {
        __m256i vec1 = _mm256_loadu_si256((__m256i*)(point_data1 + j));
        __m256i vec2 = _mm256_loadu_si256((__m256i*)(point_data2 + j));

        // 计算差值并应用偏移，将结果映射到有符号整数范围
        __m256i max_vec = _mm256_max_epu8(vec1, vec2);
        __m256i min_vec = _mm256_min_epu8(vec1, vec2);
        __m256i diff = _mm256_subs_epu8(max_vec, min_vec);

        // 比较差值是否超出阈值范围
        __m256i cmp_result_pos = _mm256_cmpgt_epi8(diff, threshold_vec);

        // 将比较结果转换为掩码
        uint32_t cmp_mask = static_cast<uint32_t>(_mm256_movemask_epi8(cmp_result_pos));

        uint32_t diff_neg_mask = static_cast<uint32_t>(_mm256_movemask_epi8(diff));

        cmp_mask |= diff_neg_mask;

        mask[blk++] |= cmp_mask;
    }
    return mask;
}

std::vector<uint32_t> Dir_Vector::get_mask_float(const void *data1, const void *data2) {
    const float *point_data1 = reinterpret_cast<const float*>(data1);
    const float *point_data2 = reinterpret_cast<const float*>(data2);

    std::vector<uint32_t> mask(vector_len, 0);
    
    float threshold = 0.10;

    __m256 threshold_vec = _mm256_set1_ps(threshold); // 设置浮点数阈值向量
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)); // 用于绝对值的掩码，清除符号位

    int blk = 0;
    for (int j = 0; j < dim; j += 8) { // 处理每8个float为一组
        // 加载两组数据
        __m256 vec1 = _mm256_loadu_ps(point_data1 + j);
        __m256 vec2 = _mm256_loadu_ps(point_data2 + j);

        // 计算差值
        __m256 diff = _mm256_sub_ps(vec1, vec2);

        // 应用掩码计算绝对值
        diff = _mm256_and_ps(diff, sign_mask);

        // 比较差值是否超出阈值
        __m256 cmp_result = _mm256_cmp_ps(diff, threshold_vec, _CMP_GT_OQ);

        // 将比较结果转换为掩码
        uint32_t cmp_mask = static_cast<uint32_t>(_mm256_movemask_ps(cmp_result));

        // 注意这里的索引转换，确保掩码正确存储
        mask[blk] |= cmp_mask << (j&31);
        if ((j&24) == 24) blk++;
    }
    // for (int j = 0; j < dim; j++) {
    //     float diff_abs = fabs(point_data1[j] - point_data2[j]);
    //     std::cout << diff_abs << ' ' << ((mask[j/32] >> (j%32)) &1 ) << ' ' << (((mask[j/32] >> (j%32)) &1 ) ^ (diff_abs >= threshold)) << '\n';
    // }
    return mask;
}

} // namespace dir_vector
