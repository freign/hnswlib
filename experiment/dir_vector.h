#pragma once

#include <vector>
#include <iostream>
#include <emmintrin.h>
#include <immintrin.h>

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
