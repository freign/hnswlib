#pragma once

#include <bits/stdc++.h>

namespace dir_vector {

class Dir_Vector {
public:
    static int dim;
    static int vector_len; // vector_len = ceil(dim/32)
    static void init(int _dim) {
        dim = _dim;
        vector_len = (dim + 31) / 32;
    }
    Dir_Vector(size_t size) {
        data = new uint32_t [size * vector_len];
        memset(data, 0, size * vector_len);
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

private:
    uint32_t *data;
    int len;
};

int Dir_Vector::dim = 0;
int Dir_Vector::vector_len = 0;

void Dir_Vector::calc_dir_vector_int8(const void *point, const void *neighbor, int id) {
    const uint8_t *point_data = reinterpret_cast<const uint8_t*>(point);
    const uint8_t *neighbor_data = reinterpret_cast<const uint8_t*>(neighbor);
    uint32_t *now_data = (data + id * vector_len);

    // _mm_prefetch((const char *) (point), _MM_HINT_T0);
    // _mm_prefetch((const char *) (neighbor), _MM_HINT_T0);

    bool relative[dim];
    for (int i = 0; i < dim; i++) {
        relative[i] = (*(point_data+i) < *(neighbor_data+i));
    }
    for (int i = 0; i < 32; i++) {
        for (int j = i; j < dim; j+= 32) {
            now_data[j/32] |= (relative[j] << i);
        }
    }
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
