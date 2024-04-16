#pragma once

#include "visited_list_pool.h"
#include "../experiment/config.h"
#include "../experiment/dir_vector.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include "hnswalg.h"

// template<typename dist_t>
// void hnswlib::HierarchicalNSW<dist_t>::degree_adjust() {
//     // Your implementation here
// }

template<typename dist_t>
void hnswlib::HierarchicalNSW<dist_t>::degree_adjust() {
    if (config->use_degree_adjust) {
        
    }
}