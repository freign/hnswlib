#pragma once

#include "k_means.h"
#include "../hnswlib/hnswlib.h"
#include "config.h"

using hnswlib::HierarchicalNSW;
template<typename dist_t, typename sum_type_t>
class IVFHNSW: public KMeans<dist_t, sum_type_t> {

public:
    IVFHNSW(int _k, int _N, int _dim, hnswlib::SpaceInterface<dist_t> *_space, DataLoader *_data_loader, vector<int> &ids)
        :KMeans<dist_t, sum_type_t>(_k, _N, _dim, _space, _data_loader, ids) {

    }
    ~IVFHNSW() {

    }
    vector<uint32_t> searchKnn(const void* data_point, int knn, int nprobe) {

    }
    
    vector<uint32_t> ivfflat_search(const void* data_point, int knn, int nprobe) {
        return KMeans<dist_t, sum_type_t>::searchKnn(data_point, knn, nprobe);
    }

    vector<uint32_t> ivfflat_search_prune(const void* data_point, int knn, int nprobe, float epsilon) {
        return KMeans<dist_t, sum_type_t>::searchKnn_prune(data_point, knn, nprobe, epsilon);
    }

    void create_hnsws(int M, int ef_construction, Config *config) {
        cout << "create graph  " << this->k << "\n";
        hnsws.resize(this->k);
        for (int i = 0; i < hnsws.size(); i++) {
            hnsws[i] = std::move(unique_ptr<HierarchicalNSW<dist_t>>(new HierarchicalNSW<dist_t>(this->space, 
                this->clusters[i].size(), M, ef_construction)));
            hnsws[i]->config = config;
            for (auto p: this->clusters[i]) {
                hnsws[i]->addPoint(this->data_loader->point_data(p), p);
            }
        }
    }
    vector<uint32_t> ivf_hnsw_search(const void *data_point, int knn,int nprobe, float epsilon) {
        auto near_centers = this->find_nearest_centers_id(data_point, nprobe);
        vector<pair<float, int> > centers;
        while(near_centers.size()) {
            centers.push_back(near_centers.top());
            near_centers.pop();
        }
        reverse(centers.begin(), centers.end());

        priority_queue<pair<float, int> > nns;

        for (auto pr: centers) {
            int id = pr.second;
            auto result = hnsws[id]->searchKnn(data_point, knn);
            while (result.size()) {
                nns.push(result.top());
                result.pop();
            }
        }

        vector<uint32_t> result;
        while(nns.size()) {
            result.push_back(nns.top().second);
            nns.pop();
        }
        reverse(result.begin(), result.end());
        return result;
    }
private:
    vector<unique_ptr<HierarchicalNSW<dist_t> > > hnsws;

};
