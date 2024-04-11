// k_means.h
#ifndef K_MEANS_H
#define K_MEANS_H

#include "data_loader.h"
#include "ArgParser.h"
#include "../hnswlib/hnswlib.h"
#include <vector>
#include <limits>

using namespace std;
using DATALOADER::DataLoader;

template<typename dist_t, typename sum_type_t>
class KMeans {
public:
    KMeans(int _k, int _N, int _dim, hnswlib::SpaceInterface<dist_t> *_space, DataLoader *_data_loader, vector<int> &ids);
    ~KMeans() = default;

    void initializeCenters();
    void assignPointsToClosestCenter();
    void updateCenters();
    void run(int maxIterations = 100);
    sum_type_t dis(int i, int j);
    sum_type_t dis(int i, const void* data_point);
    sum_type_t dis(const void* data_point, const void* data_point2);
    sum_type_t tot_dist();
    sum_type_t tot_dist_recalc();
    void output();
    vector<vector<dist_t> > get_centers_global();
    void calc_diameter();
    vector<uint32_t> searchKnn(const void* data_point, int knn, int nprobe);
    vector<uint32_t> searchKnn_prune(const void* data_point, int knn, int nprobe, float epsilon);
    priority_queue<pair<float, int> > find_nearest_centers(const void* data_point, int nprobe);
    
private:
    int k, N, dim;
    hnswlib::SpaceInterface<dist_t> *space;
    DataLoader *data_loader;
    vector<int> cluster_nums, globalIDS;
    // vector<int> centers;
    vector<vector<dist_t> > centroids;
    // assignments: [0, k)
    vector<int> assignments;
    vector<vector<int> > clusters;
    vector<sum_type_t> sum_dis;
    vector<float> diameters;
};

#include "k_means_impl.tpp"
#endif // K_MEANS_H
