#pragma once

#include "k_means.h"
#include <random>

template<typename dist_t, typename sum_type_t>
KMeans<dist_t, sum_type_t>::KMeans(int _k, int _N, int _dim, hnswlib::SpaceInterface<dist_t> *_space, 
    DataLoader *_data_loader, vector<int> &ids)
    : k(_k), N(_N), dim(_dim), space(_space), data_loader(_data_loader) {

    // Constructor implementation

    centroids.resize(k);
    for (int i = 0; i < k; i++)
        centroids[i].resize(dim);
    assignments.resize(N, -1); // 初始化点分配为-1
    cluster_nums.resize(k, 0);
    diameters.resize(k, 0);
    globalIDS = ids;
    clusters.resize(k);

    center_dis.resize(k, vector<float>(k));
    dis2centroid.resize(N, 1e9);
}

std::random_device rd; // 用于获取随机数种子
std::mt19937 gen(rd()); // 标准 mersenne_twister_engine
int rand_int(int l, int r) {
    std::uniform_int_distribution<> dis(l, r); // 定义在[min, max]范围内的均匀分布
    return dis(gen);
}

// 使用的均为[0, N)的local id
template<typename dist_t, typename sum_type_t>
void KMeans<dist_t, sum_type_t>::initializeCenters() {
    std::mt19937 gen(random_device{}()); // 使用设备随机数作为种子
    std::uniform_real_distribution<float> dist(0.0, 1.0); // 定义分布范围

    vector<int> centers;
    centers.resize(k, -1); // 初始化中心索引为-1

    // first random center
    centers[0] = rand_int(0, N-1);
    vector<double> weight(N);

    for (int i = 0; i < N; i++)
        weight[i] = MAXFLOAT;


    for (int j = 1; j < k; j++) {
        double sum = 0;
        double choice = 0;

        int lst_center = centers[j-1];

        for (int i = 0; i < N; i++) {
            weight[i] = min(weight[i], (double)this->dis(i, lst_center));
            sum += weight[i];
        }
        
        choice = dist(gen) * sum;
        for (int i = 0; i < N; i++) {
            choice -= weight[i];
            if (weight[i] > 0 && choice <= 0) {
                centers[j] = i;
                break;
            }
        }
    }

    // sort and check
    sort(centers.begin(), centers.end());
    for (int i = 0; i+1 < centers.size(); i++) {
        if (centers[i] == centers[i+1]) {
            cerr << "error: duplicate center\n";
            assert(centers[i] != centers[i+1]);
        }
    }
    
    for (int i = 0; i < k; i++) {
        memcpy(centroids[i].data(), data_loader->point_data(centers[i]), sizeof(dist_t) * dim);
    }
    
}

template<typename dist_t, typename sum_type_t>
void KMeans<dist_t, sum_type_t>::assignPointsToClosestCenter() {
    for (int j = 0; j < k; j++) {
        cluster_nums[j] = 0;
    }
    for (int i = 0; i < N; ++i) {
        float min_dist = 1e9;
        int closest_center = assignments[i];
        if (closest_center != -1) min_dist = dis(i, centroids[closest_center].data());
        for (int j = 0; j < k; ++j) {
            if (closest_center != -1 && min_dist * 2 < center_dis[closest_center][j]) continue;
            dist_t dist = dis(i, centroids[j].data());
            if (dist < min_dist) {
                min_dist = dist;
                closest_center = j;
            }
        }
        assignments[i] = closest_center;
        dis2centroid[i] = min_dist;
        cluster_nums[closest_center] ++ ;
    }
}

template<typename dist_t, typename sum_type_t>
void KMeans<dist_t, sum_type_t>::updateCenters() {
    vector<vector<sum_type_t> > tem(k);
    for (int i = 0; i < k; i++)
        tem[i].resize(dim);
    for (int i = 0; i < N; i++) {
        const dist_t *point = reinterpret_cast<const dist_t*>(data_loader->point_data(i));
        int centroid_id = assignments[i];
        for (int j = 0; j < dim; j++)
            tem[centroid_id][j] += point[j];
    }
    for (int i = 0; i < k; i++)
        for (int j = 0; j < dim; j++)
            centroids[i][j] = tem[i][j] / cluster_nums[i];
    
    return ;
    // // 对每个簇，找到新的中心
    // for (int j = 0; j < k; ++j) {
    //     int lst_center = centers[j];
    //     int num = cluster_nums[j];
    //     sum_type_t min_dist_sum = std::numeric_limits<dist_t>::max();
    //     int best_center = -1;
        
    //     vector<sum_type_t> sum_l1(dim, 0), sum_l2(dim, 0);

    //     for (int i = 0; i < N; ++i) { // 尝试将每个点作为新中心
    //         if (assignments[i] != j) continue; // 只考虑当前簇中的点

    //         if (std::is_same<dist_t, float>::value) {
    //             const float *point = reinterpret_cast<const float*>(data_loader->point_data(globalIDS[i]));
    //             for (int d = 0; d < dim; d++) {
    //                 sum_l1[d] += point[d];
    //                 sum_l2[d] += point[d] * point[d];
    //             }
    //         } else {
    //             const uint8_t *point = reinterpret_cast<const uint8_t*>(data_loader->point_data(globalIDS[i]));
    //             for (int d = 0; d < dim; d++) {
    //                 sum_l1[d] += point[d];
    //                 sum_l2[d] += (sum_type_t)point[d] * point[d];
    //             }

    //         }
    //     }

    //     for (int i = 0; i < N; ++i) { // 尝试将每个点作为新中心
    //         if (assignments[i] != j) continue; // 只考虑当前簇中的点
    //         sum_type_t sum_dis = 0;

    //         if (std::is_same<dist_t, float>::value) {
    //             const float *point = reinterpret_cast<const float*>(data_loader->point_data(globalIDS[i]));
    //             for (int d = 0; d < dim; d++) {
    //                 sum_dis += (sum_type_t)point[d] * point[d] * num;
    //                 sum_dis -= 2 * sum_l1[d] * point[d];
    //                 sum_dis += sum_l2[d];
    //             }
    //         } else {
    //             const uint8_t *point = reinterpret_cast<const uint8_t*>(data_loader->point_data(globalIDS[i]));
    //             for (int d = 0; d < dim; d++) {
    //                 sum_dis += (sum_type_t)point[d] * point[d] * num;
    //                 sum_dis -= 2 * (sum_type_t)sum_l1[d] * point[d];
    //                 sum_dis += sum_l2[d];
    //             }

    //         }
    //         if (sum_dis < min_dist_sum) {
    //             min_dist_sum = sum_dis;
    //             best_center = i;
    //         }
    //     }

    //     if (best_center != -1) { // 更新簇中心
    //         centers[j] = best_center;
    //     }
    //     int64_t tem_sum = 0;
    //     for (int i = 0; i < N; i++) {
    //         if (assignments[i] != j) continue;
    //         // tem_sum += dis(centers[j], i);
    //         tem_sum += dis(lst_center, i);
    //     }
    // }
}

template<typename dist_t, typename sum_type_t>
void KMeans<dist_t, sum_type_t>::updateCenterDis() {
    for (int i = 0; i < k; i ++)
        for (int j = i + 1; j < k; j++)
            center_dis[i][j] = center_dis[j][i] = dis(centroids[i].data(), centroids[j].data());
    for (int i = 0; i < k; i++)
        center_dis[i][i] = 0;  
}

template<typename dist_t, typename sum_type_t>
void KMeans<dist_t, sum_type_t>::run(int maxIterations) {

    initializeCenters();
    updateCenterDis();

    for (int iter = 0; iter < maxIterations; ++iter) {
        // cout << "iter = " << iter << '\n';
        assignPointsToClosestCenter();

        auto lst_centers = centroids;

        updateCenters();
        updateCenterDis();

        bool changed = 0;
        for (int j = 0; j < k; j++)
            for (int d = 0; d < dim; d++) {
                if (fabs(centroids[j][d] - lst_centers[j][d]) > 1e-3) {
                    changed = 1;
                    break;
                }
            }
        if (!changed) break;

    }
    for (int i = 0; i < N; i++) {
        clusters[assignments[i]].push_back(i);
    }
    calc_diameter();
}

template<typename dist_t, typename sum_type_t>
sum_type_t KMeans<dist_t, sum_type_t>::dis(int i, int j) {
    // dis 实现...
    i = globalIDS[i];
    j = globalIDS[j];
    return space->get_dist_func()(data_loader->point_data(i), data_loader->point_data(j), space->get_dist_func_param());
}

template<typename dist_t, typename sum_type_t>
sum_type_t KMeans<dist_t, sum_type_t>::dis(int i, const void* data_point) {
    // dis 实现...
    i = globalIDS[i];
    return space->get_dist_func()(data_loader->point_data(i), data_point, space->get_dist_func_param());
}

template<typename dist_t, typename sum_type_t>
sum_type_t KMeans<dist_t, sum_type_t>::dis(const void* data_point, const void* data_point2) {
    // dis 实现...
    return space->get_dist_func()(data_point, data_point2, space->get_dist_func_param());
}

template<typename dist_t, typename sum_type_t>
sum_type_t KMeans<dist_t, sum_type_t>::tot_dist() {
    // tot_dist 实现...
    sum_type_t sum = 0;
    return sum;
}

template<typename dist_t, typename sum_type_t>
sum_type_t KMeans<dist_t, sum_type_t>::tot_dist_recalc() {
    // tot_dist_recalc 实现...
    sum_type_t ans = 0;
    for (int i = 0; i < N; i++)
        // ans += dis(i, centers[assignments[i]]);
        ans += dis(i, centroids[assignments[i]].data());
    return ans;
}

template<typename dist_t, typename sum_type_t>
void KMeans<dist_t, sum_type_t>::calc_diameter() {
    for (int i = 0; i < k; i++) {
        diameters[i] = 0;
        for (auto p: clusters[i]) {
            // float d = dis(centers[i], p);
            float d = dis(p, centroids[i].data());
            diameters[i] = max(diameters[i], d);
        }
    }
}
template<typename dist_t, typename sum_type_t>
void KMeans<dist_t, sum_type_t>::output() {
    // output 实现...
    cout << "centers:\n";
    for (int i = 0; i < k; i++) {
        assert(clusters[i].size() == cluster_nums[i]);
        cout << i << " nums = " << cluster_nums[i] << " diameter = " << diameters[i] << '\n';
    }
}

template<typename dist_t, typename sum_type_t>
vector<vector<dist_t> > KMeans<dist_t, sum_type_t>::get_centers_global() {
    return centroids;
}

int searchCalc = 0;

template<typename dist_t, typename sum_type_t>
priority_queue<pair<float, int> > KMeans<dist_t, sum_type_t>::find_nearest_centers_id(const void* data_point, int nprobe) {
    priority_queue<pair<float, int> > near_centers;
    for (int i = 0; i < k; i++) {
        float d = dis(centroids[i].data(), data_point);
        near_centers.push(make_pair(d, i));
        if (near_centers.size() > nprobe) near_centers.pop();
    }
    return near_centers;
}

template<typename dist_t, typename sum_type_t>
vector<vector<dist_t> > KMeans<dist_t, sum_type_t>::find_nearest_centers(const void* data_point, int nprobe) {
    auto nn_ids = find_nearest_centers_id(data_point, nprobe);
    vector<vector<dist_t> > ans;
    while (nn_ids.size()) {
        ans.push_back(centroids[nn_ids.top().second]);
        nn_ids.pop();
    }
    reverse(ans.begin(), ans.end());
    return ans;
}
// ivf-flat
template<typename dist_t, typename sum_type_t>
vector<uint32_t> KMeans<dist_t, sum_type_t>::searchKnn(const void* data_point, int knn, int nprobe) {

    // centers id: [0, k-1]
    auto near_centers = find_nearest_centers_id(data_point, nprobe);

    priority_queue<pair<float, int> > nns;
    while(near_centers.size()) {
        int center_id = near_centers.top().second;
        // // statis search cluster nums
        // cout << clusters[center_id].size() << "\n";
        searchCalc += clusters[center_id].size();
        for (auto p: clusters[center_id]) {
            float d = dis(p, data_point);
            nns.push(make_pair(d, globalIDS[p]));
            if (nns.size() > knn) nns.pop();
        }
        near_centers.pop();
    }

    vector<uint32_t> result;
    while(nns.size()) {
        result.push_back(nns.top().second);
        nns.pop();
    }
    reverse(result.begin(), result.end());
    return result;
}
template<typename dist_t, typename sum_type_t>
dist_t * KMeans<dist_t, sum_type_t>::get_assign(int id) {
    return centroids[assignments[id]].data();
}

int prune_search_calc = 0;
int prune_search_probes = 0;
template<typename dist_t, typename sum_type_t>
vector<uint32_t> KMeans<dist_t, sum_type_t>::searchKnn_prune(const void* data_point, int knn, int nprobe, float epsilon) {
    auto near_centers = find_nearest_centers_id(data_point, nprobe);
    vector<pair<float, int> > centers;
    while(near_centers.size()) {
        centers.push_back(near_centers.top());
        near_centers.pop();
    }
    reverse(centers.begin(), centers.end());

    priority_queue<pair<float, int> > nns;
    float min_dis = 1e9;  
    for (auto pr: centers) {
        if (pr.first < min_dis) min_dis = pr.first;
        else if (pr.first > min_dis * (1+epsilon)) break;
        int center_id = pr.second;
        prune_search_calc += clusters[center_id].size();
        prune_search_probes ++;
        for (auto p: clusters[center_id]) {
            float d = dis(p, data_point);
            nns.push(make_pair(d, globalIDS[p]));
            if (nns.size() > knn) nns.pop();
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
