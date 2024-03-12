#include "data_loader.h"
#include "ArgParser.h"
#include "../hnswlib/hnswlib.h"

using namespace std;
using DATALOADER::DataLoader;


template<typename dist_t, typename sum_type_t>
class KMeans {
public:
    KMeans(int _k, int _N, int _dim, hnswlib::SpaceInterface<dist_t> *_space, 
        DataLoader *_data_loader, vector<int> &ids)
        : k(_k), N(_N), dim(_dim), space(_space), data_loader(_data_loader) {

        centers.resize(k, -1); // 初始化中心索引为-1
        sum_dis.resize(k, 0);
        assignments.resize(N, -1); // 初始化点分配为-1
        cluster_nums.resize(k, 0);
        globalIDS = ids;
    }
    ~KMeans() = default;
    void initializeCenters() {
        // srand(time(0));
        // 简单随机选择k个初始中心
        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0); // 填充0到N-1
        std::random_shuffle(indices.begin(), indices.end()); // 随机打乱
        for (int i = 0; i < k; ++i) {
            centers[i] = indices[i];
        }
    }

    void assignPointsToClosestCenter() {
        for (int j = 0; j < k; j++) {
            cluster_nums[j] = 0;
            sum_dis[j] = 0;
        }

        for (int i = 0; i < N; ++i) {
            dist_t min_dist = std::numeric_limits<dist_t>::max();
            int closest_center = -1;
            for (int j = 0; j < k; ++j) {
                dist_t dist = dis(i, centers[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_center = j;
                }
            }
            assignments[i] = closest_center;
            sum_dis[closest_center] += min_dist;
            cluster_nums[closest_center] ++ ;
        }
    }

    void updateCenters() {
        // 对每个簇，找到新的中心
        for (int j = 0; j < k; ++j) {
            int lst_center = centers[j];
            int num = cluster_nums[j];
            sum_type_t min_dist_sum = std::numeric_limits<dist_t>::max();
            int best_center = -1;
            
            vector<sum_type_t> sum_l1(dim, 0), sum_l2(dim, 0);

            for (int i = 0; i < N; ++i) { // 尝试将每个点作为新中心
                if (assignments[i] != j) continue; // 只考虑当前簇中的点

                if (std::is_same<dist_t, float>::value) {
                    const float *point = reinterpret_cast<const float*>(data_loader->point_data(globalIDS[i]));
                    for (int d = 0; d < dim; d++) {
                        sum_l1[d] += point[d];
                        sum_l2[d] += point[d] * point[d];
                    }
                } else {
                    const uint8_t *point = reinterpret_cast<const uint8_t*>(data_loader->point_data(globalIDS[i]));
                    for (int d = 0; d < dim; d++) {
                        sum_l1[d] += point[d];
                        sum_l2[d] += (sum_type_t)point[d] * point[d];
                    }

                }
            }

            for (int i = 0; i < N; ++i) { // 尝试将每个点作为新中心
                if (assignments[i] != j) continue; // 只考虑当前簇中的点
                sum_type_t sum_dis = 0;

                if (std::is_same<dist_t, float>::value) {
                    const float *point = reinterpret_cast<const float*>(data_loader->point_data(globalIDS[i]));
                    for (int d = 0; d < dim; d++) {
                        sum_dis += (sum_type_t)point[d] * point[d] * num;
                        sum_dis -= 2 * sum_l1[d] * point[d];
                        sum_dis += sum_l2[d];
                    }
                } else {
                    const uint8_t *point = reinterpret_cast<const uint8_t*>(data_loader->point_data(globalIDS[i]));
                    for (int d = 0; d < dim; d++) {
                        sum_dis += (sum_type_t)point[d] * point[d] * num;
                        sum_dis -= 2 * (sum_type_t)sum_l1[d] * point[d];
                        sum_dis += sum_l2[d];
                    }

                }
                if (sum_dis < min_dist_sum) {
                    min_dist_sum = sum_dis;
                    best_center = i;
                }
            }

            if (best_center != -1) { // 更新簇中心
                centers[j] = best_center;
            }
            int64_t tem_sum = 0;
            for (int i = 0; i < N; i++) {
                if (assignments[i] != j) continue;
                // tem_sum += dis(centers[j], i);
                tem_sum += dis(lst_center, i);
            }
        }
    }

    void run(int maxIterations = 100) {
        initializeCenters();
        for (int iter = 0; iter < maxIterations; ++iter) {
            assignPointsToClosestCenter();

            cout << "iter: " << iter << '\n';
            output();
            cout << "tot cost " << tot_dist() << ' ' << tot_dist_recalc() << '\n';
            auto lst_centers = centers;
            updateCenters();
            bool changed = 0;
            for (int j = 0; j < k; j++)
                if (centers[j] != lst_centers[j])
                    changed = 1;
            if (!changed) break;
        }
    }

    dist_t dis(int i, int j) {
        i = globalIDS[i];
        j = globalIDS[j];
        return space->get_dist_func()(data_loader->point_data(globalIDS[i]), data_loader->point_data(j), space->get_dist_func_param());
    }

    sum_type_t tot_dist() {
        sum_type_t sum = 0;
        for (int j = 0; j < k; j ++)
            sum += sum_dis[j];
        return sum;
    }
    sum_type_t tot_dist_recalc() {
        sum_type_t ans = 0;
        for (int i = 0; i < N; i++)
            ans += dis(i, centers[assignments[i]]);
        return ans;
    }
    void output() {
        cout << "centers:\n";
        for (int i = 0; i < k ; i++)
            cout << centers[i] << " nums = " << cluster_nums[i] << " sum dist = " << sum_dis[i] << '\n';

    }
private:
    int k; // 聚类数量
    int N; // 0 ~ N-1
    int dim;
    hnswlib::SpaceInterface<dist_t> *space;
    DataLoader *data_loader;
    std::vector<int> centers; // 中心点的索引
    std::vector<sum_type_t> sum_dis;
    std::vector<int> assignments; // 点到中心的分配
    vector<int> cluster_nums;
    vector<int> globalIDS;
};
