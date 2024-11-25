#include <bits/stdc++.h>
#include "data_loader.h"

#include <faiss/IndexPQ.h>
#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include "../hnswlib/hnswlib.h"
#include "calc_group_truth.h"
#include "recall_test.h"
#include "ArgParser.h"
#include "config.h"
#include "statis_tasks.h"
#include "dir_vector.h"
#include "k_means.h"
#include "ivf_hnsw.h"
#include "pq_dist.h"
#include "timer.h"
#include <random>

using namespace std;
using DATALOADER::DataLoader;
using namespace faiss;

float euclidean_distance_simd(const int16_t* vec1, const int16_t* vec2, int dim) {
    // 累加距离平方和的向量
    __m256i sum_vec = _mm256_setzero_si256();

    // 每次处理16个int16元素
    for (int i = 0; i < dim; i += 16) {
        // 加载vec1和vec2的16个元素
        __m256i vec1_chunk = _mm256_loadu_si256((__m256i*)&vec1[i]);
        __m256i vec2_chunk = _mm256_loadu_si256((__m256i*)&vec2[i]);

        // 计算两个向量块的差
        __m256i diff = _mm256_sub_epi16(vec1_chunk, vec2_chunk);

        // 将差平方
        __m256i diff_squared = _mm256_mullo_epi16(diff, diff);

        // 累加差平方和
        sum_vec = _mm256_add_epi32(sum_vec, _mm256_madd_epi16(diff_squared, _mm256_set1_epi16(1)));
    }

    // 将累加结果的各部分相加
    __m128i sum_low = _mm256_castsi256_si128(sum_vec);
    __m128i sum_high = _mm256_extracti128_si256(sum_vec, 1);
    __m128i sum = _mm_add_epi32(sum_low, sum_high);

    // 将128位结果中的各部分相加
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));

    // 提取最终结果
    int32_t distance_squared = _mm_cvtsi128_si32(sum);

    // 返回欧式距离
    return sqrtf((float)distance_squared);
}


int main(int argc, char *argv[])
{
    //CommandLineOptions opt = ArgParser(argc, argv);
    //int max_elements = opt.maxElements;
    int m = atoi(argv[1]);
    int nbits = atoi(argv[2]);
    int M = 16;
    int ef_construction = 200;
    string gist_dir = "/share/ann_benchmarks/gist/";

    DataLoader *data_loader = new DataLoader("f", 1000000, gist_dir + "train.fvecs", "gist");
    // DataLoader *data_loader = new DataLoader("f", 1000000, "/root/datasets/gist/train.fvecs", "gist");
    DataLoader *query_data_loader = new DataLoader("f", 1000, gist_dir + "test.fvecs", "gist");
    
    
    //GroundTruth::GT_Loader *gt_loader;

    //Config *config = new Config();

    // Set random seed for reproducibility
    srand(static_cast<unsigned int>(time(0)));

    // Dimension of the vectors
    int d = 960;

    // Number of vectors to index
    //int N = opt.maxElements;

    // Number of subquantizers
    //int m = 60;

    // Number of bits per subquantizer
    //int nbits = 4;

    //data_loader = new DataLoader("f", opt.maxElements, opt.point_data_path, "gist");
    //query_data_loader = new DataLoader("f", 0, opt.query_data_path, "gist");
    //gt_loader = new GroundTruth::GT_Loader(opt.dataDir, data_loader, query_data_loader);
    //hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(data_loader->get_dim());

    cout << "data prepared" << endl;
    hnswlib::L2Space space(960);
    //vector<float> xb(N * d * sizeof(float));
    //for (int i = 0; i < N; i++)
    //{
        //memcpy(xb.data() + (i * d * sizeof(float)), data_loader->point_data(i), d * sizeof(float));
    //}
    //StopW trainTimer;
    PQDist *pq_dist = new PQDist(d, m, nbits);

    string path = gist_dir + "encoded_data_" + to_string(m) + "_" + to_string(nbits);
    pq_dist->load(path);
    /* for (int i = 99999; i < 100000; i++)
    {
        for (int j = 0; j < 120; j++)
            cout << (int)pq_dist->codes[i * pq_dist->m + j] << " ";
        cout << "\n";
    } */
    // pq_dist->train(N, xb);
    // cout << "train time " << trainTimer.getElapsedTimeMicro() << endl;

    //hnswlib::SpaceInterface<float> *space2 = new hnswlib::L2Space(data_loader->get_dim() / 2);

    ifstream file(gist_dir + "point_search.txt");
    string line;
    vector<vector<int>> points_search(1000);
    int t = 0;
    while (getline(file, line))
    {
        istringstream ls(line);
        int id;
        while (ls >> id)
        {
            points_search[t].push_back(id);
        }
        t++;
    }
    //prefetch
    /* for (int j = 0; j < query_data_loader->get_elements(); j += 1)
    {
        pq_dist->load_query_data(reinterpret_cast<const float *>(query_data_loader->point_data(j)), 1);
        for (int i : points_search[j])
        {
            // float distPQ = pq_dist->calc_dist_pq_loaded(i);
            // cout << distPQ << " " << real_dist << "\n";
            float real_dist = space.get_dist_func()(
                query_data_loader->point_data(j), data_loader->point_data(i), space.get_dist_func_param());
        }
    }  */
    StopW LoadTimer;
    LoadTimer.reset();
    for (int j = 0; j < query_data_loader->get_elements(); j += 1)
    {
        pq_dist->load_query_data_and_cache(reinterpret_cast<const float *>(query_data_loader->point_data(j)));
    } 
    cout << "load time " << LoadTimer.getElapsedTimeMicro() / 1e3 << endl;

    
    pq_dist->load_query_data_and_cache(reinterpret_cast<const float *>(query_data_loader->point_data(0)));
    StopW PQTimer;
    PQTimer.reset();
    vector<float> pqs;
    for (int j = 0; j < query_data_loader->get_elements(); j += 1)
    {
        pq_dist->load_query_data_and_cache(reinterpret_cast<const float *>(query_data_loader->point_data(j)));

        for (int i : points_search[j])
        {
            i = 0;
            auto ids = pq_dist->get_centroids_id(i);
            float distPQ = pq_dist->calc_dist_pq_loaded_simd(0, ids.data());
            // float distPQ = pq_dist->calc_dist_pq_loaded(0, ids.data());
            pqs.push_back(distPQ);
        }
    }
    std::cout << "here" << std::endl;
    cout << "PQ time " << PQTimer.getElapsedTimeMicro() / 1e3 << endl;

    StopW RealTimer;
    RealTimer.reset();
    vector<float> reals;
    vector<float> vec_tem(960);
    memcpy(vec_tem.data(), data_loader->point_data(0), sizeof(float) * 960);
    int tot = 0;
    for (int j = 0; j < query_data_loader->get_elements(); j += 1)
    {
        int idx = 0;
        // tot +=    [j].size();
        for (int i : points_search[j])
        {
            float real_dist = space.get_dist_func()(
                query_data_loader->point_data(j), data_loader->point_data(i), space.get_dist_func_param());
            reals.push_back(real_dist);
            // cout << distPQ << " " << real_dist << "\n";
        }
    }
    cout << "tot = " << tot << "\n";
    cout << "real time " << RealTimer.getElapsedTimeMicro() / 1e3 << endl;


    StopW UINTTimer;
    UINTTimer.reset();
    
    hnswlib::L2SpaceI spacei(960);
    vector<int16_t> vec1(960), vec2(960);
    vector<float> fvec1(960), fvec2(960);
    for (int i = 0; i < 960; i++) {
        vec1[i] = rand();
        vec2[i] = rand();
        fvec1[i] = rand();
        fvec2[i] = rand();
    }
    vector<float> int16_results;
    float sum = 0;
    for (int j = 0; j < query_data_loader->get_elements(); j += 1)
    {
        for (int i : points_search[j])
        {
            i = 0;
            // float dist = L2SqrI16(vec1.data(), vec2.data(), 960);
            // float dist = L2Sqr(query_data_loader->point_data(j), data_loader->point_data(i), 960);
            float dist = euclidean_distance_simd(vec1.data(), vec2.data(), 960);

            // float dist = space.get_dist_func()(
            //     query_data_loader->point_data(j), data_loader->point_data(i), space.get_dist_func_param());
            sum += dist;
            int16_results.push_back(dist);
            // cout << distPQ << " " << real_dist << "\n";
        }
    }
    std::cout << sum << "\n";
    cout << "int16 time " << UINTTimer.getElapsedTimeMicro() / 1e3 << endl;


    assert(pqs.size() == reals.size());

    float error = 0;

    // cout << "\n\n";

    for (int i = 0; i < pqs.size(); i++)
    {
        error += (sqrt(pqs[i]) - sqrt(reals[i])) * (sqrt(pqs[i]) - sqrt(reals[i]));
        // cout << pqs[i] << ' ' << reals[i] << "\n";
    }
    error /= pqs.size();
    cout << "error = " << error << "\n";



    return 0;
}
