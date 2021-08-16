/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */



#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <set>
#include <algorithm>
#include <limits>
#include <string>
#include <fstream>
#include <iostream>
#include <stdint.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <sys/time.h>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/utils/distances.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/VectorTransform.h>

// #define MT_ANNS
#define YANDEX_DEEP
// #define YANDEX_TEXT_TO_IMAGE

/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
 *
 * and unzip it to the sudirectory sift1M.
 **/

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

FILE* read_file_head (const char *fname, int32_t *n_out, int32_t *d_out) {
    FILE *f = fopen(fname, "r");
    if(!f) {
        fprintf(stderr, "could not open %s\n", fname);
        return nullptr;
    }

    fread(n_out, sizeof(int), 1, f);
    fread(d_out, sizeof(int), 1, f);

    return f;
}

template<typename T>
int32_t read_file_data (FILE *f, int32_t batch, int32_t dim, T* buff) {
    int32_t batch_read = fread(buff, sizeof(T) * dim, batch, f);
    return batch_read;
}

float * fvecs_read (const char *fname,
                    size_t *d_out, size_t *n_out)
{
    FILE *f = fopen(fname, "r");
    if(!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d; *n_out = n;
    float *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for(size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int *ivecs_read(const char *fname, size_t *d_out, size_t *n_out)
{
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}

template<typename FILE_DISTT, typename FILE_IDT, typename DISTT, typename IDT>
void read_comp(std::vector<std::vector<std::pair<IDT, DISTT>>>& v, const std::string& gt_file, uint32_t& nq, uint32_t& topk) {
    std::ifstream gin(gt_file, std::ios::binary);

    gin.read((char*)&nq, sizeof(uint32_t));
    gin.read((char*)&topk, sizeof(uint32_t));

    v.resize(nq);
    FILE_IDT t_id;
    for (uint32_t i = 0; i < nq; ++i) {
        v[i].resize(topk);
        for (uint32_t j = 0; j < topk; ++j) {
            gin.read((char*)&t_id, sizeof(FILE_IDT));
            v[i][j].first = static_cast<IDT>(t_id);
        }
    }

    FILE_DISTT t_dist;
    for (uint32_t i = 0; i < nq; ++i) {
        for (uint32_t j = 0; j < topk; ++j) {
            gin.read((char*)&t_dist, sizeof(FILE_DISTT));
            v[i][j].second = static_cast<DISTT>(t_dist);
        }
    }
    gin.close();
}

template<typename DISTT, typename IDT>
void recall(const std::string& groundtruth_file,
                float* D,
                long* I,
                size_t answer_nq,
                size_t answer_topk, 
                faiss::MetricType metric_type,
                bool cmp_id = false) {
    uint32_t gt_nq, gt_topk;

    std::vector<std::vector<std::pair<IDT, DISTT>>> groundtruth;
    read_comp<float, uint32_t, DISTT, IDT>(groundtruth, groundtruth_file, gt_nq, gt_topk);

    if (gt_nq != answer_nq || gt_topk < answer_topk) { // || gt_topk != answer_topk) {
        std::cout << "Grountdtruth parammeters does not match. GT nq " << gt_nq
        << "(" << answer_nq << "), topk " << gt_topk << "(" << answer_topk << ")" << std::endl;
        return ;
    }

    std::vector<std::vector<std::pair<IDT, DISTT>>> resultset;
    resultset.resize(answer_nq);
    for (size_t i = 0; i < answer_nq; ++i) {
        resultset[i].resize(answer_topk);
        for (size_t j = 0; j < answer_topk; ++j) {
            const auto idx = i*answer_topk + j;
            resultset[i][j].first = I[idx];
            resultset[i][j].second = D[idx];
        }
    }

    // print_vec_id_dis<DISTT, IDT>(resultset, "show resultset:");

    int max_recall, min_recall;
    // recall statistics
    std::vector<int> border = {0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100};
    int recall0 = 0;
    int recall100 = 0;
    std::vector<int> recall_hist(border.size(), 0);
    std::vector<int> recalls(answer_nq);

    int tot_cnt = 0;
    std::cout << "recall@" << answer_topk << " between groundtruth file:"
              << groundtruth_file << " and answer file:"
              << std::endl;
    if (cmp_id) {
        for (auto i = 0; i < answer_nq; i ++) {
            std::unordered_map<IDT, bool> hash;
            int cnti = 0;
            for (auto j = 0; j < answer_topk; j ++) {
                hash[groundtruth[i][j].first] = true;
            }
            for (auto j = 0; j < answer_topk; j ++) {
                if (hash.find(resultset[i][j].first) != hash.end())
                    cnti ++;
            }
            recalls[i] = cnti;
            tot_cnt += cnti;
        }
        std::cout << "avg recall@" << answer_topk << " = " << ((double)(tot_cnt)) / answer_topk / answer_nq * 100 << "%." << std::endl;
    } else {
        if (faiss::MetricType::METRIC_L2 == metric_type) {
            for (auto i = 0; i < answer_nq; i ++) {
                int cnti = 0;
                for (auto j = 0; j < answer_topk; j ++) {
                    if (resultset[i][j].second <= groundtruth[i][answer_topk - 1].second)
                        cnti ++;
                }
                recalls[i] = cnti;
                tot_cnt += cnti;
                // std::cout << "query " << i << " recall@" << answer_topk << " is: " << ((double)(cnti)) / answer_topk * 100 << "%." << std::endl;
            }
            std::cout << "avg recall@" << answer_topk << " = " << ((double)(tot_cnt)) / answer_topk / answer_nq * 100 << "%." << std::endl;
        } else if (faiss::MetricType::METRIC_INNER_PRODUCT == metric_type) {
            for (auto i = 0; i < answer_nq; i ++) {
                int cnti = 0;
                for (auto j = 0; j < answer_topk; j ++) {
                    if (resultset[i][j].second >= groundtruth[i][answer_topk - 1].second)
                        cnti ++;
                }
                recalls[i] = cnti;
                tot_cnt += cnti;
                // std::cout << "query " << i << " recall@" << answer_topk << " is: " << ((double)(cnti)) / answer_topk * 100 << "%." << std::endl;
            }
            std::cout << "avg recall@" << answer_topk << " = " << ((double)(tot_cnt)) / answer_topk / answer_nq * 100 << "%." << std::endl;
        } else {
            std::cout << "invalid metric type: " << (int)metric_type << std::endl;
        }
    }

    for (auto i = 0; i < answer_nq; i ++) {
        if (recalls[i] == 0) {
            recall0 ++;
            continue;
        }
        if (recalls[i] == answer_topk) {
            recall100 ++;
            continue;
        }
        for (auto j = 0; j < border.size() - 1; j ++) {
            if (recalls[i] * 100 >= border[j] * answer_topk && recalls[i] * 100 < border[j + 1] * answer_topk) {
                recall_hist[j] ++;
                break;
            }
        }
    }
    int check_sum = recall0 + recall100;
    std::cout << "show more details about recall histogram:" << std::endl;
    std::cout << "recall@" << answer_topk << " in range [0, 0]: " << recall0 << std::endl;
    for (auto i = 0; i < border.size() - 1; i ++) {
        std::cout << "recall@" << answer_topk << " in range [" << border[i] << ", " <<  border[i + 1] << "): " << recall_hist[i] << std::endl;
        check_sum += recall_hist[i];
    }
    std::cout << "recall@" << answer_topk << " in range [100, 100]: " << recall100 << std::endl;
    std::cout << "check sum recall: " << check_sum << ", which should equal nq: " << answer_nq << std::endl;
}

template<typename T1, typename T2, typename R>
R IP(T1 *a, T2 *b, size_t n) {
    size_t i = 0;
    R dis = 0;
    switch(n & 7) {
        default:
            while (n > 7) {
                n -= 8; dis+=(R)a[i]*(R)b[i]; i++;
                case 7: dis+=(R)a[i]*(R)b[i]; i++;
                case 6: dis+=(R)a[i]*(R)b[i]; i++;
                case 5: dis+=(R)a[i]*(R)b[i]; i++;
                case 4: dis+=(R)a[i]*(R)b[i]; i++;
                case 3: dis+=(R)a[i]*(R)b[i]; i++;
                case 2: dis+=(R)a[i]*(R)b[i]; i++;
                case 1: dis+=(R)a[i]*(R)b[i]; i++;
            }
    }
    return dis;
}

template<typename T1, typename T2, typename R>
R L2sqr(T1 *a, T2 *b, size_t n) {
    size_t i = 0;
    R dis = 0, dif;
    switch(n & 7) {
        default:
            while (n > 7) {
                n -= 8; dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 7: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 6: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 5: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 4: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 3: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 2: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 1: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
            }
    }
    return dis;
}

static float* compute_residuals(
        const faiss::Index* quantizer,
        faiss::Index::idx_t n,
        const float* x,
        const faiss::Index::idx_t* list_nos) {
    size_t d = quantizer->d;
    float* residuals = new float[n * d];
    // TODO: parallelize?
    for (size_t i = 0; i < n; i++) {
        if (list_nos[i] < 0)
            memset(residuals + i * d, 0, sizeof(*residuals) * d);
        else
            quantizer->compute_residual(
                    x + i * d, residuals + i * d, list_nos[i]);
    }
    return residuals;
}

int main() {
    int32_t d;
    int32_t nb;

    size_t loops = 1;

#ifdef YANDEX_TEXT_TO_IMAGE
    FILE* f = read_file_head("/home/pat/datasets/Yandex-Text-to-Image/base.10M.fdata", &nb, &d);
#elif defined(YANDEX_DEEP)
    FILE* f = read_file_head("/home/pat/datasets/Yandex-DEEP/base.10M.fbin", &nb, &d);
#elif defined(MT_ANNS)
    FILE* f = read_file_head("/home/pat/datasets/Microsoft-Turing-ANNS/base1b.10M.fbin", &nb, &d);
#endif

    float *xb = new float[d * nb];
    read_file_data(f, nb, d, xb);
    fclose(f);

    printf("%ld %ld\n", d, nb);

    // for (int i = 0; i < 10; ++i) {
    //     for (int j = 0; j < d; ++j) {
    //         // printf("%f ", xb[i*d+j]);
    //     }
    //     // printf("\n");
    // }

    // const char *index_key = "IVF2000,PQ40,OPQ40";
    // const char *index_key = "IVF20000,PQ40";

    // faiss::Index* index = faiss::index_factory(d, index_key);
    // faiss::ParameterSpace().set_index_parameter(index, "nprobe", 16);
    // index->metric_type = faiss::MetricType::METRIC_L2;

    int nlist = 2000;

    faiss::MetricType metric_type = faiss::MetricType::METRIC_L2;
#ifdef YANDEX_TEXT_TO_IMAGE
    int m = 40;
    metric_type = faiss::MetricType::METRIC_INNER_PRODUCT;
#elif defined(YANDEX_DEEP)
    int m = 48;
#elif defined(MT_ANNS)
    int m = 50;
#endif

    faiss::IndexFlatL2 quantizer(d);
    faiss::OPQMatrix* opq = new faiss::OPQMatrix(d, m);
    faiss::IndexIVFPQ index(&quantizer, d, nlist, m, 8, metric_type, opq);
    // faiss::IndexIVFPQ index(&quantizer, d, nlist, m, 8, metric_type);
    // index.verbose = true;

    // faiss::gpu::StandardGpuResources res;
    // faiss::gpu::GpuIndexFlatIP index(&res, d);
    // faiss::gpu::GpuIndexIVFFlat index(&res, d, nlist, faiss::METRIC_L2);
    // faiss::gpu::GpuIndexIVFPQ index(&res, d, nlist, m, 8, faiss::METRIC_L2);
    // faiss::gpu::GpuIndexIVFScalarQuantizer index(&res, d, nlist, faiss::QuantizerType::QT_8bit);

    index.nprobe = 16;

    // printf("by_residual: %d\n", index.by_residual);
    // printf("assign_index: %p\n", index.pq.assign_index);


    // index.train(nb, xb);
    // index.add(nb, xb);

    // index.train(3e4, xb);
    index.train(nb, xb);
    // index->train(nb, xb);
    // printf("is_trained = %s\n", index->is_trained ? "true" : "false");
    // index->add(nb, xb);                     // add vectors to the index
    index.add(nb, xb);                     // add vectors to the index
    // index.add(3e4, xb);                     // add vectors to the index
    // printf("ntotal = %ld\n", index->ntotal);

    // printf("by_residual: %d\n", index->by_residual);


    int32_t nq;
    int32_t d2;

#ifdef YANDEX_TEXT_TO_IMAGE
    f = read_file_head("/home/pat/datasets/Yandex-Text-to-Image/query.public.100K.fbin", &nq, &d2);
#elif defined(YANDEX_DEEP)
    f = read_file_head("/home/pat/datasets/Yandex-DEEP/query.public.10K.fbin", &nq, &d2);
#elif defined(MT_ANNS)
    f = read_file_head("/home/pat/datasets/Microsoft-Turing-ANNS/query100K.fbin", &nq, &d2);
#endif

    float *xq = new float[nq * d2];
    read_file_data(f, nq, d2, xq);
    fclose(f);


#ifdef YANDEX_TEXT_TO_IMAGE
    std::string gt_file = "/home/pat/datasets/Yandex-Text-to-Image/text2image-10M-gt";
#elif defined(YANDEX_DEEP)
    std::string gt_file = "/home/pat/datasets/Yandex-DEEP/deep-10M-gt";
#elif defined(MT_ANNS)
    std::string gt_file = "/home/pat/datasets/Microsoft-Turing-ANNS/msturing-10M-gt";
#endif

    assert(d == d2 || !"query does not have same dimension as train set");
    size_t k = 10; // nb of results per query in the GT

    printf("d: %ld\n", d);
    printf("nb: %ld\n", nb);
    printf("nq: %ld\n", nq);
    printf("k: %ld\n", k);

    {
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        // faiss::Index::idx_t* idx0 = new faiss::Index::idx_t[nq];
        // quantizer.assign(nq, xq, idx0);
        // xq = compute_residuals(&quantizer, nq, xq, idx0);

        index.search(nq, xq, k, D, I);
        // index->search(nq, xq, k, D, I);

        // double avg = 0.0f, dur;
        // double min_time = std::numeric_limits<double>::max();

        // for (int i = 0; i < loops; i++) {
        //     printf("Loop %d\n", i);
        //     double t0 = elapsed();
        //     index.search(nq, xq, k, D, I);

        //     dur = elapsed() - t0;
        //     avg += dur;
        //     min_time = std::min(min_time, dur);
        // }
        // avg /= loops;
        printf("search done\n");
        for (auto i = 0; i < nq; ++i) {
            for (auto j = 0; j < k; ++j) {
                auto idx = i * k + j;
                if (metric_type == faiss::MetricType::METRIC_L2) {
                    D[idx] = L2sqr<const float, const float, float>(xb + I[idx]*d, xq + i*d, (size_t)d);
                } else {
                    D[idx] = IP<const float, const float, float>(xb + I[idx]*d, xq + i*d, (size_t)d);
                }
            }
        }
        recall<float, long>(gt_file, D, I, nq, k, metric_type, false);
        delete [] I;
        delete [] D;
        // delete [] idx0;
    }

    // delete opq;

    delete[] xb;
    delete[] xq;

#if 0
    // First SQ8, then IndexRefineFlat
    {
        long *I = new long[k * nq];
        float *D = new float[k * nq];
        faiss::Index* sq = new faiss::IndexScalarQuantizer(d, faiss::QuantizerType::QT_8bit);
        faiss::IndexRefineFlat rf = faiss::IndexRefineFlat(sq);
        rf.train(nb, xb);
        rf.add(nb, xb);
        rf.k_factor = 4;
        rf.search(nq, xq, k, D, I);
        double avg = 0.0f;
        for (int i = 0; i < loops; i++) {
            double t0 = elapsed();
            rf.search(nq, xq, k, D, I);
            avg += elapsed() - t0;
        }
        avg /= loops;
        printf("RefineFlat Recall: %.4f, time spent: %.3fs\n", CalcRecall(k, k, nq, gt, I), avg);
        delete [] I;
        delete [] D;
    }

    // SQ8 alone
    {
        long *I = new long[k * nq];
        float *D = new float[k * nq];
        auto sq = new faiss::IndexScalarQuantizer(d, faiss::QuantizerType::QT_8bit);
        sq->train(nb, xb);
        sq->add(nb, xb);
        sq->search(nq, xq, k, D, I);
        double avg = 0.0f;
        for (int i = 0; i < loops; i++) {
            double t0 = elapsed();
            sq->search(nq, xq, k, D, I);
            avg += elapsed() - t0;
        }
        avg /= loops;
        printf("SQ8 Recall: %.4f, time spent: %.3fs\n", CalcRecall(k, k, nq, gt, I), avg);
        delete [] I;
        delete [] D;
    }
#endif

    return 0;
}

#if 0
int main()
{
    double t0 = elapsed();

    // this is typically the fastest one.
    const char *index_key = "IVF2048,Flat";

    // these ones have better memory usage
    // const char *index_key = "Flat";
    // const char *index_key = "PQ32";
    // const char *index_key = "PCA80,Flat";
    // const char *index_key = "IVF4096,PQ8+16";
    // const char *index_key = "IVF4096,PQ32";
    // const char *index_key = "IMI2x8,PQ32";
    // const char *index_key = "IMI2x8,PQ8+16";
    // const char *index_key = "OPQ16_64,IMI2x8,PQ8+16";

    faiss::Index * index;

    size_t d;

    {
        printf ("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
        float *xt = fvecs_read("sift1M/sift_learn.fvecs", &d, &nt);

        printf ("[%.3f s] Preparing index \"%s\" d=%ld\n",
                elapsed() - t0, index_key, d);
        index = faiss::index_factory(d, index_key);

        printf ("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);

        index->train(nt, xt);
        delete [] xt;
    }


    {
        printf ("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        float *xb = fvecs_read("sift1M/sift_base.fvecs", &d2, &nb);
        assert(d == d2 || !"dataset does not have same dimension as train set");

        printf ("[%.3f s] Indexing database, size %ld*%ld\n",
                elapsed() - t0, nb, d);

        index->add(nb, xb);

        delete [] xb;
    }

    size_t nq;
    float *xq;

    {
        printf ("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        xq = fvecs_read("sift1M/sift_query.fvecs", &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");

    }

    size_t k; // nb of results per query in the GT
    faiss::Index::idx_t *gt;  // nq * k matrix of ground-truth nearest-neighbors

    {
        printf ("[%.3f s] Loading ground truth for %ld queries\n",
                elapsed() - t0, nq);

        // load ground-truth and convert int to long
        size_t nq2;
        int *gt_int = ivecs_read("sift1M/sift_groundtruth.ivecs", &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::Index::idx_t[k * nq];
        for(int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete [] gt_int;
    }

    // // Result of the auto-tuning
    // std::string selected_params;

    // { // run auto-tuning

    //     printf ("[%.3f s] Preparing auto-tune criterion 1-recall at 1 "
    //             "criterion, with k=%ld nq=%ld\n", elapsed() - t0, k, nq);

    //     faiss::OneRecallAtRCriterion crit(nq, 1);
    //     crit.set_groundtruth (k, nullptr, gt);
    //     crit.nnn = k; // by default, the criterion will request only 1 NN

    //     printf ("[%.3f s] Preparing auto-tune parameters\n", elapsed() - t0);

    //     faiss::ParameterSpace params;
    //     params.initialize(index);

    //     printf ("[%.3f s] Auto-tuning over %ld parameters (%ld combinations)\n",
    //             elapsed() - t0, params.parameter_ranges.size(),
    //             params.n_combinations());

    //     faiss::OperatingPoints ops;
    //     params.explore (index, nq, xq, crit, &ops);

    //     printf ("[%.3f s] Found the following operating points: \n",
    //             elapsed() - t0);

    //     ops.display ();

    //     // keep the first parameter that obtains > 0.5 1-recall@1
    //     for (int i = 0; i < ops.optimal_pts.size(); i++) {
    //         if (ops.optimal_pts[i].perf > 0.5) {
    //             selected_params = ops.optimal_pts[i].key;
    //             break;
    //         }
    //     }
    //     assert (selected_params.size() >= 0 ||
    //             !"could not find good enough op point");
    // }


    { // Use the found configuration to perform a search

        faiss::ParameterSpace params;

        printf ("[%.3f s] Setting parameter configuration \"%s\" on index\n",
                elapsed() - t0, selected_params.c_str());

        params.set_index_parameters (index, selected_params.c_str());

        printf ("[%.3f s] Perform a search on %ld queries\n",
                elapsed() - t0, nq);

        // output buffers
        faiss::Index::idx_t *I = new  faiss::Index::idx_t[nq * k];
        float *D = new float[nq * k];

        index->search(nq, xq, k, D, I);

        printf ("[%.3f s] Compute recalls\n", elapsed() - t0);

        // evaluate result by hand.
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for(int i = 0; i < nq; i++) {
            int gt_nn = gt[i * k];
            for(int j = 0; j < k; j++) {
                if (I[i * k + j] == gt_nn) {
                    if(j < 1) n_1++;
                    if(j < 10) n_10++;
                    if(j < 100) n_100++;
                }
            }
        }
        printf("R@1 = %.4f\n", n_1 / float(nq));
        printf("R@10 = %.4f\n", n_10 / float(nq));
        printf("R@100 = %.4f\n", n_100 / float(nq));

        delete [] I;
        delete [] D;

    }

    delete [] xq;
    delete [] gt;
    delete index;
    return 0;
}
#endif