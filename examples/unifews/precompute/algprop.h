/*
 * Author: nyLiao
 * File Created: 2023-04-19
 * File: algprop.h
 */
#ifndef ALGPROP_H
#define ALGPROP_H
#include <iostream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <queue>
#include <unordered_map>
#include <math.h>
#include <cmath>
#include <limits.h>
#include <unistd.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <thread>
#include <string>
#include <unistd.h>
#include <chrono>
#include <omp.h>
#include <sys/time.h>
#include <sys/resource.h>
#pragma warning(push, 0)
#include <Eigen/Dense>
#pragma warning(pop)

using namespace std;
using namespace Eigen;
typedef unsigned int uint;

namespace algprop {
    const int NUMTHREAD = 32;           // Number of threads

    struct Channel {                    // channel scheme
        int type;
            // -2: SGC, -3: APPNP
            // 0: SGC_AGP, 1: APPNP_AGP
            // 2: SGC_thr, 3: APPNP_thr
        bool is_thr;                    // is threshold
        bool is_acc;                    // is accumulate (APPNP)

        int hop;                        // propagation hop
        int dim;                        // feature dimension
        float delta;                    // absolute error
        float alpha;                    // summation decay, alpha=0 for SGC
        float rra, rrb;                 // left & right normalization
    };

    class A2prop{
    public:
    	uint m,n,seed;                  // edges, nodes, seed
        vector<uint> el;
        vector<uint> pl;
        Eigen::ArrayXf map_feat;        // permuted index -> index in feats
        Eigen::ArrayXi map_chn;         // index in chns -> channel type
        Eigen::ArrayXf macs;            // MACs per feature

        Channel* chns;                  // channel schemes
        Eigen::ArrayXf deg;             // node degree vector
        Eigen::ArrayXf dega, dinva;     //  left-norm degree, inversed deg_a
        Eigen::ArrayXf dinvb;           // right-norm degree, inversed deg_b
        Eigen::ArrayXf dlt_p, dlt_n;    // absolute error (positive, negative)
        Eigen::ArrayXf maxf_p, maxf_n;  // max feature coefficient

        void load(string dataset, uint mm, uint nn, uint seedd);
        float compute(uint nchnn, Channel* chnss, Eigen::Map<Eigen::MatrixXf> &feat, float &time);

        void feat_chn(Eigen::Ref<Eigen::MatrixXf>feats,int st,int ed);
        void feat_ori(Eigen::Ref<Eigen::MatrixXf>feats,int st,int ed);
    };
}

#endif // ALGPROP_H
