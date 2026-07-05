/*
 * Author: nyLiao
 * File Created: 2023-04-19
 * File: algprop.cpp
 * Ref: [AGP](https://github.com/wanghzccls/AGP-Approximate_Graph_Propagation)
 */
#include "algprop.h"
using namespace std;
using namespace Eigen;

// ====================
double get_curr_time() {
    long long time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    return static_cast<double>(time) / 1000000.0;
}

float get_proc_memory(){
    struct rusage r_usage;
    getrusage(RUSAGE_SELF,&r_usage);
    return r_usage.ru_maxrss/1000000.0;
}

float get_stat_memory(){
    long rss;
    std::string ignore;
    std::ifstream ifs("/proc/self/stat", std::ios_base::in);
    ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
        >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
        >> ignore >> ignore >> ignore >> rss;

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;
    return rss * page_size_kb / 1000000.0;
}

inline void update_maxr(const float r, float &maxrp, float &maxrn) {
    if (r > maxrp)
        maxrp = r;
    else if (r < maxrn)
        maxrn = r;
}

// ====================
namespace algprop {
// Load graph related data
void A2prop::load(string dataset, uint mm, uint nn, uint seedd) {
    m = mm;
    n = nn;
    seed = seedd;

    // Load graph adjacency
    el = vector<uint>(m);   // edge list sorted by source node degree
    pl = vector<uint>(n + 1);
    string dataset_el = dataset + "/adj_el.bin";
    const char *p1 = dataset_el.c_str();
    if (FILE *f1 = fopen(p1, "rb")) {
        size_t rtn = fread(el.data(), sizeof el[0], el.size(), f1);
        if (rtn != m)
            cout << "Error! " << dataset_el << " Incorrect read!" << endl;
        fclose(f1);
    } else {
        cout << dataset_el << " Not Exists." << endl;
        exit(1);
    }
    string dataset_pl = dataset + "/adj_pl.bin";
    const char *p2 = dataset_pl.c_str();
    if (FILE *f2 = fopen(p2, "rb")) {
        size_t rtn = fread(pl.data(), sizeof pl[0], pl.size(), f2);
        if (rtn != n + 1)
            cout << "Error! " << dataset_pl << " Incorrect read!" << endl;
        fclose(f2);
    } else {
        cout << dataset_pl << " Not Exists." << endl;
        exit(1);
    }

    deg = Eigen::ArrayXf::Zero(n);
    for (uint i = 0; i < n; i++) {
        deg(i)   = pl[i + 1] - pl[i];
        if (deg(i) <= 0) {
            deg(i) = 1;
            // cout << i << " ";
        }
    }
}

// Computation call entry
float A2prop::compute(uint nchnn, Channel* chnss, Eigen::Map<Eigen::MatrixXf> &feat, float &ttime) {
    // Node-specific array
    chns = chnss;
    assert(nchnn <= 4);
    dega = Eigen::ArrayXf::Zero(n);
    dinva = Eigen::ArrayXf::Zero(n);
    dinvb = Eigen::ArrayXf::Zero(n);
    for (uint c = 0; c < nchnn; c++) {
        dega = deg.pow(chns[c].rra);
        dinva = 1 / dega;
        dinvb = 1 / deg.pow(chns[c].rrb);
    }

    // Feat is ColMajor, shape: (n, c*F)
    int fsum = feat.cols();
    int it = 0;
    map_feat = Eigen::ArrayXf::LinSpaced(fsum, 0, fsum - 1);
    // random_shuffle(map_feat.data(), map_feat.data() + map_feat.size());
    cout << "feat dim: " << feat.cols() << ", nodes: " << feat.rows() << ", edges: " << m <<  ". ";

    // Feature-specific array
    dlt_p = Eigen::ArrayXf::Zero(fsum);
    dlt_n = Eigen::ArrayXf::Zero(fsum);
    maxf_p = Eigen::ArrayXf::Zero(fsum);
    maxf_n = Eigen::ArrayXf::Zero(fsum);
    map_chn = Eigen::ArrayXi::Zero(fsum);
    macs = Eigen::ArrayXf::Zero(fsum);
    // Loop each feature index `it`, inside channel index `i`
    for (uint c = 0; c < nchnn; c++) {
        for (int i = 0; i < chns[c].dim; i++) {
            for (uint u = 0; u < n; u++) {
                if (feat(u, i) > 0)
                    dlt_p(it) += feat(u, it) * pow(deg(u), chns[c].rrb);
                else
                    dlt_n(it) += feat(u, it) * pow(deg(u), chns[c].rrb);
                update_maxr(feat(u, it), maxf_p(it), maxf_n(it));
            }
            if (dlt_p(it) == 0)
                dlt_p(it) = 1e-12;
            if (dlt_n(it) == 0)
                dlt_n(it) = -1e-12;
            dlt_p(it) *= chns[c].delta / (1 - chns[c].alpha);
            dlt_n(it) *= chns[c].delta / (1 - chns[c].alpha);
            map_chn(it) = c;
            it++;
        }
    }

    // Begin propagation
    cout << "Propagating..." << endl;
    struct timeval ttod_start, ttod_end;
    double ttod, tclk;
    gettimeofday(&ttod_start, NULL);
    tclk = get_curr_time();
    int dim_top = 0;
    int start, ends = dim_top;

    vector<thread> threads;
    for (it = 1; it <= fsum % NUMTHREAD; it++) {
        start = ends;
        ends += ceil((float)fsum / NUMTHREAD);
        if (chns[0].type < 0)
            threads.push_back(thread(&A2prop::feat_ori, this, feat, start, ends));
        else
            threads.push_back(thread(&A2prop::feat_chn, this, feat, start, ends));
    }
    for (; it <= NUMTHREAD; it++) {
        start = ends;
        ends += fsum / NUMTHREAD;
        if (chns[0].type < 0)
            threads.push_back(thread(&A2prop::feat_ori, this, feat, start, ends));
        else
            threads.push_back(thread(&A2prop::feat_chn, this, feat, start, ends));
    }
    for (int t = 0; t < NUMTHREAD; t++)
        threads[t].join();
    vector<thread>().swap(threads);

    tclk = get_curr_time() - tclk;
    gettimeofday(&ttod_end, NULL);
    ttod = ttod_end.tv_sec - ttod_start.tv_sec + (ttod_end.tv_usec - ttod_start.tv_usec) / 1000000.0;
    cout << "[Pre] Prop  time: " << ttod << " s, ";
    cout << "Clock time: " << tclk << " s, ";
    cout << "Max   PRAM: " << get_proc_memory() << " GB, ";
    cout << "End    RAM: " << get_stat_memory() << " GB, ";
    cout << "MACs: " << macs.sum()/1e9 << " G" << endl;
    ttime = ttod;
    return macs.sum();
}

// ====================
// Feature embs
void A2prop::feat_chn(Eigen::Ref<Eigen::MatrixXf> feats, int st, int ed) {
    uint seedt = seed;
    Eigen::VectorXf res0(n), res1(n);
    Eigen::Map<Eigen::VectorXf> rprev(res1.data(), n), rcurr(res0.data(), n);

    // Loop each feature `ift`, index `it`
    for (int it = st; it < ed; it++) {
        const uint ift = map_feat(it);
        const Channel chn = chns[0];
        const float alpha = chn.alpha;
        vector<uint> plshort(pl), plshort2(pl);
        Eigen::Map<Eigen::VectorXf> feati(feats.col(ift).data(), n);

        const float dlti_p = dlt_p(ift);
        const float dlti_n = dlt_n(ift);
        const float dltinv_p = 1 / dlti_p;
        const float dltinv_n = 1 / dlti_n;
        float maxr_p = maxf_p(ift);     // max positive residue
        float maxr_n = maxf_n(ift);     // max negative residue
        uint maccnt = 0;

        // Init residue
        res1.setZero();
        res0 = feats.col(ift);
        feati.setZero();
        rprev = res1;
        rcurr = res0;

        // Loop each hop `il`
        int il;
        for (il = 0; il < chn.hop; il++) {
            // Early termination
            if ((maxr_p <= dlti_p) && (maxr_n >= dlti_n))
                break;
            rcurr.swap(rprev);
            rcurr.setZero();

            // Loop each node `u`
            for (uint u = 0; u < n; u++) {
                const float old = rprev(u);
                float thr_p = old * dltinv_p;
                float thr_n = old * dltinv_n;
                // if ((!chn.is_acc) && (m < 1e9)) {
                if (!chn.is_acc) {
                    rcurr(u) += old;
                }
                if (thr_p > 1 || thr_n > 1) {
                    float oldb = 0;
                    if (chn.is_acc) {
                        feati(u) += old * alpha;
                        oldb = old * (1-alpha) * dinvb(u);
                    }

                    // Loop each neighbor index `im`, node `v`
                    uint iv, iv2;
                    const uint ivmax = (chn.is_thr) ? plshort[u+1] : pl[u+1];
                    for (iv = pl[u]; iv < ivmax; iv++) {
                        const uint v = el[iv];
                        const float da_v = dega(v);
                        if (thr_p > da_v || thr_n > da_v) {
                            maccnt++;
                            if (chn.is_acc)
                                rcurr(v) += oldb * dinva(v);
                            else
                                rcurr(v) += old / deg(v);
                            update_maxr(rcurr(v), maxr_p, maxr_n);
                        } else {
                            // plshort[u+1] = iv;
                            break;
                        }
                    }

                    iv2 = iv;
                    const float ran = (float)RAND_MAX / (rand_r(&seedt) % RAND_MAX);
                    thr_p *= ran;
                    thr_n *= ran;
                    const uint ivmax2 = (chn.is_thr) ? plshort2[u+1] : pl[u+1];
                    for (; iv < ivmax2; iv++) {
                        const uint v = el[iv];
                        const float da_v = dega(v);
                        if (thr_p > da_v) {
                            maccnt++;
                            rcurr(v) += dlti_p * dinva(v);
                            update_maxr(rcurr(v), maxr_p, maxr_n);
                        } else if (thr_n > da_v) {
                            maccnt++;
                            rcurr(v) += dlti_n * dinva(v);
                            update_maxr(rcurr(v), maxr_p, maxr_n);
                        } else {
                            break;
                        }
                    }
                    plshort[u+1]  = (iv + iv2) / 2;
                    if (m < 1e8){
                        plshort2[u+1] = (iv + pl[u+1]) / 2;
                    }

                } else {
                    if (chn.is_acc)
                        feati(u) += old;
                }
            }
        }

        feati += rcurr;
        macs(ift) += (float)maccnt;
    }
}


void A2prop::feat_ori(Eigen::Ref<Eigen::MatrixXf> feats, int st, int ed) {
    Eigen::VectorXf res0(n), res1(n);
    Eigen::Map<Eigen::VectorXf> rprev(res1.data(), n), rcurr(res0.data(), n);

    // Loop each feature `ift`, index `it`
    for (int it = st; it < ed; it++) {
        const uint ift = map_feat(it);
        const Channel chn = chns[0];
        const float alpha = chn.alpha;
        Eigen::Map<Eigen::VectorXf> feati(feats.col(ift).data(), n);
        uint maccnt = 0;

        // Init residue
        res1.setZero();
        res0 = feats.col(ift);
        feati.setZero();
        rprev = res1;
        rcurr = res0;

        // Loop each hop `il`
        int il;
        for (il = 0; il < chn.hop; il++) {
            rcurr.swap(rprev);
            rcurr.setZero();

            // Loop each node `u`
            for (uint u = 0; u < n; u++) {
                const float old = rprev(u);
                float oldb = 0;
                if (chn.is_acc) {
                    feati(u) += old * alpha;
                    oldb = old * (1-alpha) * dinvb(u);
                }

                // Loop each neighbor index `im`, node `v`
                uint iv;
                for (iv = pl[u]; iv < pl[u+1]; iv++) {
                    const uint v = el[iv];
                    maccnt++;
                    if (chn.is_acc)
                        rcurr(v) += oldb * dinva(v);
                    else
                        rcurr(v) += old / deg(v);
                }
            }
        }

        feati += rcurr;
        macs(ift) += (float)maccnt;
    }
}

}  // namespace propagation
