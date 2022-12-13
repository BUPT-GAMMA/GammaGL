#define IDXTYPEWIDTH 64
#define REALTYPEWIDTH 32


#ifndef _GKLIB_H_
#ifdef COMPILER_MSC
#include <limits.h>
typedef __int64 int64_t;
#else
#include <inttypes.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

int fake_function_0(int64_t *nvtxs, int64_t *ncon, int64_t *xadj,
                 int64_t *adjncy, int64_t *vwgt, int64_t *vsize, int64_t *adjwgt,
                 int64_t *nparts, float *tpwgts, float *ubvec, int64_t *options,
                 int64_t *edgecut, int64_t *part);

int fake_function_1(int64_t *nvtxs,int64_t *ncon, int64_t *xadj,
                  int64_t *adjncy, int64_t *vwgt, int64_t *vsize, int64_t *adjwgt,
                  int64_t *nparts, float *tpwgts, float *ubvec, int64_t *options,
                  int64_t *edgecut, int64_t *part);


#ifdef __cplusplus
}
#endif