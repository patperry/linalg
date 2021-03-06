#ifndef LINALG_SBLAS_H
#define LINALG_SBLAS_H

#include <stddef.h>
#include "blas.h"

struct vpattern {
	size_t nz, nzmax;
	size_t *indx;
};

void vpattern_init(struct vpattern *pat);
void vpattern_deinit(struct vpattern *pat);

void vpattern_clear(struct vpattern *pat);
size_t vpattern_grow(struct vpattern *pat, size_t delta);
ptrdiff_t vpattern_find(const struct vpattern *pat, size_t i);
size_t vpattern_lb(const struct vpattern *pat, size_t i);
size_t vpattern_ub(const struct vpattern *pat, size_t i);
size_t vpattern_search(struct vpattern *pat, size_t i, int *insp);

static inline struct vpattern vpattern_make(const size_t *indx, size_t nz)
{
	struct vpattern pat;
	pat.nz = nz;
	pat.nzmax = nz;
	pat.indx = (size_t *)indx;
	return pat;
}

/* Level 1 */

/* y[indx[i]] += alpha * x[i] for i = 0, ..., nz-1 */
void sblas_daxpyi(size_t nz, double alpha, const double *x, const size_t *indx,
		  double *y);

/* x[0] * y[indx[0]] + ... + x[nz-1] * y[indx[nz-1]] */
double sblas_ddoti(size_t nz, const double *x, const size_t *indx,
		   const double *y);

/* x[i] := y[indx[i]] for i = 0, ..., nz-1 */
void sblas_dgthr(size_t nz, const double *y, double *x, const size_t *indx);

/* x[i] := y[indx[i]]; * y[indx[i]] := 0 for i = 0, ..., nz-1 */
void sblas_dgthrz(size_t nz, double *y, double *x, const size_t *indx);

/* y[indx[i]] = x[i] for i = 0, ..., nz-1 */
void sblas_dsctr(size_t nz, const double *x, const size_t *indx, double *y);

/* Level 2 */

/* y = alpha * \sum_{i = 0}^{nz-1} { op(a)[,indx[i]] * x[i] }  +  beta * y */
void sblas_dgemvi(enum blas_trans trans, size_t m, size_t n, size_t nz,
		  double alpha, const double *a, size_t lda, const double *x,
		  const size_t *indx, double beta, double *y);

void sblas_dcscmv(enum blas_trans trans, size_t m, size_t n, double alpha,
		  const double *a, const size_t *inda, const size_t *offa,
		  const double *x, double beta, double *y);

void sblas_dcscsctr(enum blas_trans trans, size_t n,
		    const double *a, const size_t *inda, const size_t *offa,
		    double *b, size_t ldb);

#endif /* LINALG_SBLAS_H */
