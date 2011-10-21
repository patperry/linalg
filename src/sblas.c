#include <assert.h>
#include <string.h>
#include "sblas.h"

/* y[indx[i]] += alpha * x[i] for i = 0, ..., nz-1 */
void sblas_daxpyi(size_t nz, double alpha, const double *x,
		  const size_t *indx, double *y)
{
	size_t i;

	for (i = 0; i < nz; i++) {
		y[indx[i]] += alpha * x[i];
	}
}

/* x[0] * y[indx[0]] + ... + x[nz-1] * y[indx[nz-1]] */
double sblas_ddoti(size_t nz, const double *x, const size_t *indx,
		   const double *y)
{
	double dot = 0.0;
	size_t i;

	for (i = 0; i < nz; i++) {
		dot += x[i] * y[indx[i]];
	}

	return dot;
}

/* x[i] := y[indx[i]] for i = 0, ..., nz-1 */
void sblas_dgthr(size_t nz, const double *y, double *x, const size_t *indx)
{
	size_t i;

	for (i = 0; i < nz; i++) {
		x[i] = y[indx[i]];
	}
}

/* x[i] := y[indx[i]]; * y[indx[i]] := 0 for i = 0, ..., nz-1 */
void sblas_dgthrz(size_t nz, double *y, double *x, const size_t *indx)
{
	size_t i;

	for (i = 0; i < nz; i++) {
		x[i] = y[indx[i]];
		y[indx[i]] = 0;
	}
}

/* y[indx[i]] = x[i] for i = 0, ..., nz-1 */
void sblas_dsctr(size_t nz, const double *x, const size_t *indx, double *y)
{
	size_t i;

	for (i = 0; i < nz; i++) {
		y[indx[i]] = x[i];
	}
}

void sblas_dgemvi(enum blas_trans trans, size_t m, size_t n, size_t nz,
		  double alpha, const double *a, size_t lda, const double *x,
		  const size_t *indx, double beta, double *y)
{
	size_t ny = (trans == BLAS_NOTRANS ? m : n);

	if (beta == 0) {
		memset(y, 0, ny * sizeof(y[0]));
	} else if (beta != 1) {
		blas_dscal(ny, beta, y, 1);
	}

	if (trans == BLAS_NOTRANS) {
		size_t i;

		for (i = 0; i < nz; i++) {
			blas_daxpy(m, alpha * x[i], a + indx[i] * lda, 1, y, 1);
		}
	} else {
		size_t j;

		for (j = 0; j < n; j++) {
			y[j] += alpha * sblas_ddoti(nz, x, indx, a + j * lda);
		}
	}
}

ptrdiff_t sblas_find(size_t nz, const size_t *indx, size_t i)
{
	const size_t *base = indx, *ptr;

	for (; nz != 0; nz >>= 1) {
		ptr = base + (nz >> 1);
		if (i == *ptr)
			return ptr - indx;
		if (i > *ptr) {
			base = ptr + 1;
			nz--;
		}
	}

	ptrdiff_t ix = base - indx;
	return ~ix;
}

void sblas_dcscmv(enum blas_trans trans, size_t m, size_t n, double alpha,
		  const double *a, const size_t *inda, const size_t *offa,
		  const double *x, double beta, double *y)
{
	size_t ny = (trans == BLAS_NOTRANS ? m : n);
	size_t j;

	if (beta == 0) {
		memset(y, 0, ny * sizeof(y[0]));
	} else if (beta != 1) {
		blas_dscal(ny, beta, y, 1);
	}

	if (trans == BLAS_NOTRANS) {
		for (j = 0; j < n; j++) {
			size_t nz = offa[j+1] - offa[j];
			const double *val = a + offa[j];
			const size_t *ind = inda + offa[j];

			sblas_daxpyi(nz, alpha * x[j], val, ind, y);
		}
	} else {
		for (j = 0; j < n; j++) {
			size_t nz = offa[j+1] - offa[j];
			const double *val = a + offa[j];
			const size_t *ind = inda + offa[j];

			y[j] += alpha * sblas_ddoti(nz, val, ind, x);
		}
	}
}


void sblas_dcscsctr(enum blas_trans trans, size_t n,
		    const double *a, const size_t *inda, const size_t *offa,
		    double *b, size_t ldb)
{
	size_t j;

	if (trans == BLAS_NOTRANS) {
		for (j = 0; j < n; j++) {
			const double *val = a + offa[j];
			const size_t *ind = inda + offa[j];
			size_t iz, nz = offa[j+1] - offa[j];

			for (iz = 0; iz < nz; iz++) {
				b[ind[iz] + j * ldb] = val[iz];
			}
		}
	} else {
		for (j = 0; j < n; j++) {
			const double *val = a + offa[j];
			const size_t *ind = inda + offa[j];
			size_t iz, nz = offa[j+1] - offa[j];

			for (iz = 0; iz < nz; iz++) {
				b[j + ind[iz] * ldb] = val[iz];
			}
		}
	}
}
