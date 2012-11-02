#include <assert.h>
#include <string.h>
#include "blas.h"
#include "lapack.h"
#include "matrixutil.h"

void matrix_dzero(size_t m, size_t n, double *a, size_t lda)
{
	if (lda == m) {
		memset(a, 0, m * n * sizeof(double));
	} else {
		size_t j;
		for (j = 0; j < n; j++) {
			memset(a + j * lda, 0, m * sizeof(double));
		}
	}
}

void matrix_dscal(size_t m, size_t n, double alpha, double *a, size_t lda)
{
	if (lda == m) {
		blas_dscal(m * n, alpha, a, 1);
	} else {
		size_t j;

		for (j = 0; j < n; j++) {
			blas_dscal(m, alpha, a + j * lda, 1);
		}
	}
}

void matrix_dscalr(size_t m, size_t n, double *s, size_t incs, double *a, size_t lda)
{
	size_t i;

	for (i = 0; i < m; i++) {
		blas_dscal(n, s[i * incs], a + i, lda);
	}
}

void matrix_dscalc(size_t m, size_t n, double *s, size_t incs, double *a, size_t lda)
{
	size_t j;

	for (j = 0; j < n; j++) {
		blas_dscal(m, s[j * incs], a + j * lda, 1);
	}
}


void matrix_daxpy(size_t m, size_t n, double alpha, const double *x, size_t ldx,
		  double *y, size_t ldy)
{
	if (ldx == m && ldy == m) {
		blas_daxpy(m * n, alpha, x, 1, y, 1);
	} else {
		size_t j;

		for (j = 0; j < n; j++) {
			blas_daxpy(m, alpha, x + j * ldx, 1, y + j * ldy, 1);
		}
	}
}

void matrix_dtrans(size_t m, size_t n, const double *a, size_t lda,
		   double *b, size_t ldb)
{
	size_t j;

	for (j = 0; j < n; j++) {
		blas_dcopy(m, a + j * lda, 1, b + j, ldb);
	}
}

void packed_dgthr(enum blas_uplo uplo, size_t n, const double *a, size_t lda,
		  double *bp)
{
	size_t i;
	double *dst = bp;
	const double *src = a;

	if (uplo == BLAS_LOWER) {
		for (i = 0; i < n; i++) {
			size_t len = n - i;

			memcpy(dst, src + i, len * sizeof(*dst));

			src += lda;
			dst += len;
		}
	} else {
		for (i = 0; i < n; i++) {
			size_t len = i + 1;

			memcpy(dst, src, len * sizeof(*dst));

			src += lda;
			dst += len;
		}
	}
}

void packed_dsctr(enum blas_uplo uplo, size_t n, const double *ap,
		  double *b, size_t ldb)
{
	size_t i;
	double *dst = b;
	const double *src = ap;

	if (uplo == BLAS_LOWER) {
		for (i = 0; i < n; i++) {
			size_t len = n - i;

			memcpy(dst + i, src, len * sizeof(*dst));

			dst += ldb;
			src += len;
		}
	} else {
		for (i = 0; i < n; i++) {
			size_t len = i + 1;

			memcpy(dst, src, len * sizeof(*dst));

			dst += ldb;
			src += len;
		}
	}
}


size_t matrix_drank_lwork(size_t m, size_t n, size_t *liwork)
{
	size_t k = (m <= n) ? m : n;

	if (k == 0) {
		*liwork = 0;
		return 0;
	}

	enum lapack_svdjob jobz = LA_SVD_NOVEC;
	size_t lwork0 = lapack_dgesdd_lwork(jobz, m, n, liwork);
	return lwork0 + k;
}

size_t matrix_drank(size_t m, size_t n, double *a, size_t lda, double svtol,
		    double *work, size_t lwork, ptrdiff_t *iwork)
{
	assert(svtol >= 0);
	size_t k = (m <= n) ? m : n;
	size_t rank = k;

	if (k == 0)
		goto out;

	enum lapack_svdjob jobz = LA_SVD_NOVEC;

	assert(lwork >= k);
	double *s = work;
	work += k;
	lwork -= k;

	double *u = NULL;
	size_t ldu = 1;
	double *vt = NULL;
	size_t ldvt = 1;

	ptrdiff_t info = lapack_dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt,
				       work, lwork, iwork);
	assert(info >= 0); /* no illegal parameter values */
	assert(info == 0); /* algorithm converged */

	while (rank > 0 && s[rank - 1] <= svtol) {
		rank--;
	}
out:
	return rank;
}
