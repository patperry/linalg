#include <string.h>
#include "blas.h"
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
