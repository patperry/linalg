#include <string.h>
#include "blas.h"
#include "matrixutil.h"

void matrix_dzero(size_t m, size_t n, struct dmatrix *a)
{
	if (a->lda == m) {
		memset(a->data, 0, m * n * sizeof(double));
	} else {
		size_t j;
		for (j = 0; j < n; j++) {
			memset(MATRIX_PTR(a, 0, j), 0, m * sizeof(double));
		}
	}
}

void matrix_dscal(size_t m, size_t n, double alpha, struct dmatrix *a)
{
	if (a->lda == m) {
		blas_dscal(m * n, alpha, a->data, 1);
	} else {
		size_t j;

		for (j = 0; j < n; j++) {
			blas_dscal(m, alpha, MATRIX_PTR(a, 0, j), 1);
		}
	}
}

void matrix_daxpy(size_t m, size_t n, double alpha, const struct dmatrix *x,
		  struct dmatrix *y)
{
	if (x->lda == m && y->lda == m) {
		blas_daxpy(m * n, alpha, x->data, 1, y->data, 1);
	} else {
		size_t j;

		for (j = 0; j < n; j++) {
			blas_daxpy(m, alpha, MATRIX_COL(x, j), 1,
				   MATRIX_COL(y, j), 1);
		}
	}
}

void matrix_dtrans(size_t m, size_t n, const struct dmatrix *a,
		   struct dmatrix *b)
{
	size_t ldb = b->lda;
	size_t j;

	for (j = 0; j < n; j++) {
		blas_dcopy(m, MATRIX_PTR(a, 0, j), 1, MATRIX_PTR(b, j, 0), ldb);
	}
}
