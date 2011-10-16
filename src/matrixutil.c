#include "blas.h"
#include "matrixutil.h"

void
matrix_dtrans(size_t m, size_t n, const double *a, size_t lda,
	      double *b, size_t ldb)
{
	size_t j;

	for (j = 0; j < n; j++) {
		blas_dcopy(m, a + j * lda, 1, b + j, ldb);
	}
}
