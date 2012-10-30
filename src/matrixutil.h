#ifndef LINALG_MATRIXUTIL_H
#define LINALG_MATRIXUTIL_H

#include <stddef.h>
#include "blas.h"

void matrix_dzero(size_t m, size_t n, double *a, size_t lda);
void matrix_dscal(size_t m, size_t n, double alpha, double *a, size_t lda);
void matrix_daxpy(size_t m, size_t n, double alpha, const double *x, size_t ldx,
		  double *y, size_t ldy);

void matrix_dtrans(size_t m, size_t n, const double *a, size_t lda,
		   double *b, size_t ldb);

void packed_dgthr(enum blas_uplo uplo, size_t n, const double *a, size_t lda,
		  double *bp);
void packed_dsctr(enum blas_uplo uplo, size_t n, const double *ap,
		  double *b, size_t ldb);


#endif /* LINALG_MATRIXUTIL_H */
