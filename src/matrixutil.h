#ifndef LINALG_MATRIXUTIL_H
#define LINALG_MATRIXUTIL_H

#include <stddef.h>
#include "blas.h"

#define MATRIX_DRANK_SVTOL0	(1e-8)

void matrix_dzero(size_t m, size_t n, double *a, size_t lda);
void matrix_dscal(size_t m, size_t n, double alpha, double *a, size_t lda);
void matrix_dscalr(size_t m, size_t n, double *s, size_t incs, double *a, size_t lda);
void matrix_dscalc(size_t m, size_t n, double *s, size_t incs, double *a, size_t lda);
void matrix_daxpy(size_t m, size_t n, double alpha, const double *x, size_t ldx,
		  double *y, size_t ldy);

void matrix_dtrans(size_t m, size_t n, const double *a, size_t lda,
		   double *b, size_t ldb);

void packed_dgthr(enum blas_uplo uplo, size_t n, const double *a, size_t lda,
		  double *bp);
void packed_dsctr(enum blas_uplo uplo, size_t n, const double *ap,
		  double *b, size_t ldb);

/* compute the numerical rank using the SVD */
size_t matrix_drank(size_t m, size_t n, double *a, size_t lda, double svtol,
		    double *work, size_t lwork, ptrdiff_t *iwork);
size_t matrix_drank_lwork(size_t m, size_t n, size_t *liwork);


#endif /* LINALG_MATRIXUTIL_H */
