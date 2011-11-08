#ifndef LINALG_MATRIXUTIL_H
#define LINALG_MATRIXUTIL_H

#include <stddef.h>
#include "blas.h"

void matrix_dzero(size_t m, size_t n, struct dmatrix *a);
void matrix_dscal(size_t m, size_t n, double alpha, struct dmatrix *a);
void matrix_daxpy(size_t m, size_t n, double alpha, const struct dmatrix *x, struct dmatrix *y);

void matrix_dtrans(size_t m, size_t n, const struct dmatrix *a,
		   struct dmatrix *b);

#endif /* LINALG_MATRIXUTIL_H */
