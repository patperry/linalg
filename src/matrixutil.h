#ifndef LINALG_MATRIXUTIL_H
#define LINALG_MATRIXUTIL_H

#include <stddef.h>

void matrix_dtrans(size_t m, size_t n, const double *a, size_t lda,
		   double *b, size_t ldb);

#endif /* LINALG_MATRIXUTIL_H */
