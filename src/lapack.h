#ifndef LAPACK_H
#define LAPACK_H

#include <stddef.h>
#include "blas.h"

enum lapack_copyjob {
	LA_COPY_UPPER = BLAS_UPPER,
	LA_COPY_LOWER = BLAS_LOWER,
	LA_COPY_ALL
};

enum lapack_eigjob {
	LA_EIG_NOVEC,
	LA_EIG_VEC
};

enum lapack_svdjob {
	LA_SVD_NOVEC,
	LA_SVD_OVER,
	LA_SVD_SEP,
	LA_SVD_ALL
};

ptrdiff_t lapack_dgesdd(enum lapack_svdjob jobz, size_t m, size_t n,
			struct dmatrix *a, double *s, struct dmatrix *u,
			struct dmatrix *vt, double *work, size_t lwork,
			ptrdiff_t *iwork);

size_t lapack_dgesdd_lwork(enum lapack_svdjob jobz, size_t m, size_t n,
			   size_t *liwork);

void lapack_dlacpy(enum lapack_copyjob uplo, size_t m, size_t n,
		   const struct dmatrix *a, struct dmatrix *b);

ptrdiff_t lapack_dposv(enum blas_uplo uplo, size_t n, size_t nrhs,
		       struct dmatrix *a, struct dmatrix *b);

ptrdiff_t lapack_dsyevd(enum lapack_eigjob jobz, enum blas_uplo uplo, size_t n,
			struct dmatrix *a, double *w, double *work,
			size_t lwork, ptrdiff_t *iwork, size_t liwork);
size_t lapack_dsyevd_lwork(enum lapack_eigjob jobz, size_t n, size_t *liwork);

ptrdiff_t lapack_dsysv(enum blas_uplo uplo, size_t n, size_t nrhs,
		       struct dmatrix *a, ptrdiff_t *ipiv, struct dmatrix *b,
		       double *work, size_t lwork);

size_t lapack_dsysv_lwork(size_t n);

#endif /* LAPACK_H */
