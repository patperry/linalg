#include <assert.h>
#include <stdint.h>
#include "f77.h"
#include "f77blas.h"
#include "f77lapack.h"
#include "lapack.h"

#define MAX(a,b)	((a < b) ?  (b) : (a))
#define MIN(a,b)	((a < b) ?  (a) : (b))

ptrdiff_t lapack_dgesdd(enum lapack_svdjob jobz, size_t m, size_t n,
			struct dmatrix *a, double *s, struct dmatrix *u,
			struct dmatrix *vt, double *work, size_t lwork,
			ptrdiff_t *iwork)
{
	F77_SVDJOB(jobz);
	F77_INT(m);
	F77_INT(n);
	F77_DMATRIX(a);
	F77_DMATRIX(u);
	F77_DMATRIX(vt);
	F77_INT(lwork);
	f77int info;

	F77_FUNC(dgesdd) (jobz_, &m_, &n_, a_, &lda_, s, u_, &ldu_, vt_, &ldvt_,
			  work, &lwork_, (f77int *)iwork, &info);
	return (ptrdiff_t)info;
}

size_t lapack_dgesdd_lwork(enum lapack_svdjob jobz, size_t m, size_t n,
			   size_t *liwork)
{
	double work = 0;
	ptrdiff_t info;

	struct dmatrix a = { NULL, MAX(1, m) };
	struct dmatrix u = { NULL, MAX(1, m) };
	struct dmatrix vt = { NULL, MAX(1, n) };

	info =
	    lapack_dgesdd(jobz, m, n, &a, NULL, &u, &vt, &work, SIZE_MAX, NULL);
	assert(info == 0);

	if (liwork)
		*liwork = 8 * MIN(m, n);

	return (size_t)work;
}

void lapack_dlacpy(enum lapack_copyjob uplo, size_t m, size_t n,
		   const struct dmatrix *a, struct dmatrix *b)
{
	F77_COPYJOB(uplo);
	F77_INT(m);
	F77_INT(n);
	F77_DMATRIX(a);
	F77_DMATRIX(b);

	F77_FUNC(dlacpy) (uplo_, &m_, &n_, a_, &lda_, b_, &ldb_);
}

ptrdiff_t lapack_dposv(enum blas_uplo uplo, size_t n, size_t nrhs,
		       struct dmatrix *a, struct dmatrix *b)
{
	F77_UPLO(uplo);
	F77_INT(n);
	F77_INT(nrhs);
	F77_DMATRIX(a);
	F77_DMATRIX(b);
	f77int info;

	F77_FUNC(dposv) (uplo_, &n_, &nrhs_, a_, &lda_, b_, &ldb_, &info);
	return (ptrdiff_t)info;
}

ptrdiff_t lapack_dsyevd(enum lapack_eigjob jobz, enum blas_uplo uplo, size_t n,
			struct dmatrix *a, double *w, double *work,
			size_t lwork, ptrdiff_t *iwork, size_t liwork)
{
	F77_EIGJOB(jobz);
	F77_UPLO(uplo);
	F77_INT(n);
	F77_DMATRIX(a);
	F77_INT(lwork);
	F77_INT(liwork);
	f77int info;

	F77_FUNC(dsyevd) (jobz_, uplo_, &n_, a_, &lda_, w, work, &lwork_,
			  (f77int *)iwork, &liwork_, &info);
	return (ptrdiff_t)info;
}

size_t lapack_dsyevd_lwork(enum lapack_eigjob jobz, size_t n, size_t *liwork)
{
	double work = 0;
	f77int iwork = 0;
	ptrdiff_t info;

	struct dmatrix a = { NULL, MAX(1, n) };

	info = lapack_dsyevd(jobz, BLAS_UPPER, n, &a, NULL,
			     &work, SIZE_MAX, (ptrdiff_t *)&iwork, SIZE_MAX);
	assert(info == 0);

	if (liwork)
		*liwork = (size_t)iwork;

	return (size_t)work;
}

ptrdiff_t lapack_dsysv(enum blas_uplo uplo, size_t n, size_t nrhs,
		       struct dmatrix *a, ptrdiff_t *ipiv, struct dmatrix *b,
		       double *work, size_t lwork)
{
	F77_UPLO(uplo);
	F77_INT(n);
	F77_INT(nrhs);
	F77_DMATRIX(a);
	F77_DMATRIX(b);
	F77_INT(lwork);
	f77int info;

	F77_FUNC(dsysv) (uplo_, &n_, &nrhs_, a_, &lda_, (f77int *)ipiv, b_,
			 &ldb_, work, &lwork_, &info);
	if (ipiv)
		f77int_unpack(ipiv, n);

	return (ptrdiff_t)info;
}

size_t lapack_dsysv_lwork(size_t n)
{
	double work = 0;
	ptrdiff_t info;

	struct dmatrix a = { NULL, MAX(1, n) };
	struct dmatrix b = { NULL, MAX(1, n) };

	info = lapack_dsysv(BLAS_UPPER, n, 1, &a, NULL, &b, &work, SIZE_MAX);
	assert(info == 0);
	return (size_t)work;
}
