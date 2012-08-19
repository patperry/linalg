#include <assert.h>
#include <stdint.h>
#include "f77.h"
#include "f77blas.h"
#include "f77lapack.h"
#include "lapack.h"

#define MAX(a,b)	((a < b) ?  (b) : (a))
#define MIN(a,b)	((a < b) ?  (a) : (b))

ptrdiff_t lapack_dgesdd(enum lapack_svdjob jobz, size_t m, size_t n,
			double *a, size_t lda, double *s, double *u, size_t ldu,
			double *vt, size_t ldvt, double *work, size_t lwork,
			ptrdiff_t *iwork)
{
	F77_SVDJOB(jobz);
	F77_INT(m);
	F77_INT(n);
	F77_INT(lda);
	F77_INT(ldu);
	F77_INT(ldvt);
	F77_INT(lwork);
	f77int info;

	F77_FUNC(dgesdd) (jobz_, &m_, &n_, a, &lda_, s, u, &ldu_, vt, &ldvt_,
			  work, &lwork_, (f77int *)iwork, &info);
	return (ptrdiff_t)info;
}

size_t lapack_dgesdd_lwork(enum lapack_svdjob jobz, size_t m, size_t n,
			   size_t *liwork)
{
	double *a = NULL;
	size_t lda = MAX(1, m);
	double *s = NULL;
	double *u = NULL;
	size_t ldu = MAX(1, m);
	double *vt = NULL;
	size_t ldvt = MAX(1, n);
	double work = 0;
	ptrdiff_t info;

	info =
	    lapack_dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, &work,
			  SIZE_MAX, NULL);
	assert(info == 0);

	if (liwork)
		*liwork = 8 * MIN(m, n);

	return (size_t)work;
}

void lapack_dlacpy(enum lapack_copyjob uplo, size_t m, size_t n,
		   const double *a, size_t lda, double *b, size_t ldb)
{
	F77_COPYJOB(uplo);
	F77_INT(m);
	F77_INT(n);
	F77_INT(lda);
	F77_INT(ldb);

	F77_FUNC(dlacpy) (uplo_, &m_, &n_, a, &lda_, b, &ldb_);
}

ptrdiff_t lapack_dposv(enum blas_uplo uplo, size_t n, size_t nrhs,
		       double *a, size_t lda, double *b, size_t ldb)
{
	F77_UPLO(uplo);
	F77_INT(n);
	F77_INT(nrhs);
	F77_INT(lda);
	F77_INT(ldb);
	f77int info;

	F77_FUNC(dposv) (uplo_, &n_, &nrhs_, a, &lda_, b, &ldb_, &info);
	return (ptrdiff_t)info;
}

ptrdiff_t lapack_dsyevd(enum lapack_eigjob jobz, enum blas_uplo uplo, size_t n,
			double *a, size_t lda, double *w, double *work,
			size_t lwork, ptrdiff_t *iwork, size_t liwork)
{
	F77_EIGJOB(jobz);
	F77_UPLO(uplo);
	F77_INT(n);
	F77_INT(lda);
	F77_INT(lwork);
	F77_INT(liwork);
	f77int info;

	F77_FUNC(dsyevd) (jobz_, uplo_, &n_, a, &lda_, w, work, &lwork_,
			  (f77int *)iwork, &liwork_, &info);
	return (ptrdiff_t)info;
}

size_t lapack_dsyevd_lwork(enum lapack_eigjob jobz, size_t n, size_t *liwork)
{
	double *a = NULL;
	size_t lda = MAX(1, n);
	double *w = NULL;
	double work = 0;
	f77int iwork = 0;
	ptrdiff_t info;

	info = lapack_dsyevd(jobz, BLAS_UPPER, n, a, lda, w,
			     &work, SIZE_MAX, (ptrdiff_t *)&iwork, SIZE_MAX);
	assert(info == 0);

	if (liwork)
		*liwork = (size_t)iwork;

	return (size_t)work;
}

ptrdiff_t lapack_dsysv(enum blas_uplo uplo, size_t n, size_t nrhs,
		       double *a, size_t lda, ptrdiff_t *ipiv, double *b,
		       size_t ldb, double *work, size_t lwork)
{
	F77_UPLO(uplo);
	F77_INT(n);
	F77_INT(nrhs);
	F77_INT(lda);
	F77_INT(ldb);
	F77_INT(lwork);
	f77int info;

	F77_FUNC(dsysv) (uplo_, &n_, &nrhs_, a, &lda_, (f77int *)ipiv, b,
			 &ldb_, work, &lwork_, &info);
	if (ipiv)
		f77int_unpack(ipiv, n);

	return (ptrdiff_t)info;
}

size_t lapack_dsysv_lwork(size_t n)
{
	double *a = NULL;
	size_t lda = MAX(1, n);
	ptrdiff_t *ipiv = NULL;
	double *b = NULL;
	size_t ldb = MAX(1, n);
	double work = 0;
	ptrdiff_t info;

	info =
	    lapack_dsysv(BLAS_UPPER, n, 1, a, lda, ipiv, b, ldb, &work,
			 SIZE_MAX);
	assert(info == 0);
	return (size_t)work;
}
