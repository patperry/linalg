#ifndef F77BLAS_H
#define F77BLAS_H

#include "f77.h"

#define F77_INT(n) f77int n ## _ = (f77int) n

#define F77_TRANS(t) const char *t ## _ = (t == BLAS_TRANS ? "T" \
		 			   : t == BLAS_CONJTRANS ? "C" : "N")
#define F77_UPLO(u) const char *u ## _ = (u == BLAS_LOWER ? "L" : "U")
#define F77_DIAG(d) const char *d ## _ = (d == BLAS_UNIT ? "U" : "D")
#define F77_SIDE(s) const char *s ## _ = (s == BLAS_RIGHT ? "R" : "L")




extern double F77_FUNC(ddot) (const f77int *n,
			      const double *x,
			      const f77int *incx,
			      const double *y, const f77int *incy);
extern double F77_FUNC(dnrm2) (const f77int *n,
			       const double *x, const f77int *incx);
extern double F77_FUNC(dasum) (const f77int *n,
			       const double *x, const f77int *incx);
extern f77int F77_FUNC(idamax) (const f77int *n,
				const double *x, const f77int *incx);
extern void F77_FUNC(dswap) (const f77int *n,
			     double *x,
			     const f77int *incx, double *y, const f77int *incy);
extern void F77_FUNC(dcopy) (const f77int *n,
			     const double *x,
			     const f77int *incx, double *y, const f77int *incy);
extern void F77_FUNC(daxpy) (const f77int *n,
			     const double *alpha,
			     const double *x,
			     const f77int *incx, double *y, const f77int *incy);
extern void F77_FUNC(drotg) (double *a, double *b, double *c, double *s);
extern void F77_FUNC(drotmg) (double *d1,
			      double *d2,
			      double *b1, const double *b2, double *p);
extern void F77_FUNC(drot) (const f77int *n,
			    double *x,
			    const f77int *incx,
			    double *y,
			    const f77int *incy,
			    const double *c, const double *s);
extern void F77_FUNC(drotm) (const f77int *n,
			     double *x,
			     const f77int *incx,
			     double *y, const f77int *incy, const double *p);
extern void F77_FUNC(dscal) (const f77int *n,
			     const double *alpha,
			     double *x, const f77int *incx);
extern void F77_FUNC(dgemv) (const char *trans,
			     const f77int *m,
			     const f77int *n,
			     const double *alpha,
			     const double *a,
			     const f77int *lda,
			     const double *x,
			     const f77int *incx,
			     const double *beta, double *y, const f77int *incy);
extern void F77_FUNC(dgbmv) (const char *trans,
			     const f77int *m,
			     const f77int *n,
			     const f77int *kl,
			     const f77int *ku,
			     const double *alpha,
			     const double *a,
			     const f77int *lda,
			     const double *x,
			     const f77int *incx,
			     const double *beta, double *y, const f77int *incy);
extern void F77_FUNC(dsbmv) (const char *uplo,
			     const f77int *n,
			     const f77int *k,
			     const double *alpha,
			     const double *a,
			     const f77int *lda,
			     const double *x,
			     const f77int *incx,
			     const double *beta, double *y, const f77int *incy);
extern void F77_FUNC(dspmv) (const char *uplo,
			     const f77int *n,
			     const double *alpha,
			     const double *ap,
			     const double *x,
			     const f77int *incx,
			     const double *beta, double *y, const f77int *incy);
extern void F77_FUNC(dsymv) (const char *uplo,
			     const f77int *n,
			     const double *alpha,
			     const double *a,
			     const f77int *lda,
			     const double *x,
			     const f77int *incx,
			     const double *beta, double *y, const f77int *incy);
extern void F77_FUNC(dtbmv) (const char *uplo,
			     const char *trans,
			     const char *diag,
			     const f77int *n,
			     const f77int *k,
			     const double *a,
			     const f77int *lda, double *x, const f77int *incx);
extern void F77_FUNC(dtpmv) (const char *uplo,
			     const char *trans,
			     const char *diag,
			     const f77int *n,
			     const double *ap, double *x, const f77int *incx);
extern void F77_FUNC(dtrmv) (const char *uplo,
			     const char *trans,
			     const char *diag,
			     const f77int *n,
			     const double *a,
			     const f77int *lda, double *x, const f77int *incx);
extern void F77_FUNC(dtbsv) (const char *uplo,
			     const char *trans,
			     const char *diag,
			     const f77int *n,
			     const f77int *k,
			     const double *a,
			     const f77int *lda, double *x, const f77int *incx);
extern void F77_FUNC(dtpsv) (const char *uplo,
			     const char *trans,
			     const char *diag,
			     const f77int *n,
			     const double *ap, double *x, const f77int *incx);
extern void F77_FUNC(dtrsv) (const char *uplo,
			     const char *trans,
			     const char *diag,
			     const f77int *n,
			     const double *a,
			     const f77int *lda, double *x, const f77int *incx);
extern void F77_FUNC(dger) (const f77int *m,
			    const f77int *n,
			    const double *alpha,
			    const double *x,
			    const f77int *incx,
			    const double *y,
			    const f77int *incy, double *a, const f77int *lda);
extern void F77_FUNC(dsyr) (const char *uplo,
			    const f77int *n,
			    const double *alpha,
			    const double *x,
			    const f77int *incx, double *a, const f77int *lda);
extern void F77_FUNC(dspr) (const char *uplo,
			    const f77int *n,
			    const double *alpha,
			    const double *x, const f77int *incx, double *ap);
extern void F77_FUNC(dsyr2) (const char *uplo,
			     const f77int *n,
			     const double *alpha,
			     const double *x,
			     const f77int *incx,
			     const double *y,
			     const f77int *incy, double *a, const f77int *lda);
extern void F77_FUNC(dspr2) (const char *uplo,
			     const f77int *n,
			     const double *alpha,
			     const double *x,
			     const f77int *incx,
			     const double *y, const f77int *incy, double *ap);
extern void F77_FUNC(dgemm) (const char *transa,
			     const char *transb,
			     const f77int *m,
			     const f77int *n,
			     const f77int *k,
			     const double *alpha,
			     const double *a,
			     const f77int *lda,
			     const double *b,
			     const f77int *ldb,
			     const double *beta, double *c, const f77int *ldc);
extern void F77_FUNC(dtrsm) (const char *side,
			     const char *uplo,
			     const char *transa,
			     const char *diag,
			     const f77int *m,
			     const f77int *n,
			     const double *alpha,
			     const double *a,
			     const f77int *lda, double *b, const f77int *ldb);
extern void F77_FUNC(dtrmm) (const char *side,
			     const char *uplo,
			     const char *transa,
			     const char *diag,
			     const f77int *m,
			     const f77int *n,
			     const double *alpha,
			     const double *a,
			     const f77int *lda, double *b, const f77int *ldb);
extern void F77_FUNC(dsymm) (const char *side,
			     const char *uplo,
			     const f77int *m,
			     const f77int *n,
			     const double *alpha,
			     const double *a,
			     const f77int *lda,
			     const double *b,
			     const f77int *ldb,
			     const double *beta, double *c, const f77int *ldc);
extern void F77_FUNC(dsyrk) (const char *uplo,
			     const char *trans,
			     const f77int *n,
			     const f77int *k,
			     const double *alpha,
			     const double *a,
			     const f77int *lda,
			     const double *beta, double *c, const f77int *ldc);
extern void F77_FUNC(dsyr2k) (const char *uplo,
			      const char *trans,
			      const f77int *n,
			      const f77int *k,
			      const double *alpha,
			      const double *a,
			      const f77int *lda,
			      const double *b,
			      const f77int *ldb,
			      const double *beta, double *c, const f77int *ldc);

#endif /* F77BLAS_H */
