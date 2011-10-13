#ifndef LINALG_SBLAS_H
#define LINALG_SBLAS_H

#include <stddef.h>

/* y[indx[i]] += alpha * x[i] for i = 0, ..., nz-1 */
void sblas_daxpyi(size_t nz, double alpha, const double *x,
		  const size_t *indx, double *y);

/* x[0] * y[indx[0]] + ... + x[nz-1] * y[indx[nz-1]] */
double sblas_ddoti(size_t nz, const double *x, const size_t *indx,
		   const double *y);

/* x[i] := y[indx[i]] for i = 0, ..., nz-1 */
void sblas_dgthr(size_t nz, const double *y, double *x, const size_t *indx);

/* x[i] := y[indx[i]]; * y[indx[i]] := 0 for i = 0, ..., nz-1 */
void sblas_dgthrz(size_t nz, double *y, double *x, const size_t *indx);

/* y[indx[i]] = x[i] for i = 0, ..., nz-1 */
void sblas_dsctr(size_t nz, const double *x, const size_t *indx, double *y);

ptrdiff_t sblas_find(size_t nz, const size_t *indx, size_t i);

#endif /* LINALG_SBLAS_H */
