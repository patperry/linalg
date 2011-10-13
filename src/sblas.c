#include <assert.h>
#include "sblas.h"

/* y[indx[i]] += alpha * x[i] for i = 0, ..., nz-1 */
void sblas_daxpyi(size_t nz, double alpha, const double *x,
		  const size_t *indx, double *y)
{
	size_t i;

	for (i = 0; i < nz; i++) {
		y[indx[i]] += alpha * x[i];
	}
}

/* x[0] * y[indx[0]] + ... + x[nz-1] * y[indx[nz-1]] */
double sblas_ddoti(size_t nz, const double *x, const size_t *indx,
		   const double *y)
{
	double dot = 0.0;
	size_t i;

	for (i = 0; i < nz; i++) {
		dot += x[i] * y[indx[i]];
	}

	return dot;
}

/* x[i] := y[indx[i]] for i = 0, ..., nz-1 */
void sblas_dgthr(size_t nz, const double *y, double *x, const size_t *indx)
{
	size_t i;

	for (i = 0; i < nz; i++) {
		x[i] = y[indx[i]];
	}
}

/* x[i] := y[indx[i]]; * y[indx[i]] := 0 for i = 0, ..., nz-1 */
void sblas_dgthrz(size_t nz, double *y, double *x, const size_t *indx)
{
	size_t i;

	for (i = 0; i < nz; i++) {
		x[i] = y[indx[i]];
		y[indx[i]] = 0;
	}
}

/* y[indx[i]] = x[i] for i = 0, ..., nz-1 */
void sblas_dsctr(size_t nz, const double *x, const size_t *indx, double *y)
{
	size_t i;

	for (i = 0; i < nz; i++) {
		y[indx[i]] = x[i];
	}
}

ptrdiff_t sblas_find(size_t nz, const size_t *indx, size_t i)
{
	const size_t *begin = indx, *end = indx + nz, *ptr;

	while (begin < end) {
		ptr = begin + ((end - begin) >> 1);
		if (*ptr < i) {
			begin = ptr + 1;
		} else if (*ptr > i) {
			end = ptr;
		} else {
			return ptr - indx;
		}
	}
	assert(begin == end);

	ptrdiff_t ix = begin - indx;
	return ~ix;
}
