#include <assert.h>
#include "f77.h"

union fcint {
	ptrdiff_t c;
	f77int f[2];
};

void f77int_pack(ptrdiff_t *base, size_t n)
{
	if (sizeof(ptrdiff_t) == sizeof(f77int))
		return;

	assert(sizeof(ptrdiff_t) == 2 * sizeof(f77int));

	union fcint *arr = (union fcint *)base;
	size_t i, j;

	if (!n)
		return;

	for (i = 0, j = 0; i < n - 1; i += 2, j++) {
		arr[j].f[0] = (f77int)arr[i].c;
		arr[j].f[1] = (f77int)arr[i + 1].c;
	}

	if (i == n - 1) {
		arr[j].f[0] = (f77int)arr[i].c;
	}
}

void f77int_unpack(ptrdiff_t *base, size_t n)
{
	if (sizeof(ptrdiff_t) == sizeof(f77int))
		return;

	assert(sizeof(ptrdiff_t) == 2 * sizeof(f77int));

	union fcint *arr = (union fcint *)base;

	if (n % 2) {
		arr[n - 1].c = (ptrdiff_t)arr[n / 2].f[0];
		n--;
	}

	for (; n > 0; n -= 2) {
		arr[n - 1].c = (ptrdiff_t)arr[(n - 1) / 2].f[1];
		arr[n - 2].c = (ptrdiff_t)arr[(n - 1) / 2].f[0];
	}
}
