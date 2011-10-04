#ifndef F77_H
#define F77_H

#ifndef F77_FUNC
# define F77_FUNC(f) f ## _
#endif

#ifndef f77int
# ifdef HAVE_BLAS64
#  define f77int long long int
# else
#  define f77int long int
# endif
#endif

#endif /* F77_H */
