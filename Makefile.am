# Process this file with automake to produce Makefile.in

ACLOCAL_AMFLAGS = -I m4

AM_CPPFLAGS = -Ilib -Isrc

noinst_LIBRARIES = \
	liblinalg.a

liblinalg_a_SOURCES = \
	lib/f77.c \
	lib/f77.h \
	lib/f77blas.h \
	lib/f77lapack.h \
	lib/xerbla.c \
	src/blas.c \
	src/blas.h \
	src/lapack.c \
	src/lapack.h \
	src/matrixutil.c \
	src/matrixutil.h \
	src/sblas.c \
	src/sblas.h

liblinalg_a_LIBS = \
	$LAPACK_LIBS \
	$BLAS_LIBS \
	$LIBS

