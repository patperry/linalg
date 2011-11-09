dnl Process this file with autoconf to produce a configure script.

AC_INIT([linalg], [1.0], [patperry@gmail.com])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])

AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])

m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

# Minimum Autoconf version required.
AC_PREREQ(2.60)

# Where to generate output; srcdir location.
AC_CONFIG_HEADERS([config.h:config.in])
AC_CONFIG_SRCDIR([src/blas.h])

dnl Checks for programs.
AC_PROG_CC
AC_PROG_RANLIB

dnl Checks for libraries
AX_CHECK_BLAS
AX_CHECK_BLAS64
AX_CHECK_LAPACK

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
