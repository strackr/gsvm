AC_INIT(gsvm, 0.1)
AM_INIT_AUTOMAKE
AC_CONFIG_HEADERS(config.h)

# checks for programs.
AC_PROG_CXX

# checks for libraries.
AC_CHECK_LIB(boost_program_options, main, , AC_MSG_ERROR(boost_program_options required but not found))
AC_CHECK_LIB(gslcblas, main, , AC_MSG_WARN(gslcblas not found))
AC_CHECK_LIB(gsl, main, , AC_MSG_ERROR(gsl required but not found))
AC_CHECK_LIB(gtest, main, , AC_MSG_ERROR(gtest required but not found))

AC_CONFIG_FILES(Makefile gsvm.spec)
AC_OUTPUT
