/**************************************************************************
 * This file is part of gsvm, a Support Vector Machine solver.
 * Copyright (C) 2012 Robert Strack (strackr@vcu.edu), Vojislav Kecman
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **************************************************************************/

#ifndef NUMERIC_H_
#define NUMERIC_H_

// comment out to use double precision
//#define SINGLE_PRECISION
// comment out to disable debug mode
#define DEBUG_MODE

#ifndef DEBUG_MODE
// turning off range check for gsl when production mode enabled
#define GSL_RANGE_CHECK_OFF
#endif

#define HAVE_INLINE

#include <string>
#include <iostream>

#include <map>
#include <vector>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>

#include <gsl/gsl_statistics.h>
#include <gsl/gsl_cdf.h>

#include <float.h>

using namespace std;

#define pow(x, n) gsl_pow_int((x), (n))
#define pow2(x) gsl_pow_2((x))

#define log(x) gsl_sf_log((x))

#ifdef SINGLE_PRECISION
// numeric types for single precision computations
#define VECTOR_TYPE gsl_vector_float
#define MATRIX_TYPE gsl_matrix_float
#define VECTOR_VIEW_TYPE gsl_vector_float_view
#define MATRIX_VIEW_TYPE gsl_matrix_float_view
#define BLAS_PREFIX gsl_blas_s

#define fvector_alloc gsl_vector_float_calloc
#define fvector_free gsl_vector_float_free
#define fvector_set(v, i, f) ((v)->data[(i)] = (f))
#define fvector_get(v, i) ((v)->data[(i)])
#define fvector_set_stride gsl_vector_float_set
#define fvector_get_stride gsl_vector_float_get
#define fvector_set_all gsl_vector_float_set_all
#define fvector_set_zero gsl_vector_float_set_zero
#define fvector_ptr(v) ((v)->data)
#define fvector_ptr_offset gsl_vector_float_ptr
#define fvector_subv gsl_vector_float_subvector
#define fvector_swap gsl_vector_float_swap_elements
#define fvector_add gsl_vector_float_add
#define fvector_sub gsl_vector_float_sub
#define fvector_add_const gsl_vector_float_add_constant
#define fvector_mul gsl_vector_float_mul
#define fvector_mul_const gsl_vector_float_scale
#define fvector_min gsl_vector_float_min_index
#define fvector_max gsl_vector_float_max_index
#define fvector_minmax gsl_vector_float_minmax_index
#define fvector_min_val gsl_vector_float_min
#define fvector_max_val gsl_vector_float_max
#define fvector_minmax_val gsl_vector_float_minmax
#define fvector_cpy gsl_vector_float_memcpy
//#define fvector_dot gsl_blas_sdot
#define fvector_norm gsl_blas_snrm2

#define fvectorv_array gsl_vector_float_view_array

#define fmatrix_alloc gsl_matrix_float_calloc
#define fmatrix_free gsl_matrix_float_free
#define fmatrix_set gsl_matrix_float_set
#define fmatrix_get gsl_matrix_float_get
#define fmatrix_cpy gsl_matrix_float_memcpy
#define fmatrix_sub gsl_matrix_float_submatrix
#define fmatrix_row gsl_matrix_float_row
#define fmatrix_col gsl_matrix_float_column
#define fmatrix_swap_rows gsl_matrix_float_swap_rows
#define fmatrix_gemv gsl_blas_sgemv
#define fmatrix_gemm gsl_blas_sgemm

#define fmatrix_array gsl_matrix_float_view_array

typedef float fvalue;
#define MIN_FVALUE FLT_MIN
#define MAX_FVALUE FLT_MAX

#define fvalue_mean gsl_stats_float_mean
#define fvalue_var gsl_stats_float_var
#define fvalue_sd gsl_stats_float_sd
#define fvalue_max gsl_stats_float_max
#define fvalue_min gsl_stats_float_min

#define exp expf
#define sqrt sqrtf

#else
// numeric types for double precision computations
#define VECTOR_TYPE gsl_vector
#define MATRIX_TYPE gsl_matrix
#define VECTOR_VIEW_TYPE gsl_vector_view
#define MATRIX_VIEW_TYPE gsl_matrix_view
#define BLAS_PREFIX gsl_blas_d

#define fvector_alloc gsl_vector_calloc
#define fvector_free gsl_vector_free
#define fvector_set(v, i, f) ((v)->data[(i)] = (f))
#define fvector_get(v, i) ((v)->data[(i)])
#define fvector_set_stride gsl_vector_set
#define fvector_get_stride gsl_vector_get
#define fvector_set_all gsl_vector_set_all
#define fvector_set_zero gsl_vector_set_zero
#define fvector_ptr(v) ((v)->data)
#define fvector_ptr_offset gsl_vector_ptr
#define fvector_subv gsl_vector_subvector
#define fvector_swap gsl_vector_swap_elements
#define fvector_add gsl_vector_add
#define fvector_sub gsl_vector_sub
#define fvector_add_const gsl_vector_add_constant
#define fvector_mul gsl_vector_mul
#define fvector_mul_const gsl_vector_scale
#define fvector_min gsl_vector_min_index
#define fvector_max gsl_vector_max_index
#define fvector_minmax gsl_vector_minmax_index
#define fvector_min_val gsl_vector_min
#define fvector_max_val gsl_vector_max
#define fvector_minmax_val gsl_vector_minmax
#define fvector_cpy gsl_vector_memcpy
//#define fvector_dot gsl_blas_ddot
#define fvector_norm gsl_blas_dnrm2

#define fvectorv_array gsl_vector_view_array

#define fmatrix_alloc gsl_matrix_calloc
#define fmatrix_free gsl_matrix_free
#define fmatrix_set gsl_matrix_set
#define fmatrix_get gsl_matrix_get
#define fmatrix_cpy gsl_matrix_memcpy
#define fmatrix_sub gsl_matrix_submatrix
#define fmatrix_row gsl_matrix_row
#define fmatrix_col gsl_matrix_column
#define fmatrix_swap_rows gsl_matrix_swap_rows
#define fmatrix_gemv gsl_blas_dgemv
#define fmatrix_gemm gsl_blas_dgemm

#define fmatrix_array gsl_matrix_view_array

typedef double fvalue;
#define MIN_FVALUE DBL_MIN
#define MAX_FVALUE DBL_MAX

#define fvalue_mean gsl_stats_mean
#define fvalue_var gsl_stats_var
#define fvalue_sd gsl_stats_sd
#define fvalue_max gsl_stats_max
#define fvalue_min gsl_stats_min

#define exp exp
#define sqrt sqrt

#endif

#define POS_INF ((fvalue) GSL_POSINF)
#define NEG_INF ((fvalue) GSL_NEGINF)

#define lvector_alloc gsl_vector_ushort_calloc
#define lvector_free gsl_vector_ushort_free
#define lvector_subv gsl_vector_ushort_subvector
#define lvector_cpy gsl_vector_ushort_memcpy
#define lvector_set(v, i, f) ((v)->data[(i)] = (f))
#define lvector_get(v, i) ((v)->data[(i)])
#define lvector_set_stride gsl_vector_ushort_set
#define lvector_get_stride gsl_vector_ushort_get
#define lvector_ptr(v) ((v)->data)
#define lvector_ptr_offset gsl_vector_ushort_ptr
#define lvector_swap gsl_vector_ushort_swap_elements

typedef VECTOR_TYPE fvector;
typedef MATRIX_TYPE fmatrix;
typedef VECTOR_VIEW_TYPE fvectorv;
typedef MATRIX_VIEW_TYPE fmatrixv;
typedef gsl_vector_ushort lvector;
typedef gsl_vector_ushort_view lvectorv;

typedef unsigned int id;
typedef id sample_id;
typedef unsigned short short_id;
typedef short_id feature_id;
typedef short_id label_id;
typedef unsigned int quantity;

#define INVALID_ID ((id) -1)
#define INVALID_SAMPLE_ID INVALID_ID
#define INVALID_SHORT_ID ((short_id) -1)
#define INVALID_FEATURE_ID INVALID_SHORT_ID
#define INVALID_LABEL_ID INVALID_SHORT_ID

#define rng_setup gsl_rng_env_setup
#define rng_alloc gsl_rng_alloc
#define rng_free gsl_rng_free
#define rng_next_int gsl_rng_uniform_int
typedef gsl_rng rng;

typedef sample_id entry_id;

#define INVALID_ENTRY_ID ((entry_id) -1)

ostream& operator <<(ostream& os, const fvector& vector);
ostream& operator <<(ostream& os, const fvectorv& view);

fvalue fvector_sum(fvector *vect);

fvalue fvector_mean(fvector *vect);
fvalue fvector_sd(fvector *vect);

void fvector_dot(fvector *u, fvector *v, fvalue *res);

//void fmatrix_swap_rows(fmatrix *m, size_t i, size_t j);

#endif
