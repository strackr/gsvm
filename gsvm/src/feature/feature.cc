/**************************************************************************
 * This file is part of gsvm, a Support Vector Machine solver.
 * Copyright (C) 2012 Robert Strack (strackr@vcu.edu), Vojislav Kecman
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **************************************************************************/

#include "feature.h"

template<>
void FeatureProcessor<dfmatrix>::normalize(dfmatrix *samples) {
	for (id col = 0; col < samples->matrix->size2; col++) {
		fvectorv column = fmatrix_col(samples->matrix, col);
		fvalue min;
		fvalue max;
		fvector_minmax_val(&column.vector, &min, &max);
		fvector_add_const(&column.vector, -min);
		fvector_mul_const(&column.vector, 1.0 / (max - min));
	}

//	TODO remove normalization: mean = 0, sigma = 0
//	for (id col = 0; col < samples->matrix->size2; col++) {
//		fvectorv column = fmatrix_col(samples->matrix, col);
//		fvalue mean = fvector_mean(&column.vector);
//		fvector_add_const(&column.vector, -mean);
//		fvalue sd = fvector_sd(&column.vector);
//		fvector_mul_const(&column.vector, 1.0 / sd);
//	}
}

template<>
void FeatureProcessor<sfmatrix>::normalize(sfmatrix *samples) {
	// project features to the range [0, 1]
	vector<fvalue> maxs(samples->width, 0.0);
	for (sample_id smpl = 0; smpl < samples->height; smpl++) {
		id offset = samples->offsets[smpl];
		while (samples->features[offset] != INVALID_FEATURE_ID) {
			feature_id feature = samples->features[offset];
			fvalue value = samples->values[offset];
			maxs[feature] = max(maxs[feature], (fvalue) fabs(value));
			offset++;
		}
	}
	for (sample_id smpl = 0; smpl < samples->height; smpl++) {
		id offset = samples->offsets[smpl];
		while (samples->features[offset] != INVALID_FEATURE_ID) {
			feature_id feature = samples->features[offset];
			samples->values[offset] /= maxs[feature];
			offset++;
		}
	}
}

template<>
void FeatureProcessor<dfmatrix>::randomize(dfmatrix *samples, label_id *labels) {
	IdGenerator gen = Generators::create();
	for (id row = 0; row < samples->height; row++) {
		id rand = gen.nextId(samples->height);
		fmatrix_swap_rows(samples->matrix, row, rand);
		swap(labels[row], labels[rand]);
	}
}

template<>
void FeatureProcessor<sfmatrix>::randomize(sfmatrix *samples, label_id *labels) {
	IdGenerator gen = Generators::create();
	for (id row = 0; row < samples->height; row++) {
		id rand = gen.nextId(samples->height);
		swap(samples->offsets[row], samples->offsets[rand]);
		swap(labels[row], labels[rand]);
	}
}

template<>
void FeatureProcessor<dfmatrix>::reduceDimensionality(dfmatrix *samples, fvalue threshold) {
// TODO implement dimensionality reduction for single precision
#ifndef SINGLE_PRECISION
	size_t dim = samples->matrix->size2;
	fmatrix *cov = fmatrix_alloc(dim, dim);
	fvector *eig = fvector_alloc(dim);
	fmatrix *evec = fmatrix_alloc(dim, dim);

	size_t subSize = min(samples->matrix->size1, (size_t) MAX_EIG_PROB_SIZE);
	fmatrixv samplView = fmatrix_sub(samples->matrix, 0, 0, subSize, dim);

	fmatrix_gemm(CblasTrans, CblasNoTrans, 1.0, &samplView.matrix, &samplView.matrix, 0.0, cov);

    gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc(dim);
    gsl_eigen_symmv(cov, eig, evec, w);
    gsl_eigen_symmv_sort(eig, evec, GSL_EIGEN_SORT_ABS_DESC);

    fvalue thr = threshold * fvector_sum(eig);
    fvalue sum = 0.0;
    size_t newDim = 0;
    while (sum < thr) {
    	sum += fvector_get(eig, newDim++);
    }

    fmatrix *newSamples = fmatrix_alloc(samples->matrix->size1, newDim);
    fmatrixv eigSubview = fmatrix_sub(evec, 0, 0, evec->size1, newDim);
	fmatrix_gemm(CblasNoTrans, CblasNoTrans, 1.0, samples->matrix, &eigSubview.matrix, 0.0, newSamples);

	fmatrix_free(cov);
	fvector_free(eig);
	fmatrix_free(evec);

	fmatrix_free(samples->matrix);
	samples->matrix = newSamples;
#endif
}

template<>
void FeatureProcessor<sfmatrix>::reduceDimensionality(sfmatrix *samples, fvalue threshold) {
	// TODO remove unused features
}
