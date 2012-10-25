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

#include "matrix.h"

EvaluatorWorkspace<dfmatrix>::EvaluatorWorkspace(dfmatrix *matrix) {
	urow = fmatrix_row(matrix->matrix, 0);
	vrow = fmatrix_row(matrix->matrix, 0);

	matrixView = fmatrix_sub(matrix->matrix, 0, 0, 1, matrix->matrix->size2);
	rowView = fmatrix_row(matrix->matrix, 0);
	bufferView = rowView;
	x2View = rowView;

	matrixData = matrix->matrix->data;
	matrixTda = matrix->matrix->tda;

	forwardMap = matrix->forwardMap;
	reverseMap = matrix->reverseMap;
}

EvaluatorWorkspace<sfmatrix>::EvaluatorWorkspace(sfmatrix *matrix) {
	buffer = new fvalue[matrix->width];
	for (size_t i = 0; i < matrix->width; i++) {
		buffer[i] = 0;
	}
}

EvaluatorWorkspace<dfmatrix>::~EvaluatorWorkspace() {
}

EvaluatorWorkspace<sfmatrix>::~EvaluatorWorkspace() {
	delete [] buffer;
}

template<>
fvalue MatrixEvaluator<dfmatrix>::dot(sample_id u, sample_id v) {
	sample_id realu = workspace.forwardMap[u];
	sample_id realv = workspace.forwardMap[v];

	workspace.urow.vector.data = workspace.matrixData + realu * workspace.matrixTda;
	workspace.vrow.vector.data = workspace.matrixData + realv * workspace.matrixTda;
	fvalue result;
	fvector_dot(&workspace.urow.vector, &workspace.vrow.vector, &result);
	return result;

	// TODO remove the following code (version without workspace)
	//	fvectorv urow = fmatrix_row(matrix, u);
	//	fvectorv vrow = fmatrix_row(matrix, v);
	//	fvalue result;
	//	fvector_dot(&urow.vector, &vrow.vector, &result);
	//	return result;
}

template<>
fvalue MatrixEvaluator<sfmatrix>::dot(sample_id u, sample_id v) {
	id uoffset = matrix->offsets[u];
	feature_id *iuptr = matrix->features + uoffset;
	fvalue *fuptr = matrix->values + uoffset;

	id voffset = matrix->offsets[v];
	feature_id *ivptr = matrix->features + voffset;
	fvalue *fvptr = matrix->values + voffset;

	fvalue sum = 0.0;
	while (*iuptr != INVALID_FEATURE_ID && *ivptr != INVALID_FEATURE_ID) {
		while (*iuptr < *ivptr) {
			iuptr++;
			fuptr++;
		}
		if (*iuptr == *ivptr) {
			sum += *fuptr * *fvptr;
			iuptr++;
			fuptr++;
		}
		swap(iuptr, ivptr);
		swap(fuptr, fvptr);
	}
	return sum;
}

template<>
fvalue MatrixEvaluator<dfmatrix>::norm2(sample_id u) {
	sample_id realu = workspace.forwardMap[u];
	fvectorv row = fmatrix_row(matrix->matrix, realu);
	return pow2(fvector_norm(&row.vector));
}

template<>
fvalue MatrixEvaluator<sfmatrix>::norm2(sample_id u) {
	fvalue sum = 0.0;
	id offset = matrix->offsets[u];
	feature_id *iptr = matrix->features + offset;
	fvalue *fptr = matrix->values + offset;
	while (*iptr++ != INVALID_FEATURE_ID) {
		sum += pow2(*fptr++);
	}
	return sum;
}

template<>
void MatrixEvaluator<dfmatrix>::dist(sample_id id,
		sample_id rangeFrom, sample_id rangeTo, fvector *buffer) {
	while (matrix->orderRange < rangeTo) {
		sample_id rev = workspace.reverseMap[matrix->orderRange];
		sample_id fwd = workspace.forwardMap[matrix->orderRange];
		fmatrix_swap_rows(matrix->matrix, fwd, matrix->orderRange);
		swap(workspace.reverseMap[fwd], workspace.reverseMap[matrix->orderRange]);
		swap(workspace.forwardMap[rev], workspace.forwardMap[matrix->orderRange]);
		matrix->orderRange++;
	}

	sample_id realid = workspace.forwardMap[id];
	sample_id range = rangeTo - rangeFrom;
	workspace.matrixView.matrix.data = workspace.matrixData + rangeFrom * workspace.matrixTda;
	workspace.matrixView.matrix.size1 = range;
	workspace.rowView.vector.data = workspace.matrixData + realid * workspace.matrixTda;
	workspace.bufferView.vector.data = buffer->data + rangeFrom;
	workspace.bufferView.vector.size = range;
	workspace.x2View.vector.data = x2 + rangeFrom;
	workspace.x2View.vector.size = range;

	fvalue id2 = x2[id];
	fvalue *bptr = workspace.bufferView.vector.data;
	fvalue *xptf = workspace.x2View.vector.data;
	for (sample_id i = 0; i < range; i++) {
		*bptr++ = id2 + *xptf++;
	}
	fmatrix_gemv(CblasNoTrans, -2.0, &workspace.matrixView.matrix,
			&workspace.rowView.vector, 1.0, &workspace.bufferView.vector);

	// blas based distance computation
	//	fvector_cpy(&workspace.bufferView.vector, &workspace.x2View.vector);
	//	fmatrix_gemv(CblasNoTrans, -2.0, &workspace.matrixView.matrix,
	//			&workspace.rowView.vector, 1.0, &workspace.bufferView.vector);
	//	fvector_add_const(&workspace.bufferView.vector, fvector_get(x2, id));

	// TODO remove the following code (version without workspace)
	//	fmatrixv samplesView = fmatrix_sub(matrix, rangeFrom, 0, range, matrix->size2);
	//	fvectorv sampleRowView = fmatrix_row(matrix, id);
	//	fvectorv bufferView = fvector_subv(buffer, rangeFrom, range);
	//	fvectorv x2View = fvector_subv(x2, rangeFrom, range);
	//	fvector_cpy(&bufferView.vector, &x2View.vector);
	//	fmatrix_gemv(CblasNoTrans, -2.0, &samplesView.matrix,
	//			&sampleRowView.vector, 1.0, &bufferView.vector);
	//	fvector_add_const(&bufferView.vector, fvector_get(x2, id));
}

template<>
void MatrixEvaluator<sfmatrix>::dist(sample_id v,
		sample_id rangeFrom, sample_id rangeTo, fvector *buffer) {
	// setup workspace
	id offset = matrix->offsets[v];
	feature_id *iptr = matrix->features + offset;
	fvalue *fptr = matrix->values + offset;
	while (*iptr != INVALID_FEATURE_ID) {
		workspace.buffer[*iptr++] = *fptr++;
	}

	// calculate dists
	fvalue *x2ptr = x2;
	fvalue v2 = x2ptr[v];
	fvalue *fbuffer = buffer->data;
	for (sample_id offst = rangeFrom; offst < rangeTo; offst++) {
		id coffset = matrix->offsets[offst];
		feature_id *icptr = matrix->features + coffset;
		fvalue *fcptr = matrix->values + coffset;
		fvalue sum = 0.0;
		while (*icptr != INVALID_FEATURE_ID) {
			sum += *fcptr++ * workspace.buffer[*icptr++];
		}
		fbuffer[offst] = x2ptr[offst] + v2 - 2.0 * sum;
	}

	// clear workspace
	iptr = matrix->features + offset;
	while (*iptr != INVALID_FEATURE_ID) {
		workspace.buffer[*iptr++] = 0.0;
	}
}

template<>
void MatrixEvaluator<dfmatrix>::swapSamples(sample_id u, sample_id v) {
//	fmatrix_swap_rows(matrix->matrix, u, v);
	swap(x2[u], x2[v]);

	sample_id revu = matrix->forwardMap[u];
	sample_id revv = matrix->forwardMap[v];
	swap(matrix->reverseMap[revu], matrix->reverseMap[revv]);
	swap(matrix->forwardMap[u], matrix->forwardMap[v]);

	sample_id minuv = min(u, v);
	if (minuv < matrix->orderRange) {
		matrix->orderRange = minuv;
	}
}

template<>
void MatrixEvaluator<sfmatrix>::swapSamples(sample_id u, sample_id v) {
	swap(matrix->offsets[u], matrix->offsets[v]);
	swap(x2[u], x2[v]);
}
