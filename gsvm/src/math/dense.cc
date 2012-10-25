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

#include "dense.h"

DenseMatrix::DenseMatrix(fmatrix *matrix) :
		matrix(matrix) {
	forwardMap = new sample_id[matrix->size1];
	reverseMap = new sample_id[matrix->size1];
	for (sample_id i = 0; i < matrix->size1; i++) {
		forwardMap[i] = i;
		reverseMap[i] = i;
	}
	orderRange = matrix->size1;
	height = matrix->size1;
	width = matrix->size2;
}

DenseMatrix::~DenseMatrix() {
	fmatrix_free(matrix);
	delete [] forwardMap;
	delete [] reverseMap;
}
