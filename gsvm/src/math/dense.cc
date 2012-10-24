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
