#include "sparse.h"

SparseMatrix::SparseMatrix(fvalue *values, feature_id *features, id *offsets,
		size_t size1, size_t size2) :
		values(values),
		features(features),
		offsets(offsets),
		height(size1),
		width(size2) {
}

SparseMatrix::~SparseMatrix() {
	delete [] values;
	delete [] features;
	delete [] offsets;
}
