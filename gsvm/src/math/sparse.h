#ifndef SPARSE_H_
#define SPARSE_H_

#include "numeric.h"

struct SparseMatrix {

	fvalue *values;
	feature_id *features;

	id *offsets;
	size_t height;
	size_t width;

	SparseMatrix(fvalue *values, feature_id *features, id *offsets, size_t size1, size_t size2);
	~SparseMatrix();

};

typedef SparseMatrix sfmatrix;

#endif
