#ifndef DENSE_H_
#define DENSE_H_

#include "numeric.h"

struct DenseMatrix {

	fmatrix *matrix;

	sample_id *forwardMap;
	sample_id *reverseMap;
	sample_id orderRange;

	size_t height;
	size_t width;

	DenseMatrix(fmatrix *matrix);
	~DenseMatrix();

};

typedef DenseMatrix dfmatrix;

#endif
