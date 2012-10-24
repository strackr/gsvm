#ifndef MATRIX_H_
#define MATRIX_H_

#include <algorithm>

#include "numeric.h"
#include "sparse.h"
#include "dense.h"

template<typename Matrix>
class EvaluatorWorkspace {

};


template<>
class EvaluatorWorkspace<dfmatrix> {
public:
	fvectorv urow;
	fvectorv vrow;

	fmatrixv matrixView;
	fvectorv rowView;
	fvectorv bufferView;
	fvectorv x2View;

	fvalue *matrixData;
	size_t matrixTda;

	sample_id *forwardMap;
	sample_id *reverseMap;

	EvaluatorWorkspace(dfmatrix *matrix);
	~EvaluatorWorkspace();

};


template<>
class EvaluatorWorkspace<sfmatrix> {
public:
	fvalue *buffer;

	EvaluatorWorkspace(sfmatrix *matrix);
	~EvaluatorWorkspace();

};


// TODO implement matrix algorithms for sparse and dense matrices
template<typename Matrix>
class MatrixEvaluator {

	Matrix *matrix;
	fvalue *x2;
	EvaluatorWorkspace<Matrix> workspace;

protected:
	quantity size(Matrix *matrix);
	quantity dim(Matrix *matrix);

	fvalue norm2(sample_id v);

public:
	MatrixEvaluator(Matrix *matrix);
	~MatrixEvaluator();

	fvalue dot(sample_id u, sample_id v);

	void dist(sample_id id, sample_id rangeFrom, sample_id rangeTo, fvector *buffer);
	fvalue dist(sample_id u, sample_id v);

	void swapSamples(sample_id u, sample_id v);

};

template<typename Matrix>
MatrixEvaluator<Matrix>::MatrixEvaluator(Matrix *matrix) :
		matrix(matrix),
		workspace(matrix) {
	quantity length = size(matrix);
	x2 = new fvalue[length];
	for (sample_id id = 0; id < length; id++) {
		x2[id] = norm2(id);
	}
}

template<typename Matrix>
MatrixEvaluator<Matrix>::~MatrixEvaluator() {
	delete x2;
}

template<typename Matrix>
fvalue MatrixEvaluator<Matrix>::dist(sample_id u, sample_id v) {
	return x2[u] + x2[v] - 2 * dot(u, v);
}

template<typename Matrix>
inline quantity MatrixEvaluator<Matrix>::size(Matrix *matrix) {
	return matrix->height;
}

template<typename Matrix>
inline quantity MatrixEvaluator<Matrix>::dim(Matrix *matrix) {
	return matrix->width;
}

#endif
