#ifndef KERNEL_H_
#define KERNEL_H_

#include "../math/numeric.h"
#include "../math/matrix.h"

#define YY_POS 1.0
#define YY_NEG(cl) (-1.0 / (cl - 1))

struct GaussKernel {
	fvalue ngamma;

	GaussKernel(fvalue gamma);

	fvalue eval(fvalue dist2);

};

inline fvalue GaussKernel::eval(fvalue dist2) {
	return exp(ngamma * dist2);
}


template<class Kernel, class Matrix>
class RbfKernelEvaluator {

private:
	Matrix *samples;
	label_id *labels;
	fvalue c;

	fvalue yyNeg;
	fvalue d1dc;

	fvalue tau;

protected:
	Kernel params;
	MatrixEvaluator<Matrix> eval;

	fvalue rbf(fvalue dist2);

public:
	RbfKernelEvaluator(Matrix *samples, label_id *labels, quantity classNumber, fvalue c, Kernel &params);
	~RbfKernelEvaluator();

	fvalue evalInnerKernel(sample_id uid, sample_id vid);
	void evalInnerKernel(sample_id id, sample_id rangeFrom, sample_id rangeTo, fvector *result);

	fvalue evalKernel(sample_id uid, sample_id vid);
	void evalKernel(sample_id id, sample_id rangeFrom, sample_id rangeTo, fvector *result);
	fvalue getKernelTau();

	void swapSamples(sample_id uid, sample_id vid);
	void setKernelParams(fvalue c, Kernel &params);

	Kernel getParams();
	fvalue getC();

};

template<class Kernel, class Matrix>
RbfKernelEvaluator<Kernel, Matrix>::RbfKernelEvaluator(
		Matrix *samples, label_id *labels, quantity classNumber,
		fvalue c, Kernel &params) :
		samples(samples),
		labels(labels),
		c(c),
		params(params),
		eval(samples) {
	yyNeg = -1.0 / (classNumber - 1);
	d1dc = 1.0 / c;
	tau = 2.0 + 1.0 / c;
}

template<class Kernel, class Matrix>
RbfKernelEvaluator<Kernel, Matrix>::~RbfKernelEvaluator() {
}

template<class Kernel, class Matrix>
inline fvalue RbfKernelEvaluator<Kernel, Matrix>::rbf(fvalue dist2) {
	return params.eval(dist2);
}

template<class Kernel, class Matrix>
inline fvalue RbfKernelEvaluator<Kernel, Matrix>::evalInnerKernel(
		sample_id uid, sample_id vid) {
	return rbf(eval.dist(uid, vid));
}

template<class Kernel, class Matrix>
void RbfKernelEvaluator<Kernel, Matrix>::evalInnerKernel(sample_id id,
		sample_id rangeFrom, sample_id rangeTo, fvector *result) {
	eval.dist(id, rangeFrom, rangeTo, result);

	fvalue *ptr = fvector_ptr(result);
	for (sample_id iid = rangeFrom; iid < rangeTo; iid++) {
		ptr[iid] = rbf(ptr[iid]);
	}
}

template<class Kernel, class Matrix>
fvalue RbfKernelEvaluator<Kernel, Matrix>::evalKernel(
		sample_id uid, sample_id vid) {
	label_id ulabel = labels[uid];
	label_id vlabel = labels[vid];
	fvalue result;
	if (ulabel == vlabel) {
		if (uid == vid) {
			result = tau;
		} else {
			result = evalInnerKernel(uid, vid) + 1.0;
		}
	} else {
		result = yyNeg * (evalInnerKernel(uid, vid) + 1.0);
	}
	return result;
}

template<class Kernel, class Matrix>
void RbfKernelEvaluator<Kernel, Matrix>::evalKernel(sample_id id,
		sample_id rangeFrom, sample_id rangeTo, fvector *result) {
	eval.dist(id, rangeFrom, rangeTo, result);

	fvalue *rptr = fvector_ptr(result);
	label_id *lptr = labels;
	label_id label = lptr[id];
	for (sample_id iid = rangeFrom; iid < rangeTo; iid++) {
		fvalue yy = lptr[iid] == label ? 1.0 : yyNeg;
		rptr[iid] = yy * (rbf(rptr[iid]) + 1.0);
	}
	if (id >= rangeFrom && id < rangeTo) {
		rptr[id] += d1dc;
	}
}

template<class Kernel, class Matrix>
inline fvalue RbfKernelEvaluator<Kernel, Matrix>::getKernelTau() {
	return tau;
}

template<class Kernel, class Matrix>
inline void RbfKernelEvaluator<Kernel, Matrix>::swapSamples(sample_id uid, sample_id vid) {
	swap(labels[uid], labels[vid]);
	eval.swapSamples(uid, vid);
}

template<class Kernel, class Matrix>
void RbfKernelEvaluator<Kernel, Matrix>::setKernelParams(fvalue c, Kernel &params) {
	this->c = c;
	this->params = params;

	d1dc = 1.0 / c;
	tau = 2.0 + 1.0 / c;
}

template<class Kernel, class Matrix>
inline Kernel RbfKernelEvaluator<Kernel, Matrix>::getParams() {
	return params;
}

template<class Kernel, class Matrix>
inline fvalue RbfKernelEvaluator<Kernel, Matrix>::getC() {
	return c;
}

#endif
