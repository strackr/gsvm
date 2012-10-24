#ifndef CLASSIFY_H_
#define CLASSIFY_H_

#include "kernel.h"

#include <map>

template<typename Kernel, typename Matrix, typename Strategy>
class CrossClassifier {
	RbfKernelEvaluator<Kernel, Matrix> *evaluator;

	fvector *alphas;
	label_id *labels;
	fvector *kernelBuffer;
	fvector *labelBuffer;
	fvector *biasBuffer;

	quantity labelNumber;
	quantity svNumber;

public:
	CrossClassifier(RbfKernelEvaluator<Kernel, Matrix> *evaluator,
			fvector *alphas, label_id *labels, fvector *kernelBuffer,
			quantity labelNumber, quantity svNumber);
	~CrossClassifier();

	label_id classify(sample_id sample);

	quantity getSvNumber();

};

template<typename Kernel, typename Matrix, typename Strategy>
CrossClassifier<Kernel, Matrix, Strategy>::CrossClassifier(
		RbfKernelEvaluator<Kernel, Matrix> *evaluator,
		fvector *alphas, label_id *labels, fvector *kernelBuffer,
		quantity labelNumber, quantity svNumber) :
		evaluator(evaluator),
		alphas(alphas),
		labels(labels),
		kernelBuffer(kernelBuffer),
		labelNumber(labelNumber),
		svNumber(svNumber) {
	labelBuffer = fvector_alloc(labelNumber);

	biasBuffer = fvector_alloc(labelNumber);
	fvalue yyNeg = YY_NEG(labelNumber);
	fvector_set_all(biasBuffer, yyNeg);
	fvalue *aptr = alphas->data;
	fvalue *bptr = biasBuffer->data;
	label_id *lptr = labels;
	fvalue yy = -yyNeg + YY_POS;
	for (sample_id svi = 0; svi < svNumber; svi++) {
		label_id svl = lptr[svi];
		bptr[svl] += yy * aptr[svi];
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
CrossClassifier<Kernel, Matrix, Strategy>::~CrossClassifier() {
	fvector_free(labelBuffer);
	fvector_free(biasBuffer);
}

template<typename Kernel, typename Matrix, typename Strategy>
label_id CrossClassifier<Kernel, Matrix, Strategy>::classify(sample_id sample) {
	evaluator->evalInnerKernel(sample, 0, svNumber, kernelBuffer);
	fvector_mul(kernelBuffer, alphas);

	fvalue yyNeg = YY_NEG(labelNumber);

	fvalue sum = fvector_sum(kernelBuffer);
	fvector_set_all(labelBuffer, yyNeg * sum);
	fvector_add(labelBuffer, biasBuffer);

	label_id *lptr = labels;
	fvalue *lbufptr = labelBuffer->data;
	fvalue *kbufptr = kernelBuffer->data;
	fvalue yy = -yyNeg + YY_POS;
	for (sample_id svi = 0; svi < svNumber; svi++) {
		label_id svl = lptr[svi];
		lbufptr[svl] += yy * kbufptr[svi];
	}
	return fvector_max(labelBuffer);
}

template<typename Kernel, typename Matrix, typename Strategy>
quantity CrossClassifier<Kernel, Matrix, Strategy>::getSvNumber() {
	return svNumber;
}

#endif
