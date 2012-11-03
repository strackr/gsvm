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

#ifndef CLASSIFY_H_
#define CLASSIFY_H_

#include "kernel.h"

#include <map>


template<typename Kernel, typename Matrix>
class Classifier {

public:
	virtual label_id classify(sample_id sample) = 0;
	virtual quantity getSvNumber() = 0;

	virtual ~Classifier() {};

};


template<typename Kernel, typename Matrix>
class UniversalClassifier: public Classifier<Kernel, Matrix> {
	RbfKernelEvaluator<Kernel, Matrix> *evaluator;

	fvector *alphas;
	label_id *labels;
	fvector *kernelBuffer;
	fvector *labelBuffer;
	fvector *biasBuffer;

	quantity labelNumber;
	quantity svNumber;

public:
	UniversalClassifier(RbfKernelEvaluator<Kernel, Matrix> *evaluator,
			fvector *alphas, label_id *labels, fvector *kernelBuffer,
			quantity labelNumber, quantity svNumber);
	virtual ~UniversalClassifier();

	label_id classify(sample_id sample);

	quantity getSvNumber();

};

template<typename Kernel, typename Matrix>
UniversalClassifier<Kernel, Matrix>::UniversalClassifier(
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

template<typename Kernel, typename Matrix>
UniversalClassifier<Kernel, Matrix>::~UniversalClassifier() {
	fvector_free(labelBuffer);
	fvector_free(biasBuffer);
}

template<typename Kernel, typename Matrix>
label_id UniversalClassifier<Kernel, Matrix>::classify(sample_id sample) {
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

template<typename Kernel, typename Matrix>
quantity UniversalClassifier<Kernel, Matrix>::getSvNumber() {
	return svNumber;
}

#endif
