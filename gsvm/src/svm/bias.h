/**************************************************************************
 * This file is part of gsvm, a Support Vector Machine solver.
 * Copyright (C) 2012 Robert Strack (strackr@vcu.edu), Vojislav Kecman
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **************************************************************************/

#ifndef BIAS_H_
#define BIAS_H_

#include "kernel.h"
#include "params.h"
#include "../math/numeric.h"

#define DEFAULT_EPSILON 0.001

#define EXPECTED_SV_NUMBER 15000

template<typename Kernel, typename Matrix>
class BiasEvaluationStrategy {

public:
	virtual vector<fvalue> getBias(vector<label_id>& labels, vector<fvalue>& alphas,
			quantity labelNumber, quantity sampleNumber, fvalue rho, fvalue c) = 0;
	virtual fvalue getBinaryBias(vector<label_id>& labels, vector<fvalue>& alphas,
			quantity sampleNumber, label_id label, fvalue rho, fvalue c) = 0;

	virtual ~BiasEvaluationStrategy();

};

template<typename Kernel, typename Matrix>
BiasEvaluationStrategy<Kernel, Matrix>::~BiasEvaluationStrategy() {
}



template<typename Kernel, typename Matrix>
class TeoreticBiasStrategy: public BiasEvaluationStrategy<Kernel, Matrix> {

public:
	virtual vector<fvalue> getBias(vector<label_id>& labels, vector<fvalue>& alphas,
			quantity labelNumber, quantity sampleNumber, fvalue rho, fvalue c);
	virtual fvalue getBinaryBias(vector<label_id>& labels, vector<fvalue>& alphas,
			quantity sampleNumber, label_id label, fvalue rho, fvalue c);

	virtual ~TeoreticBiasStrategy();

};

template<typename Kernel, typename Matrix>
TeoreticBiasStrategy<Kernel, Matrix>::~TeoreticBiasStrategy() {
}

template<typename Kernel, typename Matrix>
vector<fvalue> TeoreticBiasStrategy<Kernel, Matrix>::getBias(
		vector<label_id>& labels, vector<fvalue>& alphas,
		quantity labelNumber, quantity sampleNumber, fvalue rho, fvalue c) {
	vector<fvalue> bias(labelNumber, YY_NEG(labelNumber));

	fvalue yy = -YY_NEG(labelNumber) + YY_POS;
	for (sample_id v = 0; v < sampleNumber; v++) {
		label_id label = labels[v];
		bias[label] += yy * alphas[v];
	}
	return bias;
}

template<typename Kernel, typename Matrix>
fvalue TeoreticBiasStrategy<Kernel, Matrix>::getBinaryBias(
		vector<label_id>& labels, vector<fvalue>& alphas,
		quantity sampleNumber, label_id label, fvalue rho, fvalue c) {
	fvalue bias = 0;
	fvalue mult[] = {-1.0, 1.0};
	for (sample_id v = 0; v < sampleNumber; v++) {
		bias += alphas[v] * mult[labels[v] == label];
	}
	return bias;
}


template<typename Kernel, typename Matrix>
class AverageBiasStrategy: public BiasEvaluationStrategy<Kernel, Matrix> {

	RbfKernelEvaluator<Kernel, Matrix> *evaluator;

public:
	AverageBiasStrategy(RbfKernelEvaluator<Kernel, Matrix> *evaluator);

	virtual vector<fvalue> getBias(vector<label_id>& labels, vector<fvalue>& alphas,
			quantity labelNumber, quantity sampleNumber, fvalue rho, fvalue c);
	virtual fvalue getBinaryBias(vector<label_id>& labels, vector<fvalue>& alphas,
			quantity sampleNumber, label_id label, fvalue rho, fvalue c);

	virtual ~AverageBiasStrategy();

};

template<typename Kernel, typename Matrix>
AverageBiasStrategy<Kernel, Matrix>::AverageBiasStrategy(
		RbfKernelEvaluator<Kernel, Matrix>* evaluator) :
		evaluator(evaluator) {
}

template<typename Kernel, typename Matrix>
AverageBiasStrategy<Kernel, Matrix>::~AverageBiasStrategy() {
}

template<typename Kernel, typename Matrix>
vector<fvalue> AverageBiasStrategy<Kernel, Matrix>::getBias(
		vector<label_id>& labels, vector<fvalue>& alphas,
		quantity labelNumber, quantity sampleNumber, fvalue rho, fvalue c) {
	vector<fvalue> bias(labelNumber, 0.0);
	vector<quantity> count(labelNumber, 0);

	fvector* kernelBuffer = fvector_alloc(sampleNumber);

	for (id v = 0; v < sampleNumber; v++) {
		if (alphas[v] > 0.0) {
			label_id label = labels[v];
			fvalue value = rho - alphas[v] / c;

			// TODO use cache instead
			evaluator->evalInnerKernel(v, 0, sampleNumber, kernelBuffer);
			fvalue* kernelValues = kernelBuffer->data;
			for (id u = 0; u < sampleNumber; u++) {
				fvalue yy = (labels[u] == labels[v]) ? YY_POS : YY_NEG(labelNumber);
				value -= yy * alphas[u] * kernelValues[u];
			}
			bias[label] += value;
			count[label]++;
		}
	}
	for (label_id l = 0; l < labelNumber; l++) {
		bias[l] /= count[l];
	}

	fvector_free(kernelBuffer);
	return bias;
}

template<typename Kernel, typename Matrix>
fvalue AverageBiasStrategy<Kernel, Matrix>::getBinaryBias(
		vector<label_id>& labels, vector<fvalue>& alphas,
		quantity sampleNumber, label_id label, fvalue rho, fvalue c) {
	fvalue bias = 0.0;
	quantity count = 0;

	fvalue mult[] = {-1.0, 1.0};
	fvector* kernelBuffer = fvector_alloc(sampleNumber);

	for (id v = 0; v < sampleNumber; v++) {
		if (alphas[v] > 0.0) {
			fvalue value = rho - alphas[v] / c;

			// TODO use cache instead
			evaluator->evalInnerKernel(v, 0, sampleNumber, kernelBuffer);
			fvalue* kernelValues = kernelBuffer->data;
			for (id u = 0; u < sampleNumber; u++) {
				fvalue yyuv = mult[labels[v] == labels[u]];
				value -= yyuv * alphas[u] * kernelValues[u];
			}
			fvalue yy = mult[label == labels[v]];
			bias += yy * value;
			count++;
		}
	}

	fvector_free(kernelBuffer);
	return bias / count;
}

template<typename Kernel, typename Matrix>
class NoBiasStrategy: public BiasEvaluationStrategy<Kernel, Matrix> {

public:
	virtual vector<fvalue> getBias(vector<label_id>& labels, vector<fvalue>& alphas,
			quantity labelNumber, quantity sampleNumber, fvalue rho, fvalue c);
	virtual fvalue getBinaryBias(vector<label_id>& labels, vector<fvalue>& alphas,
			quantity sampleNumber, label_id label, fvalue rho, fvalue c);

	virtual ~NoBiasStrategy();

};

template<typename Kernel, typename Matrix>
NoBiasStrategy<Kernel, Matrix>::~NoBiasStrategy() {
}

template<typename Kernel, typename Matrix>
vector<fvalue> NoBiasStrategy<Kernel, Matrix>::getBias(
		vector<label_id>& labels, vector<fvalue>& alphas,
		quantity labelNumber, quantity sampleNumber, fvalue rho, fvalue c) {
	return vector<fvalue>(labelNumber, 0.0);
}

template<typename Kernel, typename Matrix>
fvalue NoBiasStrategy<Kernel, Matrix>::getBinaryBias(
		vector<label_id>& labels, vector<fvalue>& alphas,
		quantity sampleNumber, label_id label, fvalue rho, fvalue c) {
	return 0.0;
}

template<typename Kernel, typename Matrix>
class BiasEvaluatorFactory {

public:
	BiasEvaluationStrategy<Kernel, Matrix>* createEvaluator(BiasType type,
			RbfKernelEvaluator<Kernel, Matrix>* evaluator);

};

template<typename Kernel, typename Matrix>
BiasEvaluationStrategy<Kernel, Matrix>* BiasEvaluatorFactory<Kernel, Matrix>::createEvaluator(
		BiasType type, RbfKernelEvaluator<Kernel, Matrix>* evaluator) {
	BiasEvaluationStrategy<Kernel, Matrix>* eval = NULL;
	if (type == THEORETIC) {
		eval = new TeoreticBiasStrategy<Kernel, Matrix>();
	} else if (type == AVERAGE) {
		eval = new AverageBiasStrategy<Kernel, Matrix>(evaluator);
	} else {
		eval = new NoBiasStrategy<Kernel, Matrix>();
	}
	return eval;
}

#endif
