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

#ifndef SOLVER_H_
#define SOLVER_H_

#include <string>
#include <iostream>

#include <map>
#include <vector>

#include "kernel.h"
#include "classify.h"
#include "cache.h"
#include "stop.h"
#include "params.h"
#include "../math/random.h"
#include "../math/average.h"

#define STARTING_EPSILON 1.0
#define EPSILON_SHRINKING_FACTOR 2.0

#define SHRINKING_LEVEL 100
#define SHRINKING_ITERS 100

#define DEFAULT_C 1.0
#define DEFAULT_GAMMA 1.0

#define DEFAULT_CACHE_SIZE 16000

// compress thresholds (as in libCVM)
#define MAX_REFINE_ITER 20
#define COMPRESS_THRES 400

using namespace std;


template<typename Kernel, typename Matrix>
class Solver {

public:
	virtual ~Solver() {};

	virtual void setKernelParams(fvalue c, Kernel &params) = 0;
	virtual void train() = 0;
	virtual CrossClassifier<Kernel, Matrix>* getClassifier() = 0;

};


template<typename Matrix>
class DataHolder {

public:
	virtual ~DataHolder() {};

	virtual Matrix* getSamples() = 0;
	virtual label_id* getLabels() = 0;
	virtual map<label_id, string>& getLabelNames() = 0;

};


template<typename Matrix>
class StateHolder: public DataHolder<Matrix> {

public:
	virtual ~StateHolder() {};

	virtual void setSwapListener(SwapListener *listener) = 0;
	virtual void swapSamples(sample_id u, sample_id v) = 0;
	virtual void shrink() = 0;
	virtual void releaseSupportVectors(fold_id *membership, fold_id fold) = 0;
	virtual void setCurrentSize(quantity size) = 0;
	virtual quantity getCurrentSize() = 0;
	virtual void reset() = 0;

	virtual Matrix* getSamples() = 0;
	virtual label_id* getLabels() = 0;
	virtual map<label_id, string>& getLabelNames() = 0;

	virtual quantity getSize() = 0;
	virtual quantity getSvNumber() = 0;

};


template<typename Kernel, typename Matrix, typename Strategy>
class UniversalSolver: public Solver<Kernel, Matrix>, public StateHolder<Matrix> {

	TrainParams params;
	StopCriterionStrategy *stopStrategy;

	map<label_id, string> labelNames;

	quantity dimension;
	quantity size;
	quantity currentSize;

	Matrix *samples;
	label_id *labels;

	Strategy strategy;

	SwapListener *listener;
	CachedKernelEvaluator<Kernel, Matrix, Strategy> *cache;

protected:
	sample_id findMinNormViolator(fvalue threshold);

	virtual CachedKernelEvaluator<Kernel, Matrix, Strategy>* buildCache(fvalue c, Kernel &gparams);
	CrossClassifier<Kernel, Matrix>* buildClassifier();

	void refreshDistr();

public:
	UniversalSolver(map<label_id, string> labelNames, Matrix *samples, label_id *labels,
			TrainParams &params, StopCriterionStrategy *stopStrategy);
	virtual ~UniversalSolver();

	void setKernelParams(fvalue c, Kernel &params);
	void train();
	CrossClassifier<Kernel, Matrix>* getClassifier();

	void setSwapListener(SwapListener *listener);
	void swapSamples(sample_id u, sample_id v);
	void shrink();
	void releaseSupportVectors(fold_id *membership, fold_id fold);
	void setCurrentSize(quantity size);
	quantity getCurrentSize();
	void reset();

	Matrix* getSamples();
	label_id* getLabels();
	map<label_id, string>& getLabelNames();

	quantity getSize();
	quantity getSvNumber();
	void reportStatistics();

};

template<typename Kernel, typename Matrix, typename Strategy>
UniversalSolver<Kernel, Matrix, Strategy>::UniversalSolver(map<label_id, string> labelNames, Matrix *samples,
		label_id *labels, TrainParams &params, StopCriterionStrategy *stopStrategy) :
		labelNames(labelNames),
		samples(samples),
		labels(labels),
		strategy(Strategy(params, labelNames.size(), labels, samples->height)),
		params(params),
		stopStrategy(stopStrategy) {
	size = samples->height;
	currentSize = samples->height;
	dimension = samples->width;

	listener = NULL;
	cache = NULL;

	refreshDistr();
}

template<typename Kernel, typename Matrix, typename Strategy>
UniversalSolver<Kernel, Matrix, Strategy>::~UniversalSolver() {
	if (cache) {
		delete cache;
	}
	delete samples;
	delete [] labels;
	delete stopStrategy;
}

template<typename Kernel, typename Matrix, typename Strategy>
void UniversalSolver<Kernel, Matrix, Strategy>::setKernelParams(fvalue c, Kernel &gparams) {
	if (cache == NULL) {
		cache = buildCache(c, gparams);
		cache->setSwapListener(listener);
	} else {
		cache->setKernelParams(c, gparams);
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
sample_id UniversalSolver<Kernel, Matrix, Strategy>::findMinNormViolator(fvalue threshold) {
	quantity attempt = 0;
	while (attempt < params.drawNumber) {
		sample_id violator = strategy.generateNextId();
		if (cache->checkViolation(violator, threshold)) {
			strategy.markGeneratedIdAsCorrect();
			return violator;
		}
		strategy.markGeneratedIdAsFailed();
		attempt++;
	}
	return INVALID_ID;
}

template<typename Kernel, typename Matrix, typename Strategy>
void UniversalSolver<Kernel, Matrix, Strategy>::reportStatistics() {
	cache->reportStatistics();
}

template<typename Kernel, typename Matrix, typename Strategy>
CrossClassifier<Kernel, Matrix>* UniversalSolver<Kernel, Matrix, Strategy>::getClassifier() {
	return buildClassifier();
}

template<typename Kernel, typename Matrix, typename Strategy>
void UniversalSolver<Kernel, Matrix, Strategy>::train() {
	sample_id mnviol = INVALID_ID;
	fvalue tau = cache->getTau();
	fvalue finalEpsilon = params.epsilon;
	if (finalEpsilon <= 0) {
		finalEpsilon = stopStrategy->getEpsilon(cache->getTau(), cache->getC());
	}
	fvalue epsilon = STARTING_EPSILON;
//	quantity minVal = currentSize / SHRINKING_LEVEL;
	do {
		epsilon /= EPSILON_SHRINKING_FACTOR;
		if (epsilon < finalEpsilon) {
			epsilon = finalEpsilon;
		}

		do {
			// main update
			fvalue threshold = stopStrategy->getThreshold(cache->getWNorm(), tau, epsilon);
			mnviol = findMinNormViolator(threshold);
			if (mnviol != INVALID_ID) {
				sample_id kktviol = cache->findMaxSVKernelVal(mnviol);

				cache->performUpdate(kktviol, mnviol);
			}

			// shrinking
			if (cache->getSVNumber() > COMPRESS_THRES) {
				for (int i = 0; i < MAX_REFINE_ITER; i++) {
					cache->performSvUpdate(tau, 0);
				}
			}

//			// TODO improve shrinking procedure
//			fvalue threshold;
//			quantity iter = 0;
//			do {
//				threshold = stopStrategy->getThreshold(cache->getWNorm(), cache->getTau(), currentEpsilon);
//				iter++;
//			} while (cache->performSvUpdate(threshold, minVal) && iter < SHRINKING_ITERS);
		} while (mnviol != INVALID_ID);
	} while (epsilon > finalEpsilon);
}

template<typename Kernel, typename Matrix, typename Strategy>
CachedKernelEvaluator<Kernel, Matrix, Strategy>* UniversalSolver<Kernel, Matrix, Strategy>::buildCache(fvalue c, Kernel &gparams) {
	RbfKernelEvaluator<GaussKernel, Matrix> *rbf = new RbfKernelEvaluator<GaussKernel, Matrix>(
			this->samples, this->labels, labelNames.size(), c, gparams);
	return new CachedKernelEvaluator<GaussKernel, Matrix, Strategy>(rbf, &strategy, size, DEFAULT_CACHE_SIZE);
}

template<typename Kernel, typename Matrix, typename Strategy>
CrossClassifier<Kernel, Matrix>* UniversalSolver<Kernel, Matrix, Strategy>::buildClassifier() {
	fvector *buffer = cache->getBuffer();
	buffer->size = cache->getSVNumber();
	return new CrossClassifier<Kernel, Matrix>(cache->getEvaluator(),
			cache->getAlphas(), labels, buffer, labelNames.size(),
			cache->getSVNumber());
}

template<typename Kernel, typename Matrix, typename Strategy>
Matrix* UniversalSolver<Kernel, Matrix, Strategy>::getSamples() {
	return samples;
}

template<typename Kernel, typename Matrix, typename Strategy>
label_id* UniversalSolver<Kernel, Matrix, Strategy>::getLabels() {
	return labels;
}

template<typename Kernel, typename Matrix, typename Strategy>
map<label_id, string>& UniversalSolver<Kernel, Matrix, Strategy>::getLabelNames() {
	return labelNames;
}

template<typename Kernel, typename Matrix, typename Strategy>
void UniversalSolver<Kernel, Matrix, Strategy>::refreshDistr() {
	this->strategy.resetGenerator(labels, currentSize);
}

template<typename Kernel, typename Matrix, typename Strategy>
void UniversalSolver<Kernel, Matrix, Strategy>::setSwapListener(SwapListener *listener) {
	this->listener = listener;
	if (cache) {
		cache->setSwapListener(listener);
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
void UniversalSolver<Kernel, Matrix, Strategy>::swapSamples(sample_id u, sample_id v) {
	cache->swapSamples(u, v);
}

template<typename Kernel, typename Matrix, typename Strategy>
void UniversalSolver<Kernel, Matrix, Strategy>::reset() {
	cache->reset();
}

template<typename Kernel, typename Matrix, typename Strategy>
void UniversalSolver<Kernel, Matrix, Strategy>::shrink() {
	cache->shrink();
}

template<typename Kernel, typename Matrix, typename Strategy>
void UniversalSolver<Kernel, Matrix, Strategy>::releaseSupportVectors(fold_id *membership, fold_id fold) {
	cache->releaseSupportVectors(membership, fold);
}

template<typename Kernel, typename Matrix, typename Strategy>
void UniversalSolver<Kernel, Matrix, Strategy>::setCurrentSize(quantity size) {
	currentSize = size;
	refreshDistr();
}

template<typename Kernel, typename Matrix, typename Strategy>
inline quantity UniversalSolver<Kernel, Matrix, Strategy>::getCurrentSize() {
	return currentSize;
}

template<typename Kernel, typename Matrix, typename Strategy>
quantity UniversalSolver<Kernel, Matrix, Strategy>::getSize() {
	return size;
}

template<typename Kernel, typename Matrix, typename Strategy>
quantity UniversalSolver<Kernel, Matrix, Strategy>::getSvNumber() {
	return this->cache->getSVNumber();
}

#endif
