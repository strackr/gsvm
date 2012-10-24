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


template<typename Kernel, typename Matrix, typename Strategy>
class Solver {

protected:
	map<label_id, string> labelNames;

	quantity dimension;
	quantity size;
	quantity currentSize;

	Matrix *samples;
	label_id *labels;

	Strategy strategy;

	CachedKernelEvaluator<Kernel, Matrix, Strategy> *cache;

	TrainParams params;

	fvalue currentEpsilon;

	StopCriterionStrategy *stopStrategy;

protected:
	sample_id findMinNormViolator();

	virtual CachedKernelEvaluator<Kernel, Matrix, Strategy>* buildCache(fvalue c, Kernel &gparams);
	CrossClassifier<Kernel, Matrix, Strategy>* buildClassifier();

	void train();
	void shrink();

	void setCurrentSize(quantity size);
	void refreshDistr();

public:
	Solver(map<label_id, string> labelNames, Matrix *samples, label_id *labels,
			TrainParams &params, StopCriterionStrategy *stopStrategy);
	virtual ~Solver();

	void setKernelParams(fvalue c, Kernel &params);

	CrossClassifier<Kernel, Matrix, Strategy>* getClassifier();

	Matrix* getSamples();
	label_id* getLabels();

	void reportStatistics();

};

template<typename Kernel, typename Matrix, typename Strategy>
Solver<Kernel, Matrix, Strategy>::Solver(map<label_id, string> labelNames, Matrix *samples,
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

	cache = NULL;
	currentEpsilon = DEFAULT_EPSILON;

	refreshDistr();
}

template<typename Kernel, typename Matrix, typename Strategy>
Solver<Kernel, Matrix, Strategy>::~Solver() {
	if (cache) {
		delete cache;
	}
	delete samples;
	delete [] labels;
	delete stopStrategy;
}

template<typename Kernel, typename Matrix, typename Strategy>
void Solver<Kernel, Matrix, Strategy>::setKernelParams(fvalue c, Kernel &gparams) {
	if (cache == NULL) {
		cache = buildCache(c, gparams);
	} else {
		cache->setKernelParams(c, gparams);
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
sample_id Solver<Kernel, Matrix, Strategy>::findMinNormViolator() {
	quantity attempt = 0;
	fvalue threshold = stopStrategy->getThreshold(cache->getWNorm(), cache->getTau(), currentEpsilon);
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
void Solver<Kernel, Matrix, Strategy>::reportStatistics() {
	cache->reportStatistics();
}

template<typename Kernel, typename Matrix, typename Strategy>
CrossClassifier<Kernel, Matrix, Strategy>* Solver<Kernel, Matrix, Strategy>::getClassifier() {
	train();
	return buildClassifier();
}

template<typename Kernel, typename Matrix, typename Strategy>
void Solver<Kernel, Matrix, Strategy>::train() {
	sample_id mnviol = INVALID_ID;
	fvalue finalEpsilon = params.epsilon;
	if (finalEpsilon <= 0) {
		finalEpsilon = stopStrategy->getEpsilon(cache->getTau(), cache->getC());
	}
	currentEpsilon = STARTING_EPSILON;
//	quantity minVal = currentSize / SHRINKING_LEVEL;
	do {
		currentEpsilon /= EPSILON_SHRINKING_FACTOR;
		if (currentEpsilon < finalEpsilon) {
			currentEpsilon = finalEpsilon;
		}

		do {
			// main update
			mnviol = findMinNormViolator();
			if (mnviol != INVALID_ID) {
				sample_id kktviol = cache->findMaxSVKernelVal(mnviol);

				cache->performUpdate(kktviol, mnviol);
			}

			// shrinking
			if (cache->getSVNumber() > COMPRESS_THRES) {
				for (int i = 0; i < MAX_REFINE_ITER; i++) {
					cache->performSvUpdate(cache->getTau(), 0);
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
	} while (currentEpsilon > finalEpsilon);
}

template<typename Kernel, typename Matrix, typename Strategy>
void Solver<Kernel, Matrix, Strategy>::shrink() {
	cache->shrink();
}

template<typename Kernel, typename Matrix, typename Strategy>
CachedKernelEvaluator<Kernel, Matrix, Strategy>* Solver<Kernel, Matrix, Strategy>::buildCache(fvalue c, Kernel &gparams) {
	RbfKernelEvaluator<GaussKernel, Matrix> *rbf = new RbfKernelEvaluator<GaussKernel, Matrix>(
			this->samples, this->labels, labelNames.size(), c, gparams);
	return new CachedKernelEvaluator<GaussKernel, Matrix, Strategy>(rbf, &strategy, size, DEFAULT_CACHE_SIZE);
}

template<typename Kernel, typename Matrix, typename Strategy>
CrossClassifier<Kernel, Matrix, Strategy>* Solver<Kernel, Matrix, Strategy>::buildClassifier() {
	fvector *buffer = cache->getBuffer();
	buffer->size = cache->getSVNumber();
	return new CrossClassifier<Kernel, Matrix, Strategy>(cache->getEvaluator(),
			cache->getAlphas(), labels, buffer, labelNames.size(),
			cache->getSVNumber());
}

template<typename Kernel, typename Matrix, typename Strategy>
Matrix* Solver<Kernel, Matrix, Strategy>::getSamples() {
	return samples;
}

template<typename Kernel, typename Matrix, typename Strategy>
label_id* Solver<Kernel, Matrix, Strategy>::getLabels() {
	return labels;
}

template<typename Kernel, typename Matrix, typename Strategy>
void Solver<Kernel, Matrix, Strategy>::setCurrentSize(quantity size) {
	currentSize = size;
	refreshDistr();
}

template<typename Kernel, typename Matrix, typename Strategy>
void Solver<Kernel, Matrix, Strategy>::refreshDistr() {
	this->strategy.resetGenerator(labels, currentSize);
}

#endif
