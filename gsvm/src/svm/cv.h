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

#ifndef CV_H_
#define CV_H_

#include "solver.h"
#include "../time/timer.h"
#include "../logging/log.h"
#include "../math/random.h"

// comment to disable support vector reuse
#define SUPPORT_VECTOR_REUSE


struct TestingResult {

	double accuracy;

	TestingResult(double accuracy = 0.0) :
			accuracy(accuracy) {
	}

};


template<typename Kernel, typename Matrix, typename Strategy>
class CrossSolver: public Solver<Kernel, Matrix, Strategy> {

protected:
	fold_id *innerFoldsMembership;
	quantity innerFoldsNumber;
	quantity *innerFoldSizes;

	fold_id *outerFoldsMembership;
	quantity outerFoldsNumber;
	quantity *outerFoldSizes;

	fold_id outerFold;

	quantity outerProblemSize;

	void resetInnerFold(fold_id fold);

	TestingResult test(sample_id from, sample_id to);
	TestingResult testInner(fold_id fold);

protected:
	virtual CachedKernelEvaluator<Kernel, Matrix, Strategy>* buildCache(fvalue c, Kernel &gparams);

public:
	CrossSolver(map<label_id, string> labelNames, Matrix *samples, label_id *labels,
			TrainParams &params, StopCriterionStrategy *stopStrategy,
			quantity innerFolds, quantity outerFolds, bool fairFolds = true);
	virtual ~CrossSolver();

	TestingResult doCrossValidation();
	void resetOuterFold(fold_id fold);
	void trainOuter();
	TestingResult testOuter();

	quantity getInnerFoldsNumber() const {
		return innerFoldsNumber;
	}

	quantity getOuterFoldsNumber() const {
		return outerFoldsNumber;
	}

	fold_id getOuterFold() const {
		return outerFold;
	}

	quantity getSvNumber() const {
		return this->cache->getSVNumber();
	}

	quantity getOuterProblemSize() const {
		return outerProblemSize;
	}

private:
	void sortVectors(fold_id *membership, fold_id fold, quantity num);

};

template<typename Kernel, typename Matrix, typename Strategy>
CrossSolver<Kernel, Matrix, Strategy>::CrossSolver(map<label_id, string> labelNames, Matrix *samples,
		label_id *labels, TrainParams &params, StopCriterionStrategy *stopStrategy,
		quantity innerFolds, quantity outerFolds, bool fairFoolds) :
		Solver<Kernel, Matrix, Strategy>(labelNames, samples, labels, params, stopStrategy),
		innerFoldsNumber(innerFolds),
		outerFoldsNumber(outerFolds) {
	innerFoldsMembership = new fold_id[this->size];
	innerFoldSizes = new quantity[innerFolds];
	for (id i = 0; i < innerFolds; i++) {
		innerFoldSizes[i] = this->size;
	}
	outerFoldsMembership = new fold_id[this->size];
	outerFoldSizes = new quantity[outerFolds];
	for (id i = 0; i < outerFolds; i++) {
		outerFoldSizes[i] = this->size;
	}

	if (fairFoolds) {
		quantity labelNum = labelNames.size();
		quantity foldNum = innerFolds * outerFolds;
		id *offsets = new id[labelNum];
		quantity step = innerFolds + 1;
		quantity increase = max(foldNum / labelNum, (quantity) 1);
		for (quantity i = 0; i < labelNum; i++) {
			offsets[i] = (i * increase * step) % foldNum;
		}
		for (id i = 0; i < this->size; i++) {
			label_id label = labels[i];

			id inner = offsets[label] % innerFolds;
			innerFoldsMembership[i] = inner;
			innerFoldSizes[inner]--;

			id outer = offsets[label] / innerFolds;
			outerFoldsMembership[i] = outer;
			if (outerFolds > 1) {
				outerFoldSizes[outer]--;
			}

			offsets[label] = (offsets[label] + step) % foldNum;
		}
		delete [] offsets;
	} else {
		IdGenerator innerGen = Generators::create();
		for (id i = 0; i < this->size; i++) {
			id genId = innerGen.nextId(innerFolds);
			innerFoldsMembership[i] = genId;
			innerFoldSizes[genId]--;
		}

		IdGenerator outerGen = Generators::create();
		for (id i = 0; i < this->size; i++) {
			id genId = outerGen.nextId(outerFolds);
			outerFoldsMembership[i] = genId;
			if (outerFolds > 1) {
				outerFoldSizes[genId]--;
			}
		}
	}

	outerFold = 0;
	outerProblemSize = this->size;
}

template<typename Kernel, typename Matrix, typename Strategy>
CrossSolver<Kernel, Matrix, Strategy>::~CrossSolver() {
	delete [] innerFoldsMembership;
	delete [] outerFoldsMembership;
	delete [] innerFoldSizes;
	delete [] outerFoldSizes;
}

template<typename Kernel, typename Matrix, typename Strategy>
void CrossSolver<Kernel, Matrix, Strategy>::sortVectors(fold_id *membership, fold_id fold, quantity num) {
	id train = 0;
	id test = num - 1;
	while (train < test) {
		while (membership[train] != fold) {
			train++;
		}
		while (membership[test] == fold) {
			test--;
		}
		if (train < test) {
			this->cache->swapSamples(train, test);
			train++;
			test--;
		}
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
void CrossSolver<Kernel, Matrix, Strategy>::resetInnerFold(fold_id fold) {
#ifdef SUPPORT_VECTOR_REUSE
	this->cache->releaseSupportVectors(this->innerFoldsMembership, fold);
	this->cache->shrink();
	sortVectors(this->innerFoldsMembership, fold, outerFoldSizes[outerFold]);
#else
	sortVectors(this->innerFoldsMembership, fold, outerFoldSizes[outerFold]);
	this->cache->reset();
#endif

	this->setCurrentSize(innerFoldSizes[fold]);
}

template<typename Kernel, typename Matrix, typename Strategy>
void CrossSolver<Kernel, Matrix, Strategy>::resetOuterFold(fold_id fold) {
	outerFold = fold;

	sortVectors(this->outerFoldsMembership, fold, this->size);

	for (id i = 0; i < innerFoldsNumber; i++) {
		innerFoldSizes[i] = outerFoldSizes[fold];
	}
	for (id i = 0; i < outerFoldSizes[fold]; i++) {
		id id = innerFoldsMembership[i];
		innerFoldSizes[id]--;
	}

	outerProblemSize = outerFoldSizes[fold];
	this->cache->reset();
}

template<typename Kernel, typename Matrix, typename Strategy>
void CrossSolver<Kernel, Matrix, Strategy>::trainOuter() {
	this->cache->reset();

	this->setCurrentSize(outerFoldSizes[outerFold]);

	this->train();
}

template<typename Kernel, typename Matrix, typename Strategy>
TestingResult CrossSolver<Kernel, Matrix, Strategy>::test(sample_id from, sample_id to) {
	quantity correct = 0;
	CrossClassifier<Kernel, Matrix, Strategy> *classifier = this->buildClassifier();
	for (sample_id test = from; test < to; test++) {
		label_id label = classifier->classify(test);
		if (label == this->labels[test]) {
			correct++;
		}
	}
	delete classifier;
	return TestingResult((double) correct / (to - from));
}

template<typename Kernel, typename Matrix, typename Strategy>
inline TestingResult CrossSolver<Kernel, Matrix, Strategy>::testInner(fold_id fold) {
	return test(this->innerFoldSizes[fold], this->outerFoldSizes[outerFold]);
}

template<typename Kernel, typename Matrix, typename Strategy>
inline TestingResult CrossSolver<Kernel, Matrix, Strategy>::testOuter() {
	return test(this->outerFoldSizes[outerFold], this->size);
}

template<typename Kernel, typename Matrix, typename Strategy>
TestingResult CrossSolver<Kernel, Matrix, Strategy>::doCrossValidation() {
	TestingResult result;

	Timer timer;

	for (fold_id innerFold = 0; innerFold < innerFoldsNumber; innerFold++) {
		resetInnerFold(innerFold);

		timer.restart();
		this->train();
		timer.stop();

		logger << format("inner fold %d/%d training: time=%.2f[s], sv=%d/%d\n")
				% outerFold % innerFold % timer.getTimeElapsed() % getSvNumber() % this->currentSize;

		timer.restart();
		TestingResult foldResult = test(this->innerFoldSizes[innerFold], this->outerFoldSizes[outerFold]);
		timer.stop();

		logger << format("inner fold %d/%d testing: time=%.2f[s], accuracy=%.2f[%%]\n")
				% outerFold % innerFold % timer.getTimeElapsed() % (100.0 * foldResult.accuracy);

		result.accuracy += foldResult.accuracy / innerFoldsNumber;
//		double ratio = (double) (this->outerFoldSizes[outerFold] - this->innerFoldSizes[innerFold])
//				/ this->outerFoldSizes[outerFold];
//		result.accuracy += ratio * foldResult.accuracy;
	}

	return result;
}

template<typename Kernel, typename Matrix, typename Strategy>
CachedKernelEvaluator<Kernel, Matrix, Strategy>* CrossSolver<Kernel, Matrix, Strategy>::buildCache(fvalue c, Kernel &gparams) {
	RbfKernelEvaluator<GaussKernel, Matrix> *rbf = new RbfKernelEvaluator<GaussKernel, Matrix>(
			this->samples, this->labels, this->labelNames.size(), c, gparams);
	SwapListener<Strategy> *listener = new SwapListener<Strategy>(
			innerFoldsMembership, outerFoldsMembership, &this->strategy);
	return new CachedKernelEvaluator<GaussKernel, Matrix, Strategy>(rbf, &this->strategy, this->size, DEFAULT_CACHE_SIZE, listener);
}

#endif
