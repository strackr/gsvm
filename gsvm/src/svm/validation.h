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

#include "solver_universal.h"
#include "solver_pairwise.h"
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


class CrossSolverSwapListener: public SwapListener {

	fold_id *innerMembership;
	fold_id *outerMembership;

public:
	CrossSolverSwapListener(fold_id *innerMembership, fold_id *outerMembership);

	void notify(sample_id u, sample_id v);

};

inline CrossSolverSwapListener::CrossSolverSwapListener(
		fold_id *innerMembership, fold_id *outerMembership) :
		innerMembership(innerMembership),
		outerMembership(outerMembership) {
}

inline void CrossSolverSwapListener::notify(sample_id u, sample_id v) {
	swap(innerMembership[u], innerMembership[v]);
	swap(outerMembership[u], outerMembership[v]);
}




template<typename Kernel, typename Matrix, typename Strategy>
class CrossValidationSolver: public Solver<Kernel, Matrix>, public DataHolder<Matrix> {

	AbstractSolver<Kernel, Matrix, Strategy> *solver;

	fold_id *innerFoldsMembership;
	quantity innerFoldsNumber;
	quantity *innerFoldSizes;

	fold_id *outerFoldsMembership;
	quantity outerFoldsNumber;
	quantity *outerFoldSizes;

	fold_id outerFold;

	quantity outerProblemSize;

protected:
	void resetInnerFold(fold_id fold);

	TestingResult test(sample_id from, sample_id to);
	TestingResult testInner(fold_id fold);

	void sortVectors(fold_id *membership, fold_id fold, quantity num);

public:
	CrossValidationSolver(AbstractSolver<Kernel, Matrix, Strategy> *solver,
			quantity innerFolds, quantity outerFolds, bool fairFolds = true);
	virtual ~CrossValidationSolver();

	void setKernelParams(fvalue c, Kernel &params);
	void train();
	UniversalClassifier<Kernel, Matrix>* getClassifier();

	TestingResult doCrossValidation();
	void resetOuterFold(fold_id fold);
	void trainOuter();
	TestingResult testOuter();

	StateHolder<Matrix>& getStateHolder();

	quantity getInnerFoldsNumber();
	quantity getOuterFoldsNumber();
	fold_id getOuterFold();
	quantity getOuterProblemSize();

	Matrix* getSamples();
	label_id* getLabels();
	map<label_id, string>& getLabelNames();

};

template<typename Kernel, typename Matrix, typename Strategy>
CrossValidationSolver<Kernel, Matrix, Strategy>::CrossValidationSolver(
		AbstractSolver<Kernel, Matrix, Strategy> *solver,
		quantity innerFolds, quantity outerFolds, bool fairFoolds) :
		solver(solver),
		innerFoldsNumber(innerFolds),
		outerFoldsNumber(outerFolds) {
	quantity size = solver->getSize();
	label_id *labels = solver->getLabels();
	map<label_id, string> &labelNames = solver->getLabelNames();

	innerFoldsMembership = new fold_id[size];
	innerFoldSizes = new quantity[innerFolds];
	for (id i = 0; i < innerFolds; i++) {
		innerFoldSizes[i] = size;
	}
	outerFoldsMembership = new fold_id[size];
	outerFoldSizes = new quantity[outerFolds];
	for (id i = 0; i < outerFolds; i++) {
		outerFoldSizes[i] = size;
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
		for (id i = 0; i < size; i++) {
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
		for (id i = 0; i < size; i++) {
			id genId = innerGen.nextId(innerFolds);
			innerFoldsMembership[i] = genId;
			innerFoldSizes[genId]--;
		}

		IdGenerator outerGen = Generators::create();
		for (id i = 0; i < size; i++) {
			id genId = outerGen.nextId(outerFolds);
			outerFoldsMembership[i] = genId;
			if (outerFolds > 1) {
				outerFoldSizes[genId]--;
			}
		}
	}

	outerFold = 0;
	outerProblemSize = size;

	SwapListener *listener = new CrossSolverSwapListener(
			innerFoldsMembership, outerFoldsMembership);
	solver->setSwapListener(listener);
}

template<typename Kernel, typename Matrix, typename Strategy>
CrossValidationSolver<Kernel, Matrix, Strategy>::~CrossValidationSolver() {
	delete solver;
	delete [] innerFoldsMembership;
	delete [] outerFoldsMembership;
	delete [] innerFoldSizes;
	delete [] outerFoldSizes;
}

template<typename Kernel, typename Matrix, typename Strategy>
void CrossValidationSolver<Kernel, Matrix, Strategy>::sortVectors(
		fold_id *membership, fold_id fold, quantity num) {
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
			solver->swapSamples(train, test);
			train++;
			test--;
		}
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
void CrossValidationSolver<Kernel, Matrix, Strategy>::resetInnerFold(
		fold_id fold) {
#ifdef SUPPORT_VECTOR_REUSE
	solver->releaseSupportVectors(this->innerFoldsMembership, fold);
	solver->shrink();
	sortVectors(this->innerFoldsMembership, fold, outerFoldSizes[outerFold]);
#else
	sortVectors(this->innerFoldsMembership, fold, outerFoldSizes[outerFold]);
	solver->reset();
#endif

	solver->setCurrentSize(innerFoldSizes[fold]);
}

template<typename Kernel, typename Matrix, typename Strategy>
void CrossValidationSolver<Kernel, Matrix, Strategy>::resetOuterFold(
		fold_id fold) {
	outerFold = fold;

	sortVectors(this->outerFoldsMembership, fold, solver->getSize());

	for (id i = 0; i < innerFoldsNumber; i++) {
		innerFoldSizes[i] = outerFoldSizes[fold];
	}
	for (id i = 0; i < outerFoldSizes[fold]; i++) {
		id id = innerFoldsMembership[i];
		innerFoldSizes[id]--;
	}

	outerProblemSize = outerFoldSizes[fold];
	solver->reset();
}

template<typename Kernel, typename Matrix, typename Strategy>
void CrossValidationSolver<Kernel, Matrix, Strategy>::trainOuter() {
	solver->reset();

	solver->setCurrentSize(outerFoldSizes[outerFold]);

	this->train();
}

template<typename Kernel, typename Matrix, typename Strategy>
TestingResult CrossValidationSolver<Kernel, Matrix, Strategy>::test(
		sample_id from, sample_id to) {
	label_id *labels = solver->getLabels();
	quantity correct = 0;
	UniversalClassifier<Kernel, Matrix> *classifier = solver->getClassifier();
	for (sample_id test = from; test < to; test++) {
		label_id label = classifier->classify(test);
		if (label == labels[test]) {
			correct++;
		}
	}
	delete classifier;
	return TestingResult((double) correct / (to - from));
}

template<typename Kernel, typename Matrix, typename Strategy>
inline TestingResult CrossValidationSolver<Kernel, Matrix, Strategy>::testInner(
		fold_id fold) {
	return test(this->innerFoldSizes[fold], this->outerFoldSizes[outerFold]);
}

template<typename Kernel, typename Matrix, typename Strategy>
inline TestingResult CrossValidationSolver<Kernel, Matrix, Strategy>::testOuter() {
	return test(this->outerFoldSizes[outerFold], solver->getSize());
}

template<typename Kernel, typename Matrix, typename Strategy>
TestingResult CrossValidationSolver<Kernel, Matrix, Strategy>::doCrossValidation() {
	TestingResult result;

	Timer timer;

	for (fold_id innerFold = 0; innerFold < innerFoldsNumber; innerFold++) {
		resetInnerFold(innerFold);

		timer.restart();
		solver->train();
		timer.stop();

		logger << format("inner fold %d/%d training: time=%.2f[s], sv=%d/%d\n")
				% outerFold % innerFold % timer.getTimeElapsed()
				% solver->getSvNumber() % solver->getCurrentSize();

		timer.restart();
		TestingResult foldResult = test(
				this->innerFoldSizes[innerFold], this->outerFoldSizes[outerFold]);
		timer.stop();

		logger << format("inner fold %d/%d testing: time=%.2f[s], accuracy=%.2f[%%]\n")
				% outerFold % innerFold
				% timer.getTimeElapsed() % (100.0 * foldResult.accuracy);

		result.accuracy += foldResult.accuracy / innerFoldsNumber;
//		double ratio = (double) (this->outerFoldSizes[outerFold] - this->innerFoldSizes[innerFold])
//				/ this->outerFoldSizes[outerFold];
//		result.accuracy += ratio * foldResult.accuracy;
	}

	return result;
}

template<typename Kernel, typename Matrix, typename Strategy>
quantity CrossValidationSolver<Kernel, Matrix, Strategy>::getInnerFoldsNumber() {
	return innerFoldsNumber;
}

template<typename Kernel, typename Matrix, typename Strategy>
quantity CrossValidationSolver<Kernel, Matrix, Strategy>::getOuterFoldsNumber() {
	return outerFoldsNumber;
}

template<typename Kernel, typename Matrix, typename Strategy>
fold_id CrossValidationSolver<Kernel, Matrix, Strategy>::getOuterFold() {
	return outerFold;
}

template<typename Kernel, typename Matrix, typename Strategy>
quantity CrossValidationSolver<Kernel, Matrix, Strategy>::getOuterProblemSize() {
	return outerProblemSize;
}

template<typename Kernel, typename Matrix, typename Strategy>
inline Matrix* CrossValidationSolver<Kernel, Matrix, Strategy>::getSamples() {
	return solver->getSamples();
}

template<typename Kernel, typename Matrix, typename Strategy>
inline label_id* CrossValidationSolver<Kernel, Matrix, Strategy>::getLabels() {
	return solver->getLabels();
}

template<typename Kernel, typename Matrix, typename Strategy>
inline map<label_id, string>& CrossValidationSolver<Kernel, Matrix, Strategy>::getLabelNames() {
	return solver->getLabelNames();
}

template<typename Kernel, typename Matrix, typename Strategy>
inline StateHolder<Matrix>& CrossValidationSolver<Kernel, Matrix, Strategy>::getStateHolder() {
	return *solver;
}

template<typename Kernel, typename Matrix, typename Strategy>
inline void CrossValidationSolver<Kernel, Matrix, Strategy>::setKernelParams(
		fvalue c, Kernel& params) {
	solver->setKernelParams(c, params);
}

template<typename Kernel, typename Matrix, typename Strategy>
inline void CrossValidationSolver<Kernel, Matrix, Strategy>::train() {
	solver->train();
}

template<typename Kernel, typename Matrix, typename Strategy>
inline UniversalClassifier<Kernel, Matrix>* CrossValidationSolver<Kernel, Matrix, Strategy>::getClassifier() {
	return solver->getClassifier();
}

#endif
