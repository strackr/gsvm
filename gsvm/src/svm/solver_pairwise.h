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

#ifndef SOLVER_PAIRWISE_H_
#define SOLVER_PAIRWISE_H_

#include "solver_common.h"

template<typename Kernel, typename Matrix, typename Strategy>
class PairwiseSolver: public AbstractSolver<Kernel, Matrix, Strategy> {

public:
	PairwiseSolver(map<label_id, string> labelNames, Matrix *samples,
			label_id *labels, TrainParams &params,
			StopCriterionStrategy *stopStrategy);
	virtual ~PairwiseSolver();

	void train();
	CrossClassifier<Kernel, Matrix>* getClassifier();

};

template<typename Kernel, typename Matrix, typename Strategy>
PairwiseSolver<Kernel, Matrix, Strategy>::PairwiseSolver(
		map<label_id, string> labelNames, Matrix *samples,
		label_id *labels, TrainParams &params,
		StopCriterionStrategy *stopStrategy) :
		AbstractSolver<Kernel, Matrix, Strategy>(labelNames, samples, labels, params, stopStrategy) {
}

template<typename Kernel, typename Matrix, typename Strategy>
PairwiseSolver<Kernel, Matrix, Strategy>::~PairwiseSolver() {
}

template<typename Kernel, typename Matrix, typename Strategy>
CrossClassifier<Kernel, Matrix>* PairwiseSolver<Kernel, Matrix, Strategy>::getClassifier() {
	// XXX create pairwise classifier
	return this->buildClassifier();
}

template<typename Kernel, typename Matrix, typename Strategy>
void PairwiseSolver<Kernel, Matrix, Strategy>::train() {
	// XXX do pairwise training
	this->trainForCache(this->cache);
}

#endif
