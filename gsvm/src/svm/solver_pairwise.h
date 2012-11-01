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

template<typename K, typename V>
struct PairValueComparator {

  bool operator() (pair<K, V> pair1, pair<K, V> pair2);

};

template<typename K, typename V>
bool PairValueComparator<K, V>::operator() (pair<K, V> pair1, pair<K, V> pair2) {
	return pair1.second < pair2.second;
}


template<typename Kernel, typename Matrix, typename Strategy>
class PairwiseSolver: public AbstractSolver<Kernel, Matrix, Strategy> {

	label_id maxLabel;
	quantity* labelSizes;
	label_id* labelOrder;

	void sortLabels(label_id *labels, quantity size,
			label_id first, label_id second);

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
	maxLabel = labelNames.size();
	labelSizes = new quantity[maxLabel];
	labelOrder = new label_id[maxLabel];
	for (label_id label = 0; label < maxLabel; label++) {
		labelSizes[label] = 0;
	}
	for (sample_id sample = 0; sample < this->size; sample++) {
		labelSizes[this->labels[sample]]++;
	}
	// sort
	vector<pair<label_id, quantity> > pairs;
	for (label_id label = 0; label < maxLabel; label++) {
		pairs[label] = pair<label_id, quantity>(labelOrder[label], labelSizes[label]);
	}
	PairValueComparator<label_id, quantity> comparator;
	sort(pairs.begin(), pairs.end(), comparator);
	for (label_id label = 0; label < maxLabel; label++) {
		labelOrder[label] = pairs[maxLabel - label - 1].first;
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
PairwiseSolver<Kernel, Matrix, Strategy>::~PairwiseSolver() {
	delete [] labelSizes;
	delete [] labelOrder;
}

template<typename Kernel, typename Matrix, typename Strategy>
CrossClassifier<Kernel, Matrix>* PairwiseSolver<Kernel, Matrix, Strategy>::getClassifier() {
	// XXX create pairwise classifier
	return this->buildClassifier();
}

template<typename Kernel, typename Matrix, typename Strategy>
void PairwiseSolver<Kernel, Matrix, Strategy>::train() {
	// XXX do pairwise training
	for (label_id i = 0; i < maxLabel; i++) {
		label_id label1 = labelOrder[i];
		for (label_id j = i + 1; j < maxLabel; j++) {
			label_id label2 = labelOrder[j];
			sortLabels(this->labels, this->currentSize, label1, label2);
			this->trainForCache(this->cache);
			// TODO save results
		}
	}
}

template<typename Kernel, typename Matrix, typename Strategy>
void PairwiseSolver<Kernel, Matrix, Strategy>::sortLabels(
		label_id* labels, quantity size, label_id first, label_id second) {
	id train = 0;
	id test = size;
	while (train < test) {
		while (labels[train] == first || labels[train] == second) {
			train++;
		}
		while (labels[test] != first && labels[test] != second) {
			test--;
		}
		if (train < test) {
			this->swapSamples(train++, test--);
		}
	}
}

#endif
